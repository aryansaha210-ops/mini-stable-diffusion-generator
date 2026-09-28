import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

from deterministic_tokenizer import DeterministicTokenizer


# ============================================
# CONFIG
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 5
LR = 2e-4

TIMESTEPS = 300
TEXT_DIM = 64

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================
# VAE
# ============================================

class VAE(nn.Module):

    def __init__(self, latent_dim=16):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(
            128 * 4 * 4,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            128 * 4 * 4,
            latent_dim
        )

        self.fc_decode = nn.Linear(
            latent_dim,
            128 * 4 * 4
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, 4, 2, 1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32, 4, 2, 1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 1, 4, 2, 1
            ),
            nn.Tanh()
        )


# ============================================
# TEXT ENCODER
# ============================================

class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            10000,
            TEXT_DIM
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TEXT_DIM,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        return self.encoder(x)


# ============================================
# CROSS ATTENTION
# ============================================

class CrossAttention(nn.Module):

    def __init__(self):

        super().__init__()

        # U-Net middle feature has 128 channels
        self.query = nn.Linear(
            128,
            64
        )

        # Text embedding has 64 dimensions
        self.key = nn.Linear(
            64,
            64
        )

        self.value = nn.Linear(
            64,
            64
        )

        # Convert attention output back to
        # 128 channels for the U-Net
        self.output = nn.Linear(
            64,
            128
        )

    def forward(
        self,
        latent,
        text
    ):

        # latent:
        # [B, H, W, 128]

        # text:
        # [B, tokens, 64]

        B, H, W, C = latent.shape

        latent_flat = latent.reshape(
            B,
            H * W,
            C
        )

        # Query from spatial latent
        Q = self.query(
            latent_flat
        )

        # Key and value from text
        K = self.key(
            text
        )

        V = self.value(
            text
        )

        # Attention scores
        scores = torch.matmul(
            Q,
            K.transpose(
                -2,
                -1
            )
        )

        scores = scores / math.sqrt(64)

        attention = torch.softmax(
            scores,
            dim=-1
        )

        # Apply attention
        output = torch.matmul(
            attention,
            V
        )

        # Convert 64 -> 128 channels
        output = self.output(
            output
        )

        # Restore spatial dimensions
        output = output.reshape(
            B,
            H,
            W,
            128
        )

        return output


# ============================================
# TIME EMBEDDING
# ============================================

class TimeEmbedding(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                1,
                128
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                128
            )
        )

    def forward(self, t):

        t = t.float().unsqueeze(1)

        t = t / TIMESTEPS

        return self.network(t)


# ============================================
# RESIDUAL BLOCK
# ============================================

class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1
        )

        self.time_projection = nn.Linear(
            128,
            out_channels
        )

        self.activation = nn.ReLU()

        if in_channels != out_channels:

            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                1
            )

        else:

            self.skip = nn.Identity()

    def forward(
        self,
        x,
        time_embedding
    ):

        residual = self.skip(x)

        x = self.conv1(x)

        x = self.activation(x)

        # Inject timestep information
        time = self.time_projection(
            time_embedding
        )

        time = time[:, :, None, None]

        x = x + time

        x = self.conv2(x)

        x = self.activation(x)

        return x + residual


# ============================================
# SPATIAL TEXT U-NET
# ============================================

class SpatialTextUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Input:
        # [B, 1, 16, 16]

        self.input = nn.Conv2d(
            1,
            64,
            3,
            padding=1
        )

        self.time_embedding = TimeEmbedding()

        # 64 -> 128
        self.res1 = ResBlock(
            64,
            128
        )

        # 16x16 -> 8x8
        self.down = nn.Conv2d(
            128,
            128,
            4,
            2,
            1
        )

        self.middle = ResBlock(
            128,
            128
        )

        # Text cross-attention
        self.cross_attention = CrossAttention()

        # 8x8 -> 16x16
        self.up = nn.ConvTranspose2d(
            128,
            64,
            4,
            2,
            1
        )

        self.res2 = ResBlock(
            64,
            64
        )

        self.output = nn.Conv2d(
            64,
            1,
            3,
            padding=1
        )

    def forward(
        self,
        x,
        t,
        text
    ):

        # Time embedding
        time = self.time_embedding(t)

        # Input projection
        x = self.input(x)

        # First residual block
        x = self.res1(
            x,
            time
        )

        # Downsample
        x = self.down(x)

        # Middle residual block
        x = self.middle(
            x,
            time
        )

        # --------------------------------
        # Cross Attention
        # --------------------------------

        # [B,128,H,W]
        # ->
        # [B,H,W,128]

        x_attention = x.permute(
            0,
            2,
            3,
            1
        )

        text_condition = self.cross_attention(
            x_attention,
            text
        )

        # [B,H,W,128]
        # ->
        # [B,128,H,W]

        text_condition = text_condition.permute(
            0,
            3,
            1,
            2
        )

        # Add text information
        x = x + text_condition

        # Upsample
        x = self.up(x)

        # Residual block
        x = self.res2(
            x,
            time
        )

        # Predict noise
        x = self.output(x)

        return x


# ============================================
# DIFFUSION SCHEDULE
# ============================================

betas = torch.linspace(
    1e-4,
    0.02,
    TIMESTEPS,
    device=DEVICE
)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(
    alphas,
    dim=0
)


# ============================================
# DATASET
# ============================================

transform = transforms.Compose([
    transforms.Resize(
        (32, 32)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================
# LOAD VAE
# ============================================

vae = VAE(
    latent_dim=16
).to(DEVICE)

vae.load_state_dict(
    torch.load(
        "checkpoints/vae_epoch_5.pth",
        map_location=DEVICE
    )
)

vae.eval()

for parameter in vae.parameters():

    parameter.requires_grad = False


print()
print("VAE loaded successfully!")


# ============================================
# MODELS
# ============================================

text_encoder = TextEncoder().to(
    DEVICE
)

unet = SpatialTextUNet().to(
    DEVICE
)

tokenizer = DeterministicTokenizer()


optimizer = torch.optim.AdamW(
    list(text_encoder.parameters())
    +
    list(unet.parameters()),
    lr=LR
)


# ============================================
# DIGIT WORDS
# ============================================

digit_words = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine"
]


# ============================================
# CREATE CAPTIONS
# ============================================

def create_captions(labels):

    captions = []

    for label in labels.tolist():

        word = digit_words[label]

        caption = (
            f"a handwritten digit {word}"
        )

        captions.append(caption)

    return captions


# ============================================
# TRAINING
# ============================================

print()
print(
    "Starting deterministic "
    "spatial text diffusion training..."
)

for epoch in range(EPOCHS):

    total_loss = 0.0

    for images, labels in loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        # ====================================
        # VAE ENCODING
        # ====================================

        with torch.no_grad():

            encoded = vae.encoder(
                images
            )

            encoded = encoded.reshape(
                images.size(0),
                -1
            )

            latent = vae.fc_mu(
                encoded
            )

        # latent:
        # [B,16]

        # Convert to spatial representation
        latent = latent.reshape(
            images.size(0),
            1,
            4,
            4
        )

        # Upsample:
        # [B,1,4,4]
        # ->
        # [B,1,16,16]

        latent = torch.nn.functional.interpolate(
            latent,
            size=(16, 16),
            mode="bilinear",
            align_corners=False
        )

        # ====================================
        # RANDOM DIFFUSION TIMESTEP
        # ====================================

        t = torch.randint(
            0,
            TIMESTEPS,
            (
                images.size(0),
            ),
            device=DEVICE
        )

        # Random Gaussian noise
        noise = torch.randn_like(
            latent
        )

        alpha_bar = alpha_bars[t]

        alpha_bar = alpha_bar.reshape(
            -1,
            1,
            1,
            1
        )

        # Forward diffusion
        noisy_latent = (
            torch.sqrt(alpha_bar)
            * latent
            +
            torch.sqrt(1.0 - alpha_bar)
            * noise
        )

        # ====================================
        # CREATE TEXT CAPTIONS
        # ====================================

        captions = create_captions(
            labels
        )

        # Tokenize using deterministic
        # vocabulary

        token_ids = torch.tensor(
            [
                tokenizer.tokenize(
                    caption
                )
                for caption in captions
            ],
            dtype=torch.long,
            device=DEVICE
        )

        # ====================================
        # TEXT ENCODING
        # ====================================

        text_features = text_encoder(
            token_ids
        )

        # Shape:
        # [B,16,64]

        # ====================================
        # PREDICT NOISE
        # ====================================

        predicted_noise = unet(
            noisy_latent,
            t,
            text_features
        )

        # ====================================
        # LOSS
        # ====================================

        loss = torch.mean(
            (
                predicted_noise
                -
                noise
            ) ** 2
        )

        # ====================================
        # BACKPROPAGATION
        # ====================================

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    # ====================================
    # EPOCH RESULT
    # ====================================

    average_loss = (
        total_loss
        /
        len(loader)
    )

    print(
        f"Epoch {epoch + 1} completed "
        f"| Average Loss: "
        f"{average_loss:.4f}"
    )

    # ====================================
    # SAVE CHECKPOINT
    # ====================================

    checkpoint_path = (
        "checkpoints/"
        "spatial_text_deterministic_epoch_"
        f"{epoch + 1}.pth"
    )

    torch.save(
        {
            "text_encoder":
                text_encoder.state_dict(),

            "unet":
                unet.state_dict()
        },
        checkpoint_path
    )


# ============================================
# COMPLETE
# ============================================

print()
print(
    "===================================="
)

print(
    "DETERMINISTIC SPATIAL TEXT "
    "DIFFUSION TRAINING DONE"
)

print(
    "===================================="
)