import re
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================
# CONFIG
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 5
LR = 2e-4

TIMESTEPS = 300

LATENT_CHANNELS = 4

TEXT_DIM = 128
VOCAB_SIZE = 10000
MAX_TOKENS = 16


print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================
# DETERMINISTIC TOKENIZER
# ============================================

class DeterministicTokenizer:

    def __init__(self):

        self.PAD = 0
        self.UNK = 1
        self.START = 2
        self.END = 3

        words = [
            "a",
            "handwritten",
            "digit",
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

        self.vocab = {
            word: index + 10
            for index, word in enumerate(words)
        }

    def tokenize(self, text):

        words = re.findall(
            r"[a-z0-9]+",
            text.lower()
        )

        token_ids = [self.START]

        for word in words:

            token_ids.append(
                self.vocab.get(
                    word,
                    self.UNK
                )
            )

        token_ids.append(self.END)

        token_ids = token_ids[
            :MAX_TOKENS
        ]

        while len(token_ids) < MAX_TOKENS:

            token_ids.append(
                self.PAD
            )

        return token_ids


tokenizer = DeterministicTokenizer()


# ============================================
# SPATIAL VAE
# ============================================

class SpatialVAE(nn.Module):

    def __init__(
        self,
        latent_channels=4
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        self.mu = nn.Conv2d(
            128,
            latent_channels,
            kernel_size=3,
            padding=1
        )

        self.logvar = nn.Conv2d(
            128,
            latent_channels,
            kernel_size=3,
            padding=1
        )

        self.decoder = nn.Sequential(

            nn.Conv2d(
                latent_channels,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                padding=1
            ),

            nn.Tanh()
        )


    def encode(self, x):

        features = self.encoder(x)

        mu = self.mu(features)

        logvar = self.logvar(features)

        return mu, logvar


# ============================================
# TEXT ENCODER
# ============================================

class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            TEXT_DIM
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=TEXT_DIM,
                nhead=4,
                dim_feedforward=256,
                batch_first=True
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )


    def forward(self, tokens):

        x = self.embedding(tokens)

        x = self.encoder(x)

        return x


# ============================================
# CROSS ATTENTION
# ============================================

class CrossAttention(nn.Module):

    def __init__(
        self,
        latent_channels=128,
        text_dim=128,
        heads=4
    ):

        super().__init__()

        self.heads = heads

        self.head_dim = (
            latent_channels // heads
        )

        self.query = nn.Linear(
            latent_channels,
            latent_channels
        )

        self.key = nn.Linear(
            text_dim,
            latent_channels
        )

        self.value = nn.Linear(
            text_dim,
            latent_channels
        )

        self.output = nn.Linear(
            latent_channels,
            latent_channels
        )


    def forward(
        self,
        latent,
        text
    ):

        batch_size = latent.shape[0]

        height = latent.shape[1]

        width = latent.shape[2]

        channels = latent.shape[3]

        tokens = text.shape[1]

        # Query

        query = self.query(
            latent
        )

        # Key

        key = self.key(
            text
        )

        # Value

        value = self.value(
            text
        )

        # Reshape query

        query = query.view(
            batch_size,
            height * width,
            self.heads,
            self.head_dim
        )

        # Reshape key/value

        key = key.view(
            batch_size,
            tokens,
            self.heads,
            self.head_dim
        )

        value = value.view(
            batch_size,
            tokens,
            self.heads,
            self.head_dim
        )

        # Move heads forward

        query = query.permute(
            0,
            2,
            1,
            3
        )

        key = key.permute(
            0,
            2,
            1,
            3
        )

        value = value.permute(
            0,
            2,
            1,
            3
        )

        # Attention scores

        attention_scores = torch.matmul(
            query,
            key.transpose(
                -2,
                -1
            )
        )

        attention_scores = (
            attention_scores
            /
            (self.head_dim ** 0.5)
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        # Attention output

        output = torch.matmul(
            attention_weights,
            value
        )

        # Restore dimensions

        output = output.permute(
            0,
            2,
            1,
            3
        )

        output = output.reshape(
            batch_size,
            height,
            width,
            channels
        )

        return self.output(
            output
        )


# ============================================
# TIME EMBEDDING
# ============================================

class TimeEmbedding(nn.Module):

    def __init__(self, dim=128):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                1,
                dim
            ),

            nn.SiLU(),

            nn.Linear(
                dim,
                dim
            )
        )


    def forward(self, timestep):

        timestep = timestep.float()

        timestep = timestep.unsqueeze(1)

        timestep = (
            timestep
            /
            TIMESTEPS
        )

        return self.network(
            timestep
        )


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
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.time_projection = nn.Linear(
            128,
            out_channels
        )

        self.text_projection = nn.Linear(
            TEXT_DIM,
            out_channels
        )

        self.activation = nn.SiLU()

        if in_channels != out_channels:

            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )

        else:

            self.skip = nn.Identity()


    def forward(
        self,
        x,
        time_embedding,
        text_condition
    ):

        h = self.conv1(x)

        h = self.activation(h)

        # Time conditioning

        time_condition = self.time_projection(
            time_embedding
        )

        time_condition = (
            time_condition[:, :, None, None]
        )

        # Text conditioning

        text_condition = self.text_projection(
            text_condition
        )

        text_condition = (
            text_condition[:, :, None, None]
        )

        # Combine

        h = (
            h
            +
            time_condition
            +
            text_condition
        )

        h = self.conv2(h)

        h = self.activation(h)

        return (
            h
            +
            self.skip(x)
        )


# ============================================
# TEXT-CONDITIONED LATENT U-NET
# ============================================

class TextLatentDiffusionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.time_embedding = TimeEmbedding(
            128
        )

        # Input

        self.input = nn.Conv2d(
            4,
            64,
            kernel_size=3,
            padding=1
        )

        # Encoder

        self.res1 = ResBlock(
            64,
            128
        )

        self.down = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Middle

        self.middle1 = ResBlock(
            128,
            128
        )

        self.cross_attention = CrossAttention(
            latent_channels=128,
            text_dim=128,
            heads=4
        )

        self.middle2 = ResBlock(
            128,
            128
        )

        # Decoder

        self.up = nn.ConvTranspose2d(
            128,
            128,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.res2 = ResBlock(
            256,
            128
        )

        self.res3 = ResBlock(
            128,
            64
        )

        # Output

        self.output = nn.Conv2d(
            64,
            4,
            kernel_size=3,
            padding=1
        )


    def forward(
        self,
        x,
        timestep,
        text_embeddings
    ):

        # Time

        time_embedding = (
            self.time_embedding(
                timestep
            )
        )

        # Global text condition

        text_condition = (
            text_embeddings.mean(
                dim=1
            )
        )

        # Input

        x0 = self.input(x)

        # Encoder

        x1 = self.res1(
            x0,
            time_embedding,
            text_condition
        )

        x2 = self.down(x1)

        # Middle

        x2 = self.middle1(
            x2,
            time_embedding,
            text_condition
        )

        # Cross attention

        spatial = x2.permute(
            0,
            2,
            3,
            1
        )

        spatial = self.cross_attention(
            spatial,
            text_embeddings
        )

        x2 = spatial.permute(
            0,
            3,
            1,
            2
        )

        # Middle again

        x2 = self.middle2(
            x2,
            time_embedding,
            text_condition
        )

        # Decoder

        x3 = self.up(x2)

        x3 = torch.cat(
            [x3, x1],
            dim=1
        )

        x3 = self.res2(
            x3,
            time_embedding,
            text_condition
        )

        x3 = self.res3(
            x3,
            time_embedding,
            text_condition
        )

        return self.output(x3)


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
# LOAD SPATIAL VAE
# ============================================

vae = SpatialVAE(
    LATENT_CHANNELS
).to(DEVICE)

vae.load_state_dict(
    torch.load(
        "checkpoints/spatial_vae_epoch_10.pth",
        map_location=DEVICE
    )
)

vae.eval()

for parameter in vae.parameters():

    parameter.requires_grad = False


print()
print(
    "Spatial VAE loaded successfully!"
)


# ============================================
# CREATE MODELS
# ============================================

text_encoder = TextEncoder().to(
    DEVICE
)

model = TextLatentDiffusionUNet().to(
    DEVICE
)


# ============================================
# OPTIMIZER
# ============================================

optimizer = torch.optim.AdamW(
    list(
        model.parameters()
    )
    +
    list(
        text_encoder.parameters()
    ),
    lr=LR
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
# CAPTION CREATION
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


def create_captions(labels):

    captions = []

    for label in labels:

        digit_word = digit_words[
            label.item()
        ]

        captions.append(
            f"a handwritten digit {digit_word}"
        )

    return captions


# ============================================
# TRAINING
# ============================================

print()
print(
    "Starting text-conditioned real latent "
    "diffusion training..."
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

        batch_size = images.shape[0]

        # ------------------------------------
        # Encode images into spatial latent
        # ------------------------------------

        with torch.no_grad():

            mu, _ = vae.encode(
                images
            )

            latents = mu

        # ------------------------------------
        # Create captions
        # ------------------------------------

        captions = create_captions(
            labels
        )

        # ------------------------------------
        # Tokenize captions
        # ------------------------------------

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

        # ------------------------------------
        # Text encoding
        # ------------------------------------

        text_embeddings = text_encoder(
            token_ids
        )

        # ------------------------------------
        # Random timestep
        # ------------------------------------

        timestep = torch.randint(
            0,
            TIMESTEPS,
            (batch_size,),
            device=DEVICE
        )

        # ------------------------------------
        # Random noise
        # ------------------------------------

        noise = torch.randn_like(
            latents
        )

        # ------------------------------------
        # Forward diffusion
        # ------------------------------------

        alpha_bar = alpha_bars[
            timestep
        ]

        alpha_bar = alpha_bar[
            :, None, None, None
        ]

        noisy_latent = (
            torch.sqrt(alpha_bar)
            * latents
            +
            torch.sqrt(
                1.0 - alpha_bar
            )
            * noise
        )

        # ------------------------------------
        # Predict noise
        # ------------------------------------

        predicted_noise = model(
            noisy_latent,
            timestep,
            text_embeddings
        )

        # ------------------------------------
        # MSE loss
        # ------------------------------------

        loss = torch.mean(
            (
                predicted_noise
                -
                noise
            ) ** 2
        )

        # ------------------------------------
        # Backpropagation
        # ------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    # ----------------------------------------
    # Epoch loss
    # ----------------------------------------

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

    # ----------------------------------------
    # Save checkpoint
    # ----------------------------------------

    torch.save(
        {
            "model": model.state_dict(),
            "text_encoder": (
                text_encoder.state_dict()
            )
        },
        f"checkpoints/"
        f"text_real_latent_epoch_"
        f"{epoch + 1}.pth"
    )


# ============================================
# COMPLETE
# ============================================

print()

print(
    "===================================="
)

print(
    "TEXT-CONDITIONED REAL LATENT "
    "TRAINING DONE"
)

print(
    "===================================="
)