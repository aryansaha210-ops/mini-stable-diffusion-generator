import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-4

TIMESTEPS = 300

CFG_CHECKPOINT = (
    "checkpoints/cfg_text_real_latent_epoch_5.pth"
)

VAE_CHECKPOINT = (
    "checkpoints/spatial_vae_epoch_10.pth"
)

OUTPUT_DIR = "checkpoints"


# ============================================================
# SPATIAL VAE
# ============================================================

class SpatialVAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                3,
                1,
                1
            ),

            nn.ReLU()
        )

        self.mu = nn.Conv2d(
            128,
            4,
            kernel_size=3,
            padding=1
        )

        self.logvar = nn.Conv2d(
            128,
            4,
            kernel_size=3,
            padding=1
        )

        self.decoder = nn.Sequential(

            nn.Conv2d(
                4,
                128,
                3,
                1,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                1,
                3,
                1,
                1
            ),

            nn.Tanh()
        )

    def encode(self, x):

        h = self.encoder(x)

        mu = self.mu(h)

        logvar = self.logvar(h)

        return mu, logvar

    def decode(self, z):

        return self.decoder(z)


# ============================================================
# TOKENIZER
# ============================================================

VOCAB = {

    "a": 10,
    "handwritten": 11,
    "digit": 12,

    "zero": 13,
    "one": 14,
    "two": 15,
    "three": 16,
    "four": 17,
    "five": 18,
    "six": 19,
    "seven": 20,
    "eight": 21,
    "nine": 22
}

PAD_TOKEN = 0
UNK_TOKEN = 1
START_TOKEN = 2
END_TOKEN = 3

MAX_TOKENS = 16


def tokenize(text):

    words = text.lower().strip().split()

    tokens = [START_TOKEN]

    for word in words:

        tokens.append(
            VOCAB.get(
                word,
                UNK_TOKEN
            )
        )

    tokens.append(END_TOKEN)

    tokens = tokens[:MAX_TOKENS]

    while len(tokens) < MAX_TOKENS:

        tokens.append(PAD_TOKEN)

    return torch.tensor(
        tokens,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


# ============================================================
# TEXT ENCODER
# ============================================================

class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            10000,
            128
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        x = self.encoder(x)

        return x


# ============================================================
# CROSS ATTENTION
# ============================================================

class CrossAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.query = nn.Linear(
            128,
            128
        )

        self.key = nn.Linear(
            128,
            128
        )

        self.value = nn.Linear(
            128,
            128
        )

        self.output = nn.Linear(
            128,
            128
        )

        self.heads = 4
        self.head_dim = 32

    def forward(
        self,
        x,
        text
    ):

        B, C, H, W = x.shape

        x_flat = (
            x.flatten(2)
            .transpose(1, 2)
        )

        Q = self.query(
            x_flat
        )

        K = self.key(
            text
        )

        V = self.value(
            text
        )

        Q = Q.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        attention = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        attention = (
            attention /
            (self.head_dim ** 0.5)
        )

        attention = torch.softmax(
            attention,
            dim=-1
        )

        result = torch.matmul(
            attention,
            V
        )

        result = (
            result.transpose(1, 2)
            .contiguous()
        )

        result = result.view(
            B,
            -1,
            128
        )

        result = self.output(
            result
        )

        result = (
            result.transpose(1, 2)
            .reshape(
                B,
                128,
                H,
                W
            )
        )

        return result


# ============================================================
# TIME EMBEDDING
# ============================================================

class TimeEmbedding(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                1,
                128
            ),

            nn.SiLU(),

            nn.Linear(
                128,
                128
            )
        )

    def forward(self, t):

        t = t.float().view(
            -1,
            1
        )

        return self.network(t)


# ============================================================
# RESBLOCK
# ============================================================

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

        self.text_projection = nn.Linear(
            128,
            out_channels
        )

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
        time_embedding,
        text_embedding
    ):

        h = self.conv1(x)

        h = (
            h
            + self.time_projection(
                time_embedding
            ).unsqueeze(-1).unsqueeze(-1)
        )

        h = (
            h
            + self.text_projection(
                text_embedding
            ).unsqueeze(-1).unsqueeze(-1)
        )

        h = F.silu(h)

        h = self.conv2(h)

        h = F.silu(h)

        return h + self.skip(x)


# ============================================================
# CFG U-NET
# ============================================================

class TextLatentDiffusionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.input = nn.Conv2d(
            4,
            64,
            3,
            padding=1
        )

        self.res1 = ResBlock(
            64,
            128
        )

        self.down = nn.Conv2d(
            128,
            128,
            3,
            stride=2,
            padding=1
        )

        self.middle1 = ResBlock(
            128,
            128
        )

        self.cross_attention = CrossAttention()

        self.middle2 = ResBlock(
            128,
            128
        )

        self.up = nn.ConvTranspose2d(
            128,
            128,
            4,
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

        self.output = nn.Conv2d(
            64,
            4,
            3,
            padding=1
        )

        self.time_embedding = TimeEmbedding()

    def forward(
        self,
        x,
        t,
        text,
        control=None
    ):

        time_embedding = (
            self.time_embedding(t)
        )

        text_mean = text.mean(
            dim=1
        )

        x0 = self.input(x)

        x1 = self.res1(
            x0,
            time_embedding,
            text_mean
        )

        x2 = self.down(x1)

        x3 = self.middle1(
            x2,
            time_embedding,
            text_mean
        )

        # x3 is [B, 128, 4, 4]
        # ControlNet output must also be [B, 128, 4, 4]

        if control is not None:

            x3 = x3 + control

        attention = self.cross_attention(
            x3,
            text
        )

        x3 = x3 + attention

        x3 = self.middle2(
            x3,
            time_embedding,
            text_mean
        )

        x4 = self.up(x3)

        x4 = torch.cat(
            [
                x4,
                x1
            ],
            dim=1
        )

        x4 = self.res2(
            x4,
            time_embedding,
            text_mean
        )

        x4 = self.res3(
            x4,
            time_embedding,
            text_mean
        )

        return self.output(x4)


# ============================================================
# CONTROLNET
# ============================================================

class ControlNetCondition(nn.Module):

    def __init__(self):

        super().__init__()

        self.condition_encoder = nn.Sequential(

            # 4 x 8 x 8
            nn.Conv2d(
                4,
                64,
                3,
                padding=1
            ),

            nn.SiLU(),

            # 64 x 8 x 8
            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.SiLU(),

            # 128 x 8 x 8
            # Downsample:
            # 8 x 8 -> 4 x 4
            nn.Conv2d(
                128,
                128,
                3,
                stride=2,
                padding=1
            ),

            nn.SiLU()
        )

        # Output:
        # 128 x 4 x 4

        self.zero_conv = nn.Conv2d(
            128,
            128,
            kernel_size=1
        )

        # Zero initialization is important.
        # Initially ControlNet has zero influence
        # on the pretrained diffusion model.

        nn.init.zeros_(
            self.zero_conv.weight
        )

        nn.init.zeros_(
            self.zero_conv.bias
        )

    def forward(self, condition):

        x = self.condition_encoder(
            condition
        )

        x = self.zero_conv(x)

        return x


# ============================================================
# DATASET
# ============================================================

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
    num_workers=0,
    pin_memory=True
)


# ============================================================
# LOAD VAE
# ============================================================

print()
print("Loading SpatialVAE...")
print()

vae = SpatialVAE().to(
    DEVICE
)

vae_checkpoint = torch.load(
    VAE_CHECKPOINT,
    map_location=DEVICE
)

if "model" in vae_checkpoint:

    vae.load_state_dict(
        vae_checkpoint["model"]
    )

else:

    vae.load_state_dict(
        vae_checkpoint
    )

vae.eval()

for parameter in vae.parameters():

    parameter.requires_grad = False

print("SpatialVAE loaded.")


# ============================================================
# LOAD CFG MODEL
# ============================================================

print()
print("Loading CFG model...")
print()

checkpoint = torch.load(
    CFG_CHECKPOINT,
    map_location=DEVICE
)

model = TextLatentDiffusionUNet().to(
    DEVICE
)

text_encoder = TextEncoder().to(
    DEVICE
)

model.load_state_dict(
    checkpoint["model"]
)

text_encoder.load_state_dict(
    checkpoint["text_encoder"]
)

model.eval()
text_encoder.eval()

for parameter in model.parameters():

    parameter.requires_grad = False

for parameter in text_encoder.parameters():

    parameter.requires_grad = False

print("CFG model loaded.")


# ============================================================
# CREATE CONTROLNET
# ============================================================

print()
print("Creating ControlNet-style network...")
print()

controlnet = ControlNetCondition().to(
    DEVICE
)

print("ControlNet created.")


# ============================================================
# CHECK CONTROLNET SHAPE
# ============================================================

with torch.no_grad():

    test_condition = torch.randn(
        2,
        4,
        8,
        8,
        device=DEVICE
    )

    test_control = controlnet(
        test_condition
    )

print(
    "Control input shape:",
    test_condition.shape
)

print(
    "Control output shape:",
    test_control.shape
)

assert test_control.shape == (
    2,
    128,
    4,
    4
), (
    "ControlNet output shape is incorrect!"
)

print(
    "ControlNet shape test successful."
)


# ============================================================
# TRAINABLE PARAMETERS
# ============================================================

trainable_parameters = list(
    controlnet.parameters()
)

print()

print(
    "Trainable ControlNet parameters:",
    sum(
        p.numel()
        for p in trainable_parameters
    )
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LR,
    weight_decay=0.01
)


# ============================================================
# DIFFUSION SCHEDULE
# ============================================================

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


# ============================================================
# DIGIT NAMES
# ============================================================

digit_names = [

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


# ============================================================
# TRAIN
# ============================================================

print()
print("========================================")
print("CONTROLNET TRAINING")
print("========================================")
print()


for epoch in range(
    1,
    EPOCHS + 1
):

    controlnet.train()

    total_loss = 0.0

    for images, labels in loader:

        # ----------------------------------------------------
        # Move data to GPU
        # ----------------------------------------------------

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Encode image into spatial latent
        # ----------------------------------------------------

        with torch.no_grad():

            mu, logvar = vae.encode(
                images
            )

            latents = mu

        # ----------------------------------------------------
        # Create captions
        # ----------------------------------------------------

        token_list = []

        for label in labels.tolist():

            caption = (
                "a handwritten digit "
                + digit_names[label]
            )

            token_list.append(
                tokenize(caption)
            )

        tokens = torch.cat(
            token_list,
            dim=0
        )

        # ----------------------------------------------------
        # Text encoder
        # ----------------------------------------------------

        with torch.no_grad():

            text_features = (
                text_encoder(tokens)
            )

        # ----------------------------------------------------
        # Random timestep
        # ----------------------------------------------------

        t = torch.randint(
            0,
            TIMESTEPS,
            (
                images.shape[0],
            ),
            device=DEVICE
        )

        # ----------------------------------------------------
        # Random noise
        # ----------------------------------------------------

        noise = torch.randn_like(
            latents
        )

        # ----------------------------------------------------
        # Get alpha bar
        # ----------------------------------------------------

        alpha_bar = alpha_bars[t].view(
            -1,
            1,
            1,
            1
        )

        # ----------------------------------------------------
        # Add noise to latent
        # ----------------------------------------------------

        noisy_latents = (
            torch.sqrt(alpha_bar)
            * latents
            +
            torch.sqrt(1.0 - alpha_bar)
            * noise
        )

        # ----------------------------------------------------
        # ControlNet condition
        #
        # Educational setup:
        # clean latent is used as the condition.
        # ----------------------------------------------------

        control_features = controlnet(
            latents
        )

        # ----------------------------------------------------
        # Predict noise
        # ----------------------------------------------------

        predicted_noise = model(
            noisy_latents,
            t,
            text_features,
            control=control_features
        )

        # ----------------------------------------------------
        # MSE loss
        # ----------------------------------------------------

        loss = F.mse_loss(
            predicted_noise,
            noise
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    # --------------------------------------------------------
    # Average loss
    # --------------------------------------------------------

    average_loss = (
        total_loss /
        len(loader)
    )

    print(
        f"Epoch {epoch} completed | "
        f"Average Loss: {average_loss:.4f}"
    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    checkpoint_path = (
        f"{OUTPUT_DIR}/"
        f"controlnet_epoch_{epoch}.pth"
    )

    torch.save(
        {
            "model": controlnet.state_dict(),
            "epoch": epoch
        },
        checkpoint_path
    )

    print(
        "ControlNet checkpoint saved:",
        checkpoint_path
    )


# ============================================================
# DONE
# ============================================================

print()
print("========================================")
print("CONTROLNET TRAINING DONE")
print("========================================")
print()

print(
    "ControlNet checkpoints saved in:"
)

print(
    OUTPUT_DIR
)