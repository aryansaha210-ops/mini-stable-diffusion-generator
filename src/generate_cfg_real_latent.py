import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TIMESTEPS = 300
LATENT_CHANNELS = 4
TEXT_DIM = 128
VOCAB_SIZE = 10000
MAX_TOKENS = 16

GUIDANCE_SCALE = 5.0

VAE_CHECKPOINT = "checkpoints/spatial_vae_epoch_10.pth"
CFG_CHECKPOINT = "checkpoints/cfg_text_real_latent_epoch_5.pth"

OUTPUT_PATH = "outputs/cfg_generated.png"


# ============================================================
# DETERMINISTIC TOKENIZER
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


def tokenize(text):
    words = text.lower().strip().split()

    tokens = [START_TOKEN]

    for word in words:
        tokens.append(VOCAB.get(word, UNK_TOKEN))

    tokens.append(END_TOKEN)

    tokens = tokens[:MAX_TOKENS]

    while len(tokens) < MAX_TOKENS:
        tokens.append(PAD_TOKEN)

    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)


# ============================================================
# SPATIAL VAE
# ============================================================

class SpatialVAE(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 1, 1),
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

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 128, 3, 1, 1),
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

            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

    def encode(self, x):

        h = self.encoder(x)

        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar


# ============================================================
# TEXT ENCODER
# ============================================================

class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            TEXT_DIM
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TEXT_DIM,
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
            TEXT_DIM,
            128
        )

        self.value = nn.Linear(
            TEXT_DIM,
            128
        )

        self.output = nn.Linear(
            128,
            128
        )

        self.heads = 4

        self.head_dim = 32

    def forward(self, x, text):

        # x:
        # [B, 128, H, W]

        B, C, H, W = x.shape

        x_flat = x.flatten(
            2
        ).transpose(
            1,
            2
        )

        # [B, HW, 128]

        Q = self.query(x_flat)

        K = self.key(text)

        V = self.value(text)

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

        attention = attention / math.sqrt(
            self.head_dim
        )

        attention = F.softmax(
            attention,
            dim=-1
        )

        result = torch.matmul(
            attention,
            V
        )

        result = result.transpose(
            1,
            2
        ).contiguous()

        result = result.view(
            B,
            -1,
            128
        )

        result = self.output(result)

        result = result.transpose(
            1,
            2
        ).reshape(
            B,
            128,
            H,
            W
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

        h = h + self.time_projection(
            time_embedding
        ).unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        h = h + self.text_projection(
            text_embedding
        ).unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        h = F.silu(h)

        h = self.conv2(h)

        h = F.silu(h)

        return h + self.skip(x)


# ============================================================
# TEXT LATENT DIFFUSION U-NET
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
        text
    ):

        time_embedding = self.time_embedding(t)

        x0 = self.input(x)

        x1 = self.res1(
            x0,
            time_embedding,
            text.mean(dim=1)
        )

        x2 = self.down(x1)

        x3 = self.middle1(
            x2,
            time_embedding,
            text.mean(dim=1)
        )

        attention = self.cross_attention(
            x3,
            text
        )

        x3 = x3 + attention

        x3 = self.middle2(
            x3,
            time_embedding,
            text.mean(dim=1)
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
            text.mean(dim=1)
        )

        x4 = self.res3(
            x4,
            time_embedding,
            text.mean(dim=1)
        )

        return self.output(x4)


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("========================================")
print("CFG TEXT-TO-IMAGE GENERATION")
print("========================================")
print()

print("Device:", DEVICE)

print()
print("Loading SpatialVAE...")

vae = SpatialVAE().to(DEVICE)

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

print("SpatialVAE loaded.")


print()
print("Loading CFG diffusion model...")

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

print("CFG model loaded.")


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
# TEXT
# ============================================================

prompt = input(
    "Enter prompt "
    "(example: a handwritten digit seven): "
).strip()

if not prompt:

    prompt = "a handwritten digit seven"


print()
print("Prompt:", prompt)
print("Guidance Scale:", GUIDANCE_SCALE)


# Conditional tokens
conditional_tokens = tokenize(
    prompt
)

# Unconditional tokens
unconditional_tokens = tokenize(
    ""
)


with torch.no_grad():

    conditional_text = text_encoder(
        conditional_tokens
    )

    unconditional_text = text_encoder(
        unconditional_tokens
    )


# ============================================================
# START FROM RANDOM LATENT
# ============================================================

latent = torch.randn(
    1,
    LATENT_CHANNELS,
    8,
    8,
    device=DEVICE
)


print()
print("Starting reverse diffusion...")


# ============================================================
# REVERSE DIFFUSION
# ============================================================

with torch.no_grad():

    for t in reversed(
        range(TIMESTEPS)
    ):

        timestep = torch.tensor(
            [t],
            device=DEVICE
        )

        # --------------------------------------------
        # UNCONDITIONAL PREDICTION
        # --------------------------------------------

        eps_unconditional = model(
            latent,
            timestep,
            unconditional_text
        )

        # --------------------------------------------
        # CONDITIONAL PREDICTION
        # --------------------------------------------

        eps_conditional = model(
            latent,
            timestep,
            conditional_text
        )

        # --------------------------------------------
        # CLASSIFIER-FREE GUIDANCE
        #
        # eps_cfg =
        # eps_unconditional +
        # guidance_scale *
        # (eps_conditional - eps_unconditional)
        # --------------------------------------------

        eps_cfg = (
            eps_unconditional
            + GUIDANCE_SCALE
            * (
                eps_conditional
                - eps_unconditional
            )
        )

        alpha_t = alphas[t]

        alpha_bar_t = alpha_bars[t]

        beta_t = betas[t]

        # DDPM reverse mean

        mean = (
            1.0
            / torch.sqrt(alpha_t)
        ) * (
            latent
            - (
                beta_t
                / torch.sqrt(
                    1.0 - alpha_bar_t
                )
            )
            * eps_cfg
        )

        # Add noise except at final step

        if t > 0:

            noise = torch.randn_like(
                latent
            )

            latent = (
                mean
                + torch.sqrt(beta_t)
                * noise
            )

        else:

            latent = mean

        # Progress

        if (
            t % 50 == 0
            or t == TIMESTEPS - 1
        ):

            print(
                f"Reverse step: {t}"
            )


# ============================================================
# DECODE LATENT
# ============================================================

print()
print("Decoding latent with SpatialVAE...")

with torch.no_grad():

    image = vae.decode(
        latent
    )

# [-1, 1] → [0, 1]

image = (
    image.clamp(-1, 1)
    + 1
) / 2


# ============================================================
# SAVE IMAGE
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

torchvision.utils.save_image(
    image,
    OUTPUT_PATH
)


print()
print("========================================")
print("GENERATION COMPLETE")
print("========================================")
print()
print("Prompt:", prompt)
print("Guidance Scale:", GUIDANCE_SCALE)
print("Output:", OUTPUT_PATH)
print()