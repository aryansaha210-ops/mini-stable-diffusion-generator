import torch
import torch.nn as nn
import torch.nn.functional as F
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

GUIDANCE_SCALE = 2.0

VAE_CHECKPOINT = (
    "checkpoints/spatial_vae_epoch_10.pth"
)

CFG_CHECKPOINT = (
    "checkpoints/cfg_text_real_latent_epoch_5.pth"
)

OUTPUT_PATH = (
    "outputs/cfg_epoch5_generated.png"
)


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
    "nine": 22,
}

PAD = 0
UNK = 1
START = 2
END = 3


def tokenize(text):

    words = text.lower().strip().split()

    tokens = [START]

    for word in words:

        tokens.append(
            VOCAB.get(word, UNK)
        )

    tokens.append(END)

    tokens = tokens[:MAX_TOKENS]

    while len(tokens) < MAX_TOKENS:

        tokens.append(PAD)

    return torch.tensor(
        tokens,
        dtype=torch.long
    )


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
            3,
            padding=1
        )

        self.logvar = nn.Conv2d(
            128,
            4,
            3,
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

        t = t.float().unsqueeze(1)

        # IMPORTANT:
        # Training normalized t by TIMESTEPS.
        t = t / TIMESTEPS

        return self.network(t)


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

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=TEXT_DIM,
                nhead=4,
                dim_feedforward=256,
                batch_first=True
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=2
            )
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

        q = self.query(x_flat)

        k = self.key(text)

        v = self.value(text)

        q = q.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        attention = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        attention = (
            attention
            / (self.head_dim ** 0.5)
        )

        attention = torch.softmax(
            attention,
            dim=-1
        )

        out = torch.matmul(
            attention,
            v
        )

        out = (
            out
            .transpose(1, 2)
            .contiguous()
        )

        out = out.view(
            B,
            H * W,
            128
        )

        out = self.output(out)

        out = out.transpose(
            1,
            2
        )

        out = out.view(
            B,
            128,
            H,
            W
        )

        # CrossAttention already contains
        # the residual connection.
        return x + out


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
        time_emb,
        text_emb
    ):

        h = self.conv1(x)

        h = F.silu(h)

        time_condition = (
            self.time_projection(
                time_emb
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        text_condition = (
            self.text_projection(
                text_emb
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        h = (
            h
            + time_condition
            + text_condition
        )

        h = self.conv2(h)

        h = F.silu(h)

        return (
            h
            + self.skip(x)
        )


# ============================================================
# U-NET
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

        self.cross_attention = (
            CrossAttention()
        )

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

        self.time_embedding = (
            TimeEmbedding()
        )

    def forward(
        self,
        x,
        t,
        text
    ):

        time_emb = (
            self.time_embedding(t)
        )

        text_mean = text.mean(
            dim=1
        )

        x0 = self.input(x)

        x1 = self.res1(
            x0,
            time_emb,
            text_mean
        )

        skip = x1

        x2 = self.down(x1)

        x2 = self.middle1(
            x2,
            time_emb,
            text_mean
        )

        x2 = self.cross_attention(
            x2,
            text
        )

        x2 = self.middle2(
            x2,
            time_emb,
            text_mean
        )

        x2 = self.up(x2)

        x2 = torch.cat(
            [
                x2,
                skip
            ],
            dim=1
        )

        x2 = self.res2(
            x2,
            time_emb,
            text_mean
        )

        x2 = self.res3(
            x2,
            time_emb,
            text_mean
        )

        return self.output(x2)


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
# MAIN
# ============================================================

print("=" * 60)
print("CFG EPOCH 5 GENERATION TEST")
print("=" * 60)

print()

print("Device:", DEVICE)

if DEVICE == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD VAE
# ============================================================

print()
print("Loading SpatialVAE...")

vae = SpatialVAE().to(
    DEVICE
)

vae_state = torch.load(
    VAE_CHECKPOINT,
    map_location=DEVICE
)

vae.load_state_dict(
    vae_state
)

vae.eval()

print("SpatialVAE loaded.")


# ============================================================
# LOAD CFG MODEL
# ============================================================

print()
print("Loading epoch-5 CFG model...")

model = TextLatentDiffusionUNet().to(
    DEVICE
)

text_encoder = TextEncoder().to(
    DEVICE
)

checkpoint = torch.load(
    CFG_CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model"]
)

text_encoder.load_state_dict(
    checkpoint["text_encoder"]
)

model.eval()
text_encoder.eval()

print("Epoch-5 CFG model loaded.")


# ============================================================
# PROMPT
# ============================================================

prompt = input(
    "Enter prompt "
    "(example: a handwritten digit seven): "
).strip()

if prompt == "":

    prompt = (
        "a handwritten digit seven"
    )


print()
print("Prompt:", prompt)

print(
    "CFG guidance scale:",
    GUIDANCE_SCALE
)


# ============================================================
# TEXT
# ============================================================

tokens = tokenize(
    prompt
).unsqueeze(0).to(
    DEVICE
)

empty_tokens = tokenize(
    ""
).unsqueeze(0).to(
    DEVICE
)


with torch.no_grad():

    conditional_text = (
        text_encoder(tokens)
    )

    unconditional_text = (
        text_encoder(empty_tokens)
    )


print(
    "Text condition:",
    tuple(
        conditional_text.shape
    )
)


# ============================================================
# INITIAL LATENT
# ============================================================

# IMPORTANT:
# Real VAE latents have mean ~1.46
# and std ~2.36.
#
# But the diffusion process starts from
# approximately N(0,1) at t=299.
#
# Therefore standard normal initialization
# is correct for DDPM sampling.

latent = torch.randn(
    1,
    LATENT_CHANNELS,
    8,
    8,
    device=DEVICE
)


print()
print("Initial latent:")

print(
    "Mean:",
    latent.mean().item()
)

print(
    "Std:",
    latent.std().item()
)


# ============================================================
# REVERSE DIFFUSION
# ============================================================

print()
print(
    "Starting 300-step reverse diffusion..."
)


with torch.no_grad():

    for timestep in reversed(
        range(TIMESTEPS)
    ):

        t = torch.tensor(
            [timestep],
            device=DEVICE,
            dtype=torch.long
        )

        # ----------------------------------------------------
        # Conditional prediction
        # ----------------------------------------------------

        conditional_noise = model(
            latent,
            t,
            conditional_text
        )

        # ----------------------------------------------------
        # Unconditional prediction
        # ----------------------------------------------------

        unconditional_noise = model(
            latent,
            t,
            unconditional_text
        )

        # ----------------------------------------------------
        # Classifier-Free Guidance
        # ----------------------------------------------------

        guided_noise = (
            unconditional_noise
            +
            GUIDANCE_SCALE
            *
            (
                conditional_noise
                -
                unconditional_noise
            )
        )

        # ----------------------------------------------------
        # DDPM posterior mean
        # ----------------------------------------------------

        beta_t = betas[timestep]

        alpha_t = alphas[timestep]

        alpha_bar_t = (
            alpha_bars[timestep]
        )

        if timestep > 0:

            alpha_bar_prev = (
                alpha_bars[
                    timestep - 1
                ]
            )

        else:

            alpha_bar_prev = torch.tensor(
                1.0,
                device=DEVICE
            )

        posterior_variance = (
            beta_t
            *
            (
                1.0
                -
                alpha_bar_prev
            )
            /
            (
                1.0
                -
                alpha_bar_t
            )
        )

        posterior_mean = (
            1.0
            /
            torch.sqrt(alpha_t)
        ) * (
            latent
            -
            (
                beta_t
                /
                torch.sqrt(
                    1.0
                    -
                    alpha_bar_t
                )
            )
            *
            guided_noise
        )

        # ----------------------------------------------------
        # Add posterior noise
        # ----------------------------------------------------

        if timestep > 0:

            noise = torch.randn_like(
                latent
            )

            latent = (
                posterior_mean
                +
                torch.sqrt(
                    posterior_variance
                )
                * noise
            )

        else:

            latent = posterior_mean

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            timestep % 50 == 0
            or timestep == 299
        ):

            print(
                f"Step {timestep:3d} | "
                f"mean={latent.mean().item():.4f} | "
                f"std={latent.std().item():.4f}"
            )


# ============================================================
# FINAL LATENT
# ============================================================

print()
print("=" * 60)
print("FINAL LATENT")
print("=" * 60)

print(
    "Shape:",
    tuple(latent.shape)
)

print(
    "Mean:",
    latent.mean().item()
)

print(
    "Std:",
    latent.std().item()
)

print(
    "Min:",
    latent.min().item()
)

print(
    "Max:",
    latent.max().item()
)


# ============================================================
# DECODE
# ============================================================

print()
print("Decoding latent...")

with torch.no_grad():

    generated = vae.decode(
        latent
    )

print(
    "Generated shape:",
    tuple(generated.shape)
)


# ============================================================
# CONVERT TO IMAGE
# ============================================================

generated = (
    generated.clamp(
        -1,
        1
    )
    + 1
) / 2

image_tensor = (
    generated[0, 0]
    .detach()
    .cpu()
    * 255
)

image_array = (
    image_tensor
    .byte()
    .numpy()
)

image = Image.fromarray(
    image_array,
    mode="L"
)


# ============================================================
# SAVE
# ============================================================

print()
print("Saving image...")

image.save(
    OUTPUT_PATH
)

print()
print("Generated image saved:")

print(
    "D:\\StableDiffusionProject\\"
    + OUTPUT_PATH
)


print()
print("=" * 60)
print("EPOCH 5 GENERATION TEST COMPLETE")
print("=" * 60)