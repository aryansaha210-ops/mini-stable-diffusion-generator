import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from spatial_vae import SpatialVAE


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = r".\checkpoints\spatial_vae_epoch_10.pth"

CFG_CHECKPOINT = r".\checkpoints\cfg_text_real_latent_epoch_20.pth"

TIMESTEPS = 300

VOCAB_SIZE = 10000

TEXT_DIM = 128

MAX_TOKENS = 16

CFG_SCALE = 5.0

DEVICE = torch.device(DEVICE)


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
            VOCAB.get(
                word,
                UNK
            )
        )

    tokens.append(END)

    tokens = tokens[:MAX_TOKENS]

    while len(tokens) < MAX_TOKENS:

        tokens.append(PAD)

    return torch.tensor(
        [tokens],
        dtype=torch.long,
        device=DEVICE
    )


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

        return self.encoder(x)


# ============================================================
# TIME EMBEDDING
# ============================================================

class TimeEmbedding(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

    def forward(self, t):

        t = t.float().view(-1, 1)

        t = t / TIMESTEPS

        return self.network(t)


# ============================================================
# RES BLOCK
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
        ).unsqueeze(-1).unsqueeze(-1)

        h = h + self.text_projection(
            text_embedding.mean(dim=1)
        ).unsqueeze(-1).unsqueeze(-1)

        h = F.silu(h)

        h = self.conv2(h)

        return F.silu(
            h + self.skip(x)
        )


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

    def forward(
        self,
        x,
        text
    ):

        b, c, h, w = x.shape

        x_flat = x.permute(
            0,
            2,
            3,
            1
        ).reshape(
            b,
            h * w,
            c
        )

        q = self.query(x_flat)

        k = self.key(text)

        v = self.value(text)

        attention = torch.softmax(
            torch.matmul(
                q,
                k.transpose(-1, -2)
            ) / (128 ** 0.5),
            dim=-1
        )

        result = torch.matmul(
            attention,
            v
        )

        result = self.output(result)

        result = result.reshape(
            b,
            h,
            w,
            c
        ).permute(
            0,
            3,
            1,
            2
        )

        return x + result


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
        timestep,
        text_embedding
    ):

        time_embedding = self.time_embedding(
            timestep
        )

        x0 = self.input(x)

        x1 = self.res1(
            x0,
            time_embedding,
            text_embedding
        )

        skip = x1

        x2 = self.down(x1)

        x2 = self.middle1(
            x2,
            time_embedding,
            text_embedding
        )

        x2 = self.cross_attention(
            x2,
            text_embedding
        )

        x2 = self.middle2(
            x2,
            time_embedding,
            text_embedding
        )

        x2 = self.up(x2)

        if x2.shape[-2:] != skip.shape[-2:]:

            x2 = F.interpolate(
                x2,
                size=skip.shape[-2:],
                mode="nearest"
            )

        x2 = torch.cat(
            [
                x2,
                skip
            ],
            dim=1
        )

        x2 = self.res2(
            x2,
            time_embedding,
            text_embedding
        )

        x2 = self.res3(
            x2,
            time_embedding,
            text_embedding
        )

        return self.output(x2)


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 60)
print("CFG EPOCH 20 GENERATION TEST")
print("=" * 60)

print("Device:", DEVICE)

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ------------------------------------------------------------
# VAE
# ------------------------------------------------------------

print("\nLoading SpatialVAE...")

vae = SpatialVAE().to(DEVICE)

vae_checkpoint = torch.load(
    VAE_CHECKPOINT,
    map_location=DEVICE
)

if (
    isinstance(vae_checkpoint, dict)
    and "model" in vae_checkpoint
):

    vae.load_state_dict(
        vae_checkpoint["model"]
    )

else:

    vae.load_state_dict(
        vae_checkpoint
    )

vae.eval()

print("SpatialVAE loaded.")


# ------------------------------------------------------------
# CFG model
# ------------------------------------------------------------

print("\nLoading CFG model...")

checkpoint = torch.load(
    CFG_CHECKPOINT,
    map_location=DEVICE
)

model = TextLatentDiffusionUNet().to(DEVICE)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()

print("CFG model loaded.")


# ------------------------------------------------------------
# Text encoder
# ------------------------------------------------------------

print("\nLoading text encoder...")

text_encoder = TextEncoder().to(DEVICE)

text_encoder.load_state_dict(
    checkpoint["text_encoder"]
)

text_encoder.eval()

print("Text encoder loaded.")


# ============================================================
# PROMPT
# ============================================================

prompt = "a handwritten digit seven"

print("\nPrompt:", prompt)

tokens = tokenize(prompt)

empty_tokens = tokenize("")


with torch.no_grad():

    text_condition = text_encoder(
        tokens
    )

    empty_condition = text_encoder(
        empty_tokens
    )


print(
    "Text condition:",
    tuple(text_condition.shape)
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
# INITIAL LATENT
# ============================================================

x = torch.randn(
    1,
    4,
    7,
    7,
    device=DEVICE
)


print("\nInitial latent:")
print(
    "Mean:",
    x.mean().item()
)

print(
    "Std:",
    x.std().item()
)


# ============================================================
# REVERSE DIFFUSION
# ============================================================

print("\nStarting 300-step reverse diffusion...")

with torch.no_grad():

    for timestep in reversed(
        range(TIMESTEPS)
    ):

        t = torch.tensor(
            [timestep],
            device=DEVICE
        )

        noise_cond = model(
            x,
            t,
            text_condition
        )

        noise_uncond = model(
            x,
            t,
            empty_condition
        )

        noise_pred = (
            noise_uncond
            +
            CFG_SCALE
            *
            (
                noise_cond
                -
                noise_uncond
            )
        )

        beta_t = betas[timestep]

        alpha_t = alphas[timestep]

        alpha_bar_t = alpha_bars[
            timestep
        ]

        if timestep > 0:

            noise = torch.randn_like(x)

        else:

            noise = torch.zeros_like(x)

        x = (
            1.0
            / torch.sqrt(alpha_t)
        ) * (
            x
            -
            (
                (1.0 - alpha_t)
                /
                torch.sqrt(
                    1.0 - alpha_bar_t
                )
            )
            * noise_pred
        ) + torch.sqrt(
            beta_t
        ) * noise

        if timestep in [
            299,
            250,
            200,
            150,
            100,
            50,
            0
        ]:

            print(
                f"Step {timestep:3d} | "
                f"mean={x.mean().item():.4f} | "
                f"std={x.std().item():.4f}"
            )


# ============================================================
# LATENT STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("FINAL LATENT")
print("=" * 60)

print(
    "Mean:",
    x.mean().item()
)

print(
    "Std:",
    x.std().item()
)

print(
    "Min:",
    x.min().item()
)

print(
    "Max:",
    x.max().item()
)


# ============================================================
# DECODE
# ============================================================

print("\nDecoding latent...")

with torch.no_grad():

    generated = vae.decode(
        x
    )

generated = generated.clamp(
    -1,
    1
)

print(
    "Generated shape:",
    tuple(generated.shape)
)


# ============================================================
# SAVE IMAGE
# ============================================================

image = generated[
    0,
    0
].cpu().numpy()


output_path = (
    r".\outputs\cfg_epoch20_generated.png"
)


plt.figure(
    figsize=(5, 5)
)

plt.imshow(
    image,
    cmap="gray",
    vmin=-1,
    vmax=1
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight",
    pad_inches=0
)

plt.close()


print(
    "\nGenerated image saved:"
)

print(
    output_path
)

print("\n" + "=" * 60)
print("EPOCH 20 GENERATION TEST COMPLETE")
print("=" * 60)