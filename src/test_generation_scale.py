import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from spatial_vae import SpatialVAE


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = r".\checkpoints\spatial_vae_epoch_10.pth"
CFG_CHECKPOINT = r".\checkpoints\cfg_text_real_latent_epoch_5.pth"

LATENT_SCALE = 0.42028894798203276

TIMESTEPS = 300

DEVICE = torch.device(DEVICE)


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
    "nine": 22,
}

PAD = 0
UNK = 1
START = 2
END = 3

MAX_TOKENS = 16
VOCAB_SIZE = 10000


def tokenize(text):

    words = text.lower().strip().split()

    tokens = [START]

    for word in words:

        if word in VOCAB:
            tokens.append(VOCAB[word])
        else:
            tokens.append(UNK)

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

    def __init__(self, in_channels, out_channels):

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

        h = torch.nn.functional.silu(h)

        h = self.conv2(h)

        return torch.nn.functional.silu(
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

    def forward(self, x, text):

        b, c, h, w = x.shape

        x_flat = x.permute(
            0, 2, 3, 1
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

        # Handle possible 1-pixel spatial difference
        if x2.shape[-2:] != skip.shape[-2:]:

            x2 = torch.nn.functional.interpolate(
                x2,
                size=skip.shape[-2:],
                mode="nearest"
            )

        x2 = torch.cat(
            [x2, skip],
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
print("GENERATION SCALE TEST")
print("=" * 60)

print("Device:", DEVICE)

print("\nLoading SpatialVAE...")

vae = SpatialVAE().to(DEVICE)

vae_checkpoint = torch.load(
    VAE_CHECKPOINT,
    map_location=DEVICE
)

if isinstance(vae_checkpoint, dict) and "model" in vae_checkpoint:
    vae.load_state_dict(
        vae_checkpoint["model"]
    )
else:
    vae.load_state_dict(
        vae_checkpoint
    )

vae.eval()

print("SpatialVAE loaded.")


print("\nLoading text encoder...")

text_encoder = TextEncoder().to(DEVICE)

checkpoint = torch.load(
    CFG_CHECKPOINT,
    map_location=DEVICE
)

text_encoder.load_state_dict(
    checkpoint["text_encoder"]
)

text_encoder.eval()

print("Text encoder loaded.")


print("\nLoading CFG model...")

model = TextLatentDiffusionUNet().to(DEVICE)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()

print("CFG model loaded.")


# ============================================================
# TEXT
# ============================================================

prompt = "a handwritten digit seven"

tokens = tokenize(prompt)

with torch.no_grad():

    text_condition = text_encoder(tokens)

print("\nPrompt:", prompt)

print(
    "Text condition shape:",
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
# INITIAL NOISE
# ============================================================

# IMPORTANT:
# The VAE latent space has std ≈ 2.38.
# We therefore test noise at the corresponding scale.

latent_height = 7
latent_width = 7

x = torch.randn(
    1,
    4,
    latent_height,
    latent_width,
    device=DEVICE
)

print("\nInitial noise:")
print("Shape:", tuple(x.shape))
print("Mean:", x.mean().item())
print("Std:", x.std().item())


# ============================================================
# REVERSE DIFFUSION
# ============================================================

print("\nStarting reverse diffusion...")

CFG_SCALE = 5.0

with torch.no_grad():

    for timestep in reversed(range(TIMESTEPS)):

        t = torch.tensor(
            [timestep],
            device=DEVICE
        )

        # Conditional prediction
        noise_cond = model(
            x,
            t,
            text_condition
        )

        # Unconditional prediction
        empty_tokens = tokenize("")

        empty_condition = text_encoder(
            empty_tokens
        )

        noise_uncond = model(
            x,
            t,
            empty_condition
        )

        # Classifier-Free Guidance
        noise_pred = (
            noise_uncond
            + CFG_SCALE
            * (
                noise_cond
                - noise_uncond
            )
        )

        beta_t = betas[timestep]

        alpha_t = alphas[timestep]

        alpha_bar_t = alpha_bars[timestep]

        if timestep > 0:

            noise = torch.randn_like(x)

        else:

            noise = torch.zeros_like(x)

        x = (
            1 / torch.sqrt(alpha_t)
        ) * (
            x
            - (
                (1 - alpha_t)
                / torch.sqrt(
                    1 - alpha_bar_t
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
# GENERATED LATENT STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("GENERATED LATENT")
print("=" * 60)

print("Shape:", tuple(x.shape))

print("Mean:", x.mean().item())

print("Std:", x.std().item())

print("Min:", x.min().item())

print("Max:", x.max().item())


# ============================================================
# DECODE TESTS
# ============================================================

print("\n" + "=" * 60)
print("TESTING DIFFERENT LATENT SCALES")
print("=" * 60)

scales = [
    1.0,
    LATENT_SCALE,
    1.0 / LATENT_SCALE
]

images = []

with torch.no_grad():

    for scale in scales:

        latent = x * scale

        print(
            f"\nScale {scale:.4f}"
        )

        print(
            "Latent mean:",
            latent.mean().item()
        )

        print(
            "Latent std:",
            latent.std().item()
        )

        decoded = vae.decode(
            latent
        )

        decoded = decoded.clamp(
            -1,
            1
        )

        images.append(
            decoded[0, 0]
            .cpu()
            .numpy()
        )

        print(
            "Decoded min:",
            decoded.min().item()
        )

        print(
            "Decoded max:",
            decoded.max().item()
        )


# ============================================================
# SAVE COMPARISON
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 4)
)

titles = [
    "Scale = 1.0",
    f"Scale = {LATENT_SCALE:.3f}",
    f"Scale = {1.0 / LATENT_SCALE:.3f}"
]

for ax, image, title in zip(
    axes,
    images,
    titles
):

    ax.imshow(
        image,
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    ax.set_title(title)

    ax.axis("off")


plt.tight_layout()

output_path = (
    r".\outputs\latent_scale_comparison.png"
)

plt.savefig(
    output_path,
    dpi=150
)

plt.close()

print(
    "\nSaved:",
    output_path
)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)