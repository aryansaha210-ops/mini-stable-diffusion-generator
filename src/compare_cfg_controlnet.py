import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMESTEPS = 300
CFG_SCALE = 5.0


# ============================================================
# Spatial VAE
# ============================================================

class SpatialVAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.SiLU(),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.SiLU()
        )

        self.mu = nn.Conv2d(
            128, 4, 3, 1, 1
        )

        self.logvar = nn.Conv2d(
            128, 4, 3, 1, 1
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(4, 128, 3, 1, 1),
            nn.SiLU(),

            nn.ConvTranspose2d(
                128, 64, 4, 2, 1
            ),
            nn.SiLU(),

            nn.ConvTranspose2d(
                64, 32, 4, 2, 1
            ),
            nn.SiLU(),

            nn.Conv2d(32, 1, 3, 1, 1),
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
# Text Encoder
# ============================================================

VOCAB_SIZE = 10000
TEXT_DIM = 128
MAX_TOKENS = 16


class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            TEXT_DIM
        )

        layer = nn.TransformerEncoderLayer(
            d_model=TEXT_DIM,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        return self.encoder(x)


# ============================================================
# Tokenizer
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


def tokenize(text):

    tokens = [2]

    words = text.lower().split()

    for word in words:

        tokens.append(
            VOCAB.get(word, 1)
        )

    tokens.append(3)

    while len(tokens) < MAX_TOKENS:
        tokens.append(0)

    tokens = tokens[:MAX_TOKENS]

    return torch.tensor(
        tokens,
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)


# ============================================================
# Time Embedding
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
# ResBlock
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

        residual = self.skip(x)

        h = self.conv1(x)

        h = h + self.time_projection(
            time_emb
        ).unsqueeze(-1).unsqueeze(-1)

        h = h + self.text_projection(
            text_emb.mean(dim=1)
        ).unsqueeze(-1).unsqueeze(-1)

        h = F.silu(h)

        h = self.conv2(h)

        return F.silu(
            h + residual
        )


# ============================================================
# Cross Attention
# ============================================================

class CrossAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.query = nn.Linear(
            128, 128
        )

        self.key = nn.Linear(
            128, 128
        )

        self.value = nn.Linear(
            128, 128
        )

        self.output = nn.Linear(
            128, 128
        )

    def forward(
        self,
        latent,
        text
    ):

        b, c, h, w = latent.shape

        x = latent.permute(
            0, 2, 3, 1
        ).reshape(
            b,
            h * w,
            c
        )

        q = self.query(x)

        k = self.key(text)

        v = self.value(text)

        attention = torch.softmax(
            torch.matmul(
                q,
                k.transpose(-1, -2)
            ) / (128 ** 0.5),
            dim=-1
        )

        out = torch.matmul(
            attention,
            v
        )

        out = self.output(out)

        out = out.reshape(
            b,
            h,
            w,
            c
        ).permute(
            0, 3, 1, 2
        )

        return latent + out


# ============================================================
# Text Latent Diffusion U-Net
# ============================================================

class TextLatentDiffusionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.input = nn.Conv2d(
            4, 64, 3, padding=1
        )

        self.res1 = ResBlock(
            64, 128
        )

        self.down = nn.Conv2d(
            128,
            128,
            3,
            stride=2,
            padding=1
        )

        self.middle1 = ResBlock(
            128, 128
        )

        self.cross_attention = CrossAttention()

        self.middle2 = ResBlock(
            128, 128
        )

        self.up = nn.ConvTranspose2d(
            128,
            128,
            4,
            stride=2,
            padding=1
        )

        self.res2 = ResBlock(
            256, 128
        )

        self.res3 = ResBlock(
            128, 64
        )

        self.output = nn.Conv2d(
            64, 4, 3, padding=1
        )

        self.time_embedding = TimeEmbedding()

    def forward(
        self,
        x,
        t,
        text
    ):

        time_emb = self.time_embedding(t)

        x1 = self.input(x)

        x2 = self.res1(
            x1,
            time_emb,
            text
        )

        x3 = self.down(x2)

        x3 = self.middle1(
            x3,
            time_emb,
            text
        )

        x3 = self.cross_attention(
            x3,
            text
        )

        x3 = self.middle2(
            x3,
            time_emb,
            text
        )

        x4 = self.up(x3)

        x4 = torch.cat(
            [x4, x2],
            dim=1
        )

        x4 = self.res2(
            x4,
            time_emb,
            text
        )

        x4 = self.res3(
            x4,
            time_emb,
            text
        )

        return self.output(x4)


# ============================================================
# ControlNet
# ============================================================

class ControlNetCondition(nn.Module):

    def __init__(self):

        super().__init__()

        self.condition_encoder = nn.Sequential(

            nn.Conv2d(
                4, 64, 3, padding=1
            ),

            nn.SiLU(),

            nn.Conv2d(
                64, 128, 3, padding=1
            ),

            nn.SiLU(),

            nn.Conv2d(
                128,
                128,
                3,
                stride=2,
                padding=1
            ),

            nn.SiLU()
        )

        self.zero_conv = nn.Conv2d(
            128,
            128,
            1
        )

    def forward(self, condition):

        return self.zero_conv(
            self.condition_encoder(condition)
        )


# ============================================================
# Load models
# ============================================================

print("Loading SpatialVAE...")

vae = SpatialVAE().to(DEVICE)

vae_checkpoint = torch.load(
    "checkpoints/spatial_vae_epoch_10.pth",
    map_location=DEVICE
)

vae.load_state_dict(
    vae_checkpoint
)

vae.eval()

print("SpatialVAE loaded.")
print()


print("Loading CFG model...")

model = TextLatentDiffusionUNet().to(DEVICE)

checkpoint = torch.load(
    "checkpoints/cfg_text_real_latent_epoch_5.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model"]
)

text_encoder = TextEncoder().to(DEVICE)

text_encoder.load_state_dict(
    checkpoint["text_encoder"]
)

model.eval()
text_encoder.eval()

print("CFG model loaded.")
print()


print("Loading ControlNet...")

controlnet = ControlNetCondition().to(DEVICE)

control_checkpoint = torch.load(
    "checkpoints/controlnet_epoch_5.pth",
    map_location=DEVICE
)

if "model" in control_checkpoint:
    controlnet.load_state_dict(
        control_checkpoint["model"]
    )
else:
    controlnet.load_state_dict(
        control_checkpoint
    )

controlnet.eval()

print("ControlNet loaded.")
print()


# ============================================================
# Diffusion schedule
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
# Generate CFG image
# ============================================================

@torch.no_grad()
def generate_cfg(text):

    tokens = tokenize(text)

    text_condition = text_encoder(
        tokens
    )

    unconditional_tokens = tokenize("")

    unconditional = text_encoder(
        unconditional_tokens
    )

    latent = torch.randn(
        1,
        4,
        8,
        8,
        device=DEVICE
    )

    for step in reversed(range(TIMESTEPS)):

        t = torch.tensor(
            [step],
            device=DEVICE
        )

        noise_cond = model(
            latent,
            t,
            text_condition
        )

        noise_uncond = model(
            latent,
            t,
            unconditional
        )

        noise = (
            noise_uncond
            + CFG_SCALE
            * (noise_cond - noise_uncond)
        )

        alpha = alphas[step]
        alpha_bar = alpha_bars[step]
        beta = betas[step]

        latent = (
            1 / torch.sqrt(alpha)
        ) * (
            latent
            - (
                (1 - alpha)
                / torch.sqrt(1 - alpha_bar)
            ) * noise
        )

        if step > 0:

            latent += torch.sqrt(beta) * (
                torch.randn_like(latent)
            )

    image = vae.decode(latent)

    return image


# ============================================================
# Generate ControlNet image
# ============================================================

@torch.no_grad()
def generate_controlnet(
    text,
    condition_image
):

    tokens = tokenize(text)

    text_condition = text_encoder(
        tokens
    )

    unconditional_tokens = tokenize("")

    unconditional = text_encoder(
        unconditional_tokens
    )

    condition_latent, _ = vae.encode(
        condition_image
    )

    control = controlnet(
        condition_latent
    )

    latent = torch.randn(
        1,
        4,
        8,
        8,
        device=DEVICE
    )

    for step in reversed(range(TIMESTEPS)):

        t = torch.tensor(
            [step],
            device=DEVICE
        )

        noise_cond = model(
            latent,
            t,
            text_condition
        )

        noise_uncond = model(
            latent,
            t,
            unconditional
        )

        noise = (
            noise_uncond
            + CFG_SCALE
            * (noise_cond - noise_uncond)
        )

        # Educational ControlNet injection
        hidden = model.input(latent)

        hidden = model.res1(
            hidden,
            model.time_embedding(t),
            text_condition
        )

        hidden = model.down(hidden)

        hidden = model.middle1(
            hidden,
            model.time_embedding(t),
            text_condition
        )

        hidden = hidden + control

        hidden = model.cross_attention(
            hidden,
            text_condition
        )

        hidden = model.middle2(
            hidden,
            model.time_embedding(t),
            text_condition
        )

        hidden = model.up(hidden)

        hidden = torch.cat(
            [hidden, model.res1(
                model.input(latent),
                model.time_embedding(t),
                text_condition
            )],
            dim=1
        )

        hidden = model.res2(
            hidden,
            model.time_embedding(t),
            text_condition
        )

        hidden = model.res3(
            hidden,
            model.time_embedding(t),
            text_condition
        )

        noise_control = model.output(hidden)

        noise = (
            noise
            + 0.1 * noise_control
        )

        alpha = alphas[step]
        alpha_bar = alpha_bars[step]
        beta = betas[step]

        latent = (
            1 / torch.sqrt(alpha)
        ) * (
            latent
            - (
                (1 - alpha)
                / torch.sqrt(1 - alpha_bar)
            ) * noise
        )

        if step > 0:

            latent += torch.sqrt(beta) * (
                torch.randn_like(latent)
            )

    image = vae.decode(latent)

    return image


# ============================================================
# MNIST conditions
# ============================================================

mnist = datasets.MNIST(
    "./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

conditions = []

for digit in range(10):

    for image, label in mnist:

        if label == digit:

            image = image.unsqueeze(0).to(
                DEVICE
            )

            image = image * 2 - 1

            conditions.append(image)

            break


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
# Generate comparison
# ============================================================

cfg_images = []
control_images = []

print("Generating comparison images...")
print()

for digit in range(10):

    prompt = (
        f"a handwritten digit "
        f"{digit_names[digit]}"
    )

    print(
        f"Generating digit {digit}..."
    )

    cfg_image = generate_cfg(
        prompt
    )

    control_image = generate_controlnet(
        prompt,
        conditions[digit]
    )

    cfg_images.append(
        cfg_image.cpu()
    )

    control_images.append(
        control_image.cpu()
    )


# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(
    2,
    10,
    figsize=(20, 5)
)

for i in range(10):

    cfg = cfg_images[i][0, 0]

    control = control_images[i][0, 0]

    axes[0, i].imshow(
        cfg,
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    axes[0, i].set_title(
        digit_names[i]
    )

    axes[0, i].axis("off")

    axes[1, i].imshow(
        control,
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    axes[1, i].axis("off")


axes[0, 0].set_ylabel(
    "CFG",
    fontsize=14
)

axes[1, 0].set_ylabel(
    "ControlNet",
    fontsize=14
)

plt.suptitle(
    "CFG vs ControlNet Comparison",
    fontsize=18
)

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "cfg_vs_controlnet_comparison.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print()
print("Comparison saved:")
print(output_path)
print()
print("CFG vs CONTROLNET EVALUATION DONE")