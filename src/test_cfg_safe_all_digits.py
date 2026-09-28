import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw


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

# ============================================================
# IMPORTANT:
# Use the SAFE fine-tuned checkpoint
# ============================================================

CFG_CHECKPOINT = (
    "checkpoints/cfg_text_real_latent_SAFE_BEST.pth"
)

OUTPUT_DIR = (
    "outputs/cfg_safe_all_digits"
)

GRID_PATH = (
    "outputs/cfg_safe_all_digits_grid.png"
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

        # MUST match training.
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

        # Residual connection already included.
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

print("=" * 65)
print("SAFE CFG CHECKPOINT - ALL 10 DIGITS TEST")
print("=" * 65)

print()

print("Device:", DEVICE)

if DEVICE == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
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
# LOAD SAFE CFG MODEL
# ============================================================

print()
print("Loading SAFE_BEST CFG model...")

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

print("SAFE_BEST checkpoint loaded.")

if "validation_mse" in checkpoint:

    print(
        "Checkpoint validation MSE:",
        checkpoint["validation_mse"]
    )


# ============================================================
# PROMPTS
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

prompts = [
    "a handwritten digit " + name
    for name in digit_names
]


print()
print("Prompts:")

for i, prompt in enumerate(prompts):

    print(
        f"{i}: {prompt}"
    )


# ============================================================
# TOKENIZATION
# ============================================================

tokens = torch.stack(
    [
        tokenize(prompt)
        for prompt in prompts
    ]
).to(DEVICE)

empty_tokens = torch.stack(
    [
        tokenize("")
        for _ in prompts
    ]
).to(DEVICE)


# ============================================================
# TEXT ENCODING
# ============================================================

print()
print("Encoding text...")

with torch.no_grad():

    conditional_text = (
        text_encoder(tokens)
    )

    unconditional_text = (
        text_encoder(empty_tokens)
    )


print(
    "Conditional text shape:",
    tuple(
        conditional_text.shape
    )
)

print(
    "Unconditional text shape:",
    tuple(
        unconditional_text.shape
    )
)


# ============================================================
# INITIAL LATENTS
# ============================================================

# Standard normal initial distribution,
# matching the DDPM sampling starting point.

latent = torch.randn(
    10,
    LATENT_CHANNELS,
    8,
    8,
    device=DEVICE
)


print()
print("Initial latent shape:")

print(
    tuple(latent.shape)
)

print(
    "Initial mean:",
    latent.mean().item()
)

print(
    "Initial std:",
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

        t = torch.full(
            (10,),
            timestep,
            device=DEVICE,
            dtype=torch.long
        )

        # ----------------------------------------------------
        # CONDITIONAL PREDICTION
        # ----------------------------------------------------

        conditional_noise = model(
            latent,
            t,
            conditional_text
        )

        # ----------------------------------------------------
        # UNCONDITIONAL PREDICTION
        # ----------------------------------------------------

        unconditional_noise = model(
            latent,
            t,
            unconditional_text
        )

        # ----------------------------------------------------
        # CLASSIFIER-FREE GUIDANCE
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
        # DDPM POSTERIOR
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
        # SAMPLE
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
                *
                noise
            )

        else:

            latent = posterior_mean

        # ----------------------------------------------------
        # PROGRESS
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
# FINAL LATENT STATISTICS
# ============================================================

print()
print("=" * 65)
print("FINAL LATENT")
print("=" * 65)

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
print("Decoding 10 generated latents...")

with torch.no_grad():

    generated = vae.decode(
        latent
    )


print(
    "Generated tensor shape:",
    tuple(generated.shape)
)


# ============================================================
# NORMALIZE IMAGE
# ============================================================

generated = (
    generated.clamp(
        -1,
        1
    )
    + 1
) / 2


# ============================================================
# SAVE INDIVIDUAL IMAGES
# ============================================================

images = []

for i in range(10):

    image_tensor = (
        generated[i, 0]
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

    images.append(image)

    individual_path = os.path.join(
        OUTPUT_DIR,
        f"digit_{i}.png"
    )

    image.save(
        individual_path
    )

    print(
        f"Saved digit {i}: "
        f"{individual_path}"
    )


# ============================================================
# CREATE GRID
# ============================================================

CELL_SIZE = 32

LABEL_HEIGHT = 24

GRID_COLUMNS = 5
GRID_ROWS = 2

GRID_WIDTH = (
    GRID_COLUMNS
    * CELL_SIZE
)

GRID_HEIGHT = (
    GRID_ROWS
    * (
        CELL_SIZE
        + LABEL_HEIGHT
    )
)

grid = Image.new(
    "L",
    (
        GRID_WIDTH,
        GRID_HEIGHT
    ),
    0
)

draw = ImageDraw.Draw(
    grid
)


# ============================================================
# PUT IMAGES INTO GRID
# ============================================================

for i, image in enumerate(images):

    row = i // GRID_COLUMNS

    col = i % GRID_COLUMNS

    x = (
        col
        * CELL_SIZE
    )

    y = (
        row
        * (
            CELL_SIZE
            + LABEL_HEIGHT
        )
    )

    grid.paste(
        image,
        (
            x,
            y
        )
    )

    draw.text(
        (
            x + 13,
            y + CELL_SIZE + 4
        ),
        str(i),
        fill=255
    )


# ============================================================
# SAVE GRID
# ============================================================

grid.save(
    GRID_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 65)
print("SAFE ALL-DIGITS TEST COMPLETE")
print("=" * 65)

print()
print("Grid saved to:")

print(
    "D:\\StableDiffusionProject\\"
    + GRID_PATH
)

print()
print("Individual images saved to:")

print(
    "D:\\StableDiffusionProject\\"
    + OUTPUT_DIR
)

print()
print(
    "CFG guidance scale:",
    GUIDANCE_SCALE
)

print()
print(
    "Checkpoint tested:"
)

print(
    CFG_CHECKPOINT
)

print()
print("=" * 65)