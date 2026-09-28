import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG_CHECKPOINT = (
    r".\checkpoints\cfg_text_real_latent_FINAL.pth"
)

VAE_CHECKPOINT = (
    r".\checkpoints\spatial_vae_epoch_10.pth"
)

OUTPUT_DIR = (
    r".\outputs\cfg_100_eval"
)

TIMESTEPS = 300

LATENT_CHANNELS = 4

TEXT_DIM = 128

VOCAB_SIZE = 10000

MAX_TOKENS = 16

CFG_SCALE = 2.0

SAMPLES_PER_DIGIT = 10

TOTAL_SAMPLES = 100

BATCH_SIZE = 10


DIGIT_NAMES = [
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
# HEADER
# ============================================================

print("=" * 70)
print("100-SAMPLE CFG EVALUATION GENERATOR")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

if DEVICE == "cuda":

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
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
    "nine": 22
}


PAD = 0
UNK = 1
START = 2
END = 3


def tokenize(text):

    words = text.lower().split()

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
        tokens,
        dtype=torch.long
    )


# ============================================================
# EXACT SPATIAL VAE
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

        out = out.transpose(1, 2)

        out = out.view(
            B,
            128,
            H,
            W
        )

        return x + out


# ============================================================
# RESIDUAL BLOCK
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

        return h + self.skip(x)


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
# LOAD VAE
# ============================================================

print("\nLoading VAE...")

vae = SpatialVAE().to(
    DEVICE
)

vae_checkpoint = torch.load(
    VAE_CHECKPOINT,
    map_location=DEVICE
)

vae.load_state_dict(
    vae_checkpoint
)

vae.eval()

for parameter in vae.parameters():

    parameter.requires_grad = False

print("VAE loaded.")


# ============================================================
# LOAD FINAL CFG MODEL
# ============================================================

print("\nLoading FINAL CFG model...")

model = (
    TextLatentDiffusionUNet()
    .to(DEVICE)
)

text_encoder = (
    TextEncoder()
    .to(DEVICE)
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

print("FINAL CFG checkpoint loaded.")


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
# OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

for digit in range(10):

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            str(digit)
        ),
        exist_ok=True
    )


# ============================================================
# GENERATION FUNCTION
# ============================================================

@torch.no_grad()
def generate_batch(prompts):

    batch_size = len(prompts)

    conditional_tokens = torch.stack(
        [
            tokenize(prompt)
            for prompt in prompts
        ]
    ).to(DEVICE)

    unconditional_tokens = torch.stack(
        [
            tokenize("")
            for _ in prompts
        ]
    ).to(DEVICE)

    conditional_text = (
        text_encoder(
            conditional_tokens
        )
    )

    unconditional_text = (
        text_encoder(
            unconditional_tokens
        )
    )

    latent = torch.randn(
        batch_size,
        4,
        8,
        8,
        device=DEVICE
    )

    print(
        f"    Initial latent: "
        f"mean={latent.mean().item():.4f}, "
        f"std={latent.std().item():.4f}"
    )

    for t_value in reversed(
        range(TIMESTEPS)
    ):

        t = torch.full(
            (batch_size,),
            t_value,
            device=DEVICE,
            dtype=torch.long
        )

        conditional_noise = model(
            latent,
            t,
            conditional_text
        )

        unconditional_noise = model(
            latent,
            t,
            unconditional_text
        )

        guided_noise = (
            unconditional_noise
            + CFG_SCALE
            * (
                conditional_noise
                - unconditional_noise
            )
        )

        beta_t = betas[t_value]

        alpha_t = alphas[t_value]

        alpha_bar_t = alpha_bars[
            t_value
        ]

        if t_value > 0:

            alpha_bar_prev = (
                alpha_bars[
                    t_value - 1
                ]
            )

        else:

            alpha_bar_prev = torch.tensor(
                1.0,
                device=DEVICE
            )

        posterior_variance = (
            beta_t
            * (
                1.0
                - alpha_bar_prev
            )
            / (
                1.0
                - alpha_bar_t
            )
        )

        posterior_mean = (
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
            * guided_noise
        )

        if t_value > 0:

            noise = torch.randn_like(
                latent
            )

            latent = (
                posterior_mean
                + torch.sqrt(
                    posterior_variance
                )
                * noise
            )

        else:

            latent = posterior_mean

        if t_value in [
            299,
            250,
            200,
            150,
            100,
            50,
            0
        ]:

            print(
                f"    Step {t_value:3d}: "
                f"mean={latent.mean().item():.4f} "
                f"std={latent.std().item():.4f}"
            )

    decoded = vae.decode(
        latent
    )

    decoded = (
        decoded.clamp(-1, 1)
        + 1
    ) / 2

    decoded = (
        decoded
        .squeeze(1)
        .cpu()
        .numpy()
    )

    return decoded


# ============================================================
# GENERATE 100 SAMPLES
# ============================================================

print("\n" + "=" * 70)
print("STARTING 100-IMAGE GENERATION")
print("=" * 70)

sample_counter = 0

for digit in range(10):

    prompt = (
        "a handwritten digit "
        + DIGIT_NAMES[digit]
    )

    print(
        f"\nDigit {digit}: "
        f"{prompt}"
    )

    for batch_start in range(
        0,
        SAMPLES_PER_DIGIT,
        BATCH_SIZE
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            SAMPLES_PER_DIGIT
        )

        batch_size = (
            batch_end
            - batch_start
        )

        prompts = [
            prompt
            for _ in range(batch_size)
        ]

        print(
            f"  Generating samples "
            f"{batch_start:02d} "
            f"to "
            f"{batch_end - 1:02d}"
        )

        generated = generate_batch(
            prompts
        )

        for i in range(
            batch_size
        ):

            sample_number = (
                batch_start + i
            )

            image_array = (
                generated[i] * 255
            ).clip(
                0,
                255
            ).astype(
                np.uint8
            )

            image = Image.fromarray(
                image_array,
                mode="L"
            )

            filename = os.path.join(
                OUTPUT_DIR,
                str(digit),
                f"sample_{sample_number:02d}.png"
            )

            image.save(
                filename
            )

            sample_counter += 1

            print(
                f"    Saved: "
                f"digit {digit}, "
                f"sample "
                f"{sample_number:02d}"
            )


# ============================================================
# CREATE GRID
# ============================================================

print("\nCreating 100-image grid...")

CELL_SIZE = 32

grid = Image.new(
    "L",
    (
        10 * CELL_SIZE,
        10 * CELL_SIZE
    ),
    color=255
)

for digit in range(10):

    for sample in range(10):

        filename = os.path.join(
            OUTPUT_DIR,
            str(digit),
            f"sample_{sample:02d}.png"
        )

        image = Image.open(
            filename
        ).convert("L")

        x = sample * CELL_SIZE

        y = digit * CELL_SIZE

        grid.paste(
            image,
            (x, y)
        )


grid_path = os.path.join(
    OUTPUT_DIR,
    "cfg_100_eval_grid.png"
)

grid.save(
    grid_path
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = os.path.join(
    OUTPUT_DIR,
    "metadata.txt"
)

with open(
    metadata_path,
    "w"
) as f:

    f.write(
        "100-SAMPLE CFG EVALUATION\n"
    )

    f.write(
        "=" * 50 + "\n\n"
    )

    f.write(
        f"Checkpoint: "
        f"{CFG_CHECKPOINT}\n"
    )

    f.write(
        f"VAE checkpoint: "
        f"{VAE_CHECKPOINT}\n"
    )

    f.write(
        f"CFG scale: "
        f"{CFG_SCALE}\n"
    )

    f.write(
        f"Timesteps: "
        f"{TIMESTEPS}\n"
    )

    f.write(
        f"Samples per digit: "
        f"{SAMPLES_PER_DIGIT}\n"
    )

    f.write(
        f"Total samples: "
        f"{sample_counter}\n"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("100-IMAGE GENERATION COMPLETE")
print("=" * 70)

print(
    f"\nTotal images generated: "
    f"{sample_counter}"
)

print(
    "\nOutput directory:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)

print(
    "\nGrid:"
)

print(
    os.path.abspath(
        grid_path
    )
)

print(
    "\nMetadata:"
)

print(
    os.path.abspath(
        metadata_path
    )
)

print("\n" + "=" * 70)
print("READY FOR CNN EVALUATION")
print("=" * 70)