import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128

TIMESTEPS = 300

LATENT_CHANNELS = 4

TEXT_DIM = 128

VOCAB_SIZE = 10000

MAX_TOKENS = 16

CFG_GUIDANCE = 5.0

VAE_CHECKPOINT = (
    "checkpoints/spatial_vae_epoch_10.pth"
)

CFG_CHECKPOINT = (
    "checkpoints/cfg_text_real_latent_epoch_30.pth"
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

        if word in VOCAB:
            tokens.append(VOCAB[word])
        else:
            tokens.append(UNK)

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

        return (
            h
            + self.skip(x)
        )


# ============================================================
# CFG U-NET
# ============================================================

class TextLatentDiffusionUNet(
    nn.Module
):

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
# LOAD MODELS
# ============================================================

print("=" * 60)
print("CFG MODEL QUALITY TEST")
print("=" * 60)

print()

print(
    "Device:",
    DEVICE
)

if DEVICE == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ------------------------------------------------------------
# VAE
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# CFG model
# ------------------------------------------------------------

print()
print("Loading CFG model...")

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

print("CFG model loaded.")


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
    shuffle=True
)


# ============================================================
# TEXT TOKENS
# ============================================================

print()
print("Preparing text conditions...")

conditional_tokens = []

unconditional_tokens = []

for digit in range(10):

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

    prompt = (
        "a handwritten digit "
        + digit_names[digit]
    )

    conditional_tokens.append(
        tokenize(prompt)
    )

    unconditional_tokens.append(
        tokenize("")
    )


conditional_tokens = torch.stack(
    conditional_tokens
).to(
    DEVICE
)

unconditional_tokens = torch.stack(
    unconditional_tokens
).to(
    DEVICE
)

with torch.no_grad():

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


# ============================================================
# GET REAL LATENTS
# ============================================================

print()
print("Encoding real MNIST images...")

images, labels = next(
    iter(loader)
)

images = images.to(
    DEVICE
)

labels = labels.to(
    DEVICE
)

with torch.no_grad():

    latents, _ = vae.encode(
        images
    )


print(
    "Images:",
    tuple(images.shape)
)

print(
    "Latents:",
    tuple(latents.shape)
)


# ============================================================
# TEST TIMESTEPS
# ============================================================

test_timesteps = [
    0,
    50,
    100,
    150,
    200,
    250,
    299
]


print()
print("=" * 60)
print("NOISE PREDICTION TEST")
print("=" * 60)

print()

print(
    "Testing 128 real MNIST latent samples."
)

print()


results = []


with torch.no_grad():

    for timestep in test_timesteps:

        t = torch.full(
            (
                latents.shape[0],
            ),
            timestep,
            device=DEVICE,
            dtype=torch.long
        )

        # ----------------------------------------------------
        # Forward diffusion
        # ----------------------------------------------------

        noise = torch.randn_like(
            latents
        )

        alpha_bar = alpha_bars[
            timestep
        ]

        alpha_bar = alpha_bar.view(
            1,
            1,
            1,
            1
        )

        noisy_latents = (
            torch.sqrt(
                alpha_bar
            )
            * latents
            +
            torch.sqrt(
                1.0 - alpha_bar
            )
            * noise
        )

        # ----------------------------------------------------
        # Select correct text condition
        # ----------------------------------------------------

        batch_text_cond = (
            conditional_text[
                labels
            ]
        )

        batch_text_uncond = (
            unconditional_text[
                labels
            ]
        )

        # ----------------------------------------------------
        # Conditional prediction
        # ----------------------------------------------------

        conditional_prediction = model(
            noisy_latents,
            t,
            batch_text_cond
        )

        # ----------------------------------------------------
        # Unconditional prediction
        # ----------------------------------------------------

        unconditional_prediction = model(
            noisy_latents,
            t,
            batch_text_uncond
        )

        # ----------------------------------------------------
        # CFG prediction
        # ----------------------------------------------------

        cfg_prediction = (
            unconditional_prediction
            +
            CFG_GUIDANCE
            *
            (
                conditional_prediction
                -
                unconditional_prediction
            )
        )

        # ----------------------------------------------------
        # MSE
        # ----------------------------------------------------

        conditional_mse = torch.mean(
            (
                conditional_prediction
                -
                noise
            ) ** 2
        ).item()

        unconditional_mse = torch.mean(
            (
                unconditional_prediction
                -
                noise
            ) ** 2
        ).item()

        cfg_mse = torch.mean(
            (
                cfg_prediction
                -
                noise
            ) ** 2
        ).item()

        # ----------------------------------------------------
        # Conditional difference
        # ----------------------------------------------------

        condition_difference = torch.mean(
            (
                conditional_prediction
                -
                unconditional_prediction
            ) ** 2
        ).item()

        # ----------------------------------------------------
        # Noise statistics
        # ----------------------------------------------------

        noise_std = (
            noise.std().item()
        )

        prediction_std = (
            conditional_prediction.std().item()
        )

        results.append(
            (
                timestep,
                conditional_mse,
                unconditional_mse,
                cfg_mse,
                condition_difference,
                noise_std,
                prediction_std
            )
        )

        print(
            f"t={timestep:3d} | "
            f"conditional MSE={conditional_mse:.6f} | "
            f"unconditional MSE={unconditional_mse:.6f} | "
            f"CFG MSE={cfg_mse:.6f} | "
            f"cond diff={condition_difference:.6f}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print()

print(
    "Lower noise-prediction MSE is better."
)

print()

for result in results:

    (
        timestep,
        conditional_mse,
        unconditional_mse,
        cfg_mse,
        condition_difference,
        noise_std,
        prediction_std
    ) = result

    print(
        f"t={timestep:3d} | "
        f"cond={conditional_mse:.6f} | "
        f"uncond={unconditional_mse:.6f} | "
        f"CFG={cfg_mse:.6f}"
    )


# ============================================================
# AVERAGES
# ============================================================

avg_conditional = sum(
    r[1] for r in results
) / len(results)

avg_unconditional = sum(
    r[2] for r in results
) / len(results)

avg_cfg = sum(
    r[3] for r in results
) / len(results)

avg_difference = sum(
    r[4] for r in results
) / len(results)


print()
print("=" * 60)
print("AVERAGES")
print("=" * 60)

print()

print(
    "Average conditional MSE:",
    f"{avg_conditional:.6f}"
)

print(
    "Average unconditional MSE:",
    f"{avg_unconditional:.6f}"
)

print(
    "Average CFG MSE:",
    f"{avg_cfg:.6f}"
)

print(
    "Average conditional/unconditional difference:",
    f"{avg_difference:.6f}"
)


# ============================================================
# BASELINE
# ============================================================

print()
print("=" * 60)
print("RANDOM BASELINE")
print("=" * 60)

print()

print(
    "If the model predicted zero noise everywhere,"
)

print(
    "the expected MSE would be approximately 1.0."
)

print()

if avg_conditional < 1.0:

    print(
        "Conditional model is beating the "
        "zero-prediction baseline."
    )

else:

    print(
        "Conditional model is NOT beating "
        "the zero-prediction baseline."
    )


if avg_conditional < 0.5:

    print(
        "The model is learning meaningful "
        "noise prediction."
    )

elif avg_conditional < 0.8:

    print(
        "The model has learned some noise "
        "prediction, but it is weak."
    )

else:

    print(
        "Noise prediction is weak."
    )


print()
print("=" * 60)
print("CFG MODEL QUALITY TEST COMPLETE")
print("=" * 60)