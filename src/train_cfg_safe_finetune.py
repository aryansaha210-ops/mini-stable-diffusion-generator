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

# Only a short controlled experiment.
EPOCHS = 3

# Much smaller than the original 1e-4.
LR = 1e-5

TIMESTEPS = 300

LATENT_CHANNELS = 4

TEXT_DIM = 128
VOCAB_SIZE = 10000
MAX_TOKENS = 16

CFG_DROPOUT = 0.15

VAE_CHECKPOINT = (
    "checkpoints/spatial_vae_epoch_10.pth"
)

START_CHECKPOINT = (
    "checkpoints/cfg_text_real_latent_BEST.pth"
)

BEST_CHECKPOINT = (
    "checkpoints/cfg_text_real_latent_SAFE_BEST.pth"
)

os.makedirs(
    "checkpoints",
    exist_ok=True
)


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
        # Must match original training.
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

        # Residual connection is already included.
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


def add_noise(
    x,
    noise,
    t
):

    alpha_bar = alpha_bars[t]

    alpha_bar = (
        alpha_bar
        .view(-1, 1, 1, 1)
    )

    noisy = (
        torch.sqrt(alpha_bar)
        * x
        +
        torch.sqrt(
            1 - alpha_bar
        )
        * noise
    )

    return noisy


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


train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)


test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# FIXED VALIDATION BATCH
# ============================================================

# Use the first fixed batch from the test set.
#
# This makes every validation measurement comparable.

validation_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

validation_images, validation_labels = next(
    iter(validation_loader)
)

validation_images = validation_images.to(
    DEVICE
)

validation_labels = validation_labels.to(
    DEVICE
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

for parameter in vae.parameters():

    parameter.requires_grad = False

print("SpatialVAE loaded.")


# ============================================================
# LOAD CFG MODEL
# ============================================================

print()
print("Loading BEST epoch-5 checkpoint...")

model = TextLatentDiffusionUNet().to(
    DEVICE
)

text_encoder = TextEncoder().to(
    DEVICE
)

checkpoint = torch.load(
    START_CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model"]
)

text_encoder.load_state_dict(
    checkpoint["text_encoder"]
)

print("BEST checkpoint loaded.")


# ============================================================
# FREEZE TEXT ENCODER
# ============================================================

# The successful epoch-5 text representation is preserved.
#
# We only fine-tune the diffusion U-Net in this controlled
# experiment.

for parameter in text_encoder.parameters():

    parameter.requires_grad = False

text_encoder.eval()


# ============================================================
# OPTIMIZER
# ============================================================

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LR
)


# ============================================================
# FIXED VALIDATION NOISE
# ============================================================

torch.manual_seed(12345)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        12345
    )


with torch.no_grad():

    validation_mu, _ = vae.encode(
        validation_images
    )

    validation_captions = []

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

    for label in validation_labels:

        digit = int(
            label.item()
        )

        validation_captions.append(
            "a handwritten digit "
            + digit_names[digit]
        )

    validation_tokens = torch.stack([

        tokenize(caption)

        for caption
        in validation_captions

    ]).to(DEVICE)

    validation_text = (
        text_encoder(
            validation_tokens
        )
    )


# Fixed noise and timestep for every validation.
torch.manual_seed(54321)

validation_noise = torch.randn_like(
    validation_mu
)

validation_t = torch.randint(
    0,
    TIMESTEPS,
    (
        validation_mu.size(0),
    ),
    device=DEVICE
)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate_validation():

    model.eval()

    with torch.no_grad():

        noisy_latents = add_noise(
            validation_mu,
            validation_noise,
            validation_t
        )

        predicted_noise = model(
            noisy_latents,
            validation_t,
            validation_text
        )

        loss = F.mse_loss(
            predicted_noise,
            validation_noise
        )

    return loss.item()


# ============================================================
# INITIAL VALIDATION
# ============================================================

print()
print("=" * 65)
print("SAFE FINE-TUNING")
print("=" * 65)

print()
print("Device:", DEVICE)

if DEVICE == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print("Starting checkpoint:")
print(
    START_CHECKPOINT
)

print()
print("Learning rate:", LR)
print("Epochs:", EPOCHS)
print("Text encoder: FROZEN")
print("VAE: FROZEN")


initial_validation_loss = (
    evaluate_validation()
)

print()
print(
    "Initial validation MSE:",
    f"{initial_validation_loss:.6f}"
)


# ============================================================
# SAVE INITIAL BEST
# ============================================================

best_loss = initial_validation_loss

torch.save(
    {
        "model": model.state_dict(),
        "text_encoder": text_encoder.state_dict(),
        "validation_mse": best_loss,
        "source": "epoch5_safe_finetune"
    },
    BEST_CHECKPOINT
)

print()
print(
    "Initial BEST checkpoint saved:"
)

print(
    BEST_CHECKPOINT
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    print()
    print(
        "-" * 65
    )

    print(
        f"Fine-tuning Epoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        "-" * 65
    )

    for batch_index, (
        images,
        labels
    ) in enumerate(
        train_loader
    ):

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        # ----------------------------------------------------
        # IMAGE -> LATENT
        # ----------------------------------------------------

        with torch.no_grad():

            mu, logvar = (
                vae.encode(images)
            )

            latents = mu

        # ----------------------------------------------------
        # CREATE CAPTIONS
        # ----------------------------------------------------

        captions = []

        for label in labels:

            digit = int(
                label.item()
            )

            caption = (
                "a handwritten digit "
                + digit_names[digit]
            )

            captions.append(
                caption
            )

        # ----------------------------------------------------
        # CFG CAPTION DROPOUT
        # ----------------------------------------------------

        dropped_captions = []

        for caption in captions:

            random_value = (
                torch.rand(1).item()
            )

            if random_value < CFG_DROPOUT:

                dropped_captions.append("")

            else:

                dropped_captions.append(
                    caption
                )

        # ----------------------------------------------------
        # TOKENIZATION
        # ----------------------------------------------------

        tokens = torch.stack([

            tokenize(caption)

            for caption
            in dropped_captions

        ]).to(DEVICE)

        # ----------------------------------------------------
        # TEXT ENCODER
        # ----------------------------------------------------

        with torch.no_grad():

            text_features = (
                text_encoder(tokens)
            )

        # ----------------------------------------------------
        # RANDOM TIMESTEP
        # ----------------------------------------------------

        t = torch.randint(
            0,
            TIMESTEPS,
            (
                images.size(0),
            ),
            device=DEVICE
        )

        # ----------------------------------------------------
        # RANDOM NOISE
        # ----------------------------------------------------

        noise = torch.randn_like(
            latents
        )

        # ----------------------------------------------------
        # ADD NOISE
        # ----------------------------------------------------

        noisy_latents = add_noise(
            latents,
            noise,
            t
        )

        # ----------------------------------------------------
        # PREDICT NOISE
        # ----------------------------------------------------

        predicted_noise = model(
            noisy_latents,
            t,
            text_features
        )

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = F.mse_loss(
            predicted_noise,
            noise
        )

        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += (
            loss.item()
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            batch_index % 100 == 0
        ):

            print(
                f"Batch "
                f"{batch_index:4d}/"
                f"{len(train_loader):4d} | "
                f"Loss: "
                f"{loss.item():.6f}"
            )

    # ========================================================
    # TRAINING LOSS
    # ========================================================

    average_loss = (
        total_loss
        /
        len(train_loader)
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_loss = (
        evaluate_validation()
    )

    print()
    print(
        f"Epoch {epoch + 1} "
        f"training loss: "
        f"{average_loss:.6f}"
    )

    print(
        f"Epoch {epoch + 1} "
        f"validation MSE: "
        f"{validation_loss:.6f}"
    )

    print(
        f"Current BEST MSE: "
        f"{best_loss:.6f}"
    )

    # ========================================================
    # CHECK IMPROVEMENT
    # ========================================================

    if validation_loss < best_loss:

        best_loss = validation_loss

        torch.save(
            {
                "model":
                    model.state_dict(),

                "text_encoder":
                    text_encoder.state_dict(),

                "validation_mse":
                    best_loss,

                "source":
                    "safe_finetune"
            },
            BEST_CHECKPOINT
        )

        print()
        print(
            "NEW BEST CHECKPOINT SAVED!"
        )

        print(
            BEST_CHECKPOINT
        )

    else:

        print()
        print(
            "No validation improvement."
        )

        print(
            "Restoring previous BEST checkpoint..."
        )

        best_checkpoint = torch.load(
            BEST_CHECKPOINT,
            map_location=DEVICE
        )

        model.load_state_dict(
            best_checkpoint["model"]
        )

        text_encoder.load_state_dict(
            best_checkpoint["text_encoder"]
        )

        print(
            "BEST checkpoint restored."
        )

        print()
        print(
            "Early stopping."
        )

        break


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 65)
print("SAFE FINE-TUNING COMPLETE")
print("=" * 65)

print()
print(
    "Best validation MSE:",
    f"{best_loss:.6f}"
)

print()
print(
    "Best checkpoint:"
)

print(
    BEST_CHECKPOINT
)

print()
print(
    "Original epoch-5 checkpoint remains untouched:"
)

print(
    START_CHECKPOINT
)

print()
print("=" * 65)