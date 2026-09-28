import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spatial_vae import SpatialVAE


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
LR = 1e-4

START_EPOCH = 20
END_EPOCH = 30

TIMESTEPS = 300

TEXT_DIM = 128
VOCAB_SIZE = 10000
MAX_TOKENS = 16

DROPOUT_PROB = 0.10

VAE_CHECKPOINT = r".\checkpoints\spatial_vae_epoch_10.pth"

CFG_CHECKPOINT = (
    r".\checkpoints\cfg_text_real_latent_epoch_20.pth"
)

OUTPUT_CHECKPOINT = (
    r".\checkpoints\cfg_text_real_latent_epoch_30.pth"
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

        time_condition = self.time_projection(
            time_embedding
        )

        time_condition = time_condition.unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        text_condition = self.text_projection(
            text_embedding.mean(dim=1)
        )

        text_condition = text_condition.unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        h = h + time_condition + text_condition

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
            )
            / (128 ** 0.5),
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
        )

        result = result.permute(
            0,
            3,
            1,
            2
        )

        return x + result


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
# START
# ============================================================

print("=" * 60)
print("EXTENDING CFG TRAINING: EPOCH 20 -> 30")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD VAE
# ============================================================

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

for parameter in vae.parameters():

    parameter.requires_grad = False

print("SpatialVAE loaded.")


# ============================================================
# LOAD CFG MODEL
# ============================================================

print("\nLoading CFG model...")

checkpoint = torch.load(
    CFG_CHECKPOINT,
    map_location=DEVICE
)

model = TextLatentDiffusionUNet().to(DEVICE)

model.load_state_dict(
    checkpoint["model"]
)

print("CFG model loaded.")


# ============================================================
# LOAD TEXT ENCODER
# ============================================================

print("\nLoading text encoder...")

text_encoder = TextEncoder().to(DEVICE)

text_encoder.load_state_dict(
    checkpoint["text_encoder"]
)

text_encoder.eval()

for parameter in text_encoder.parameters():

    parameter.requires_grad = False

print("Text encoder loaded.")


# ============================================================
# MNIST DATASET
# ============================================================

print("\nLoading MNIST...")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ]
)

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

print(
    "Training samples:",
    len(dataset)
)


# ============================================================
# TEXT EMBEDDINGS
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

print("\nPreparing text embeddings...")

text_embeddings = {}

with torch.no_grad():

    for digit in range(10):

        prompt = (
            "a handwritten digit "
            + digit_names[digit]
        )

        tokens = tokenize(
            prompt
        ).unsqueeze(
            0
        ).to(DEVICE)

        embedding = text_encoder(
            tokens
        )

        text_embeddings[digit] = (
            embedding.squeeze(0)
            .detach()
        )


# Empty/unconditional embedding

empty_tokens = tokenize(
    ""
).unsqueeze(
    0
).to(DEVICE)

with torch.no_grad():

    empty_embedding = (
        text_encoder(
            empty_tokens
        )
        .squeeze(0)
        .detach()
    )


print("Text embeddings ready.")


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
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
# TRAINING
# ============================================================

model.train()

for epoch in range(
    START_EPOCH + 1,
    END_EPOCH + 1
):

    total_loss = 0.0

    batches = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # SpatialVAE.encode() returns:
        #
        #     mu, logvar
        #
        # We use mu as the deterministic latent.
        # ----------------------------------------------------

        with torch.no_grad():

            mu, logvar = vae.encode(
                images
            )

            latent = mu

        # ----------------------------------------------------
        # Random diffusion timestep
        # ----------------------------------------------------

        timestep = torch.randint(
            0,
            TIMESTEPS,
            (
                images.shape[0],
            ),
            device=DEVICE
        )

        alpha_bar = alpha_bars[
            timestep
        ].view(
            -1,
            1,
            1,
            1
        )

        # ----------------------------------------------------
        # Random Gaussian noise
        # ----------------------------------------------------

        noise = torch.randn_like(
            latent
        )

        # ----------------------------------------------------
        # Forward diffusion
        # ----------------------------------------------------

        noisy_latent = (
            torch.sqrt(alpha_bar)
            * latent
            +
            torch.sqrt(
                1.0 - alpha_bar
            )
            * noise
        )

        # ----------------------------------------------------
        # Text conditioning
        # ----------------------------------------------------

        batch_text = torch.stack(
            [
                text_embeddings[
                    int(label.item())
                ]
                for label in labels
            ]
        )

        # ----------------------------------------------------
        # Classifier-free guidance dropout
        # ----------------------------------------------------

        dropout_mask = (
            torch.rand(
                images.shape[0],
                device=DEVICE
            )
            < DROPOUT_PROB
        )

        if dropout_mask.any():

            batch_text = batch_text.clone()

            batch_text[
                dropout_mask
            ] = empty_embedding

        # ----------------------------------------------------
        # Predict noise
        # ----------------------------------------------------

        predicted_noise = model(
            noisy_latent,
            timestep,
            batch_text
        )

        # ----------------------------------------------------
        # MSE noise prediction loss
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

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

        batches += 1

    average_loss = (
        total_loss / batches
    )

    print(
        f"Epoch {epoch} completed | "
        f"Average Loss: {average_loss:.4f}"
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

torch.save(
    {
        "model": model.state_dict(),
        "text_encoder": text_encoder.state_dict(),
    },
    OUTPUT_CHECKPOINT
)


print("\n" + "=" * 60)
print("EXTENDED CFG TRAINING COMPLETE")
print("=" * 60)

print(
    "Checkpoint saved:"
)

print(
    OUTPUT_CHECKPOINT
)