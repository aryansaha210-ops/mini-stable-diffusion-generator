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
EPOCHS = 5

# Reduced from 1e-3 for stable LoRA fine-tuning
LR = 1e-4

RANK = 4
ALPHA = 1.0

TIMESTEPS = 300

LATENT_CHANNELS = 4
TEXT_DIM = 128
VOCAB_SIZE = 10000
MAX_TOKENS = 16

GRADIENT_CLIP = 1.0

CFG_CHECKPOINT = (
    "checkpoints/cfg_text_real_latent_epoch_5.pth"
)

VAE_CHECKPOINT = (
    "checkpoints/spatial_vae_epoch_10.pth"
)

OUTPUT_DIR = "checkpoints"


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

        tokens.append(
            VOCAB.get(
                word,
                UNK_TOKEN
            )
        )

    tokens.append(END_TOKEN)

    tokens = tokens[:MAX_TOKENS]

    while len(tokens) < MAX_TOKENS:

        tokens.append(PAD_TOKEN)

    return torch.tensor(
        tokens,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


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
# LoRA LINEAR
# ============================================================

class LoRALinear(nn.Module):

    def __init__(
        self,
        original_layer,
        rank=4,
        alpha=1.0
    ):

        super().__init__()

        self.original = original_layer

        self.rank = rank

        self.alpha = alpha

        in_features = original_layer.in_features

        out_features = original_layer.out_features

        # LoRA A
        self.lora_A = nn.Parameter(
            torch.randn(
                rank,
                in_features,
                device=original_layer.weight.device
            ) * 0.005
        )

        # LoRA B starts at zero
        self.lora_B = nn.Parameter(
            torch.zeros(
                out_features,
                rank,
                device=original_layer.weight.device
            )
        )

        # Freeze original layer
        for parameter in self.original.parameters():

            parameter.requires_grad = False

    def forward(self, x):

        original_output = self.original(x)

        lora_output = (
            x @ self.lora_A.T
        )

        lora_output = (
            lora_output @ self.lora_B.T
        )

        scaling = (
            self.alpha / self.rank
        )

        return (
            original_output
            + scaling * lora_output
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

        attention = (
            attention /
            (self.head_dim ** 0.5)
        )

        attention = torch.softmax(
            attention,
            dim=-1
        )

        result = torch.matmul(
            attention,
            V
        )

        result = (
            result.transpose(1, 2)
            .contiguous()
        )

        result = result.view(
            B,
            -1,
            128
        )

        result = self.output(result)

        result = (
            result.transpose(1, 2)
            .reshape(
                B,
                128,
                H,
                W
            )
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

        h = (
            h
            + self.time_projection(
                time_embedding
            ).unsqueeze(-1).unsqueeze(-1)
        )

        h = (
            h
            + self.text_projection(
                text_embedding
            ).unsqueeze(-1).unsqueeze(-1)
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

        time_embedding = (
            self.time_embedding(t)
        )

        text_mean = text.mean(
            dim=1
        )

        x0 = self.input(x)

        x1 = self.res1(
            x0,
            time_embedding,
            text_mean
        )

        x2 = self.down(x1)

        x3 = self.middle1(
            x2,
            time_embedding,
            text_mean
        )

        attention = self.cross_attention(
            x3,
            text
        )

        x3 = x3 + attention

        x3 = self.middle2(
            x3,
            time_embedding,
            text_mean
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
            text_mean
        )

        x4 = self.res3(
            x4,
            time_embedding,
            text_mean
        )

        return self.output(x4)


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
    shuffle=True,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# LOAD SPATIAL VAE
# ============================================================

print()
print("Loading SpatialVAE...")

vae = SpatialVAE().to(
    DEVICE
)

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

for parameter in vae.parameters():

    parameter.requires_grad = False

print("SpatialVAE loaded.")


# ============================================================
# LOAD CFG MODEL
# ============================================================

print()
print("Loading CFG model...")

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

for parameter in text_encoder.parameters():

    parameter.requires_grad = False

print("CFG model loaded.")


# ============================================================
# FREEZE BASE MODEL
# ============================================================

for parameter in model.parameters():

    parameter.requires_grad = False


# ============================================================
# APPLY LoRA
# ============================================================

print()
print("Applying LoRA...")


model.cross_attention.query = LoRALinear(
    model.cross_attention.query,
    rank=RANK,
    alpha=ALPHA
).to(DEVICE)


model.cross_attention.key = LoRALinear(
    model.cross_attention.key,
    rank=RANK,
    alpha=ALPHA
).to(DEVICE)


model.cross_attention.value = LoRALinear(
    model.cross_attention.value,
    rank=RANK,
    alpha=ALPHA
).to(DEVICE)


model.cross_attention.output = LoRALinear(
    model.cross_attention.output,
    rank=RANK,
    alpha=ALPHA
).to(DEVICE)


# ============================================================
# TRAINABLE PARAMETERS
# ============================================================

trainable_parameters = []

for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        trainable_parameters.append(
            parameter
        )

        print(
            "Trainable:",
            name
        )


print()

total_trainable = sum(
    parameter.numel()
    for parameter in trainable_parameters
)

print(
    "Total trainable parameters:",
    total_trainable
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LR,
    betas=(0.9, 0.999),
    weight_decay=0.01
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
# DIGIT NAMES
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


# ============================================================
# LoRA TRAINING
# ============================================================

print()
print("========================================")
print("LoRA TRAINING")
print("========================================")
print()


for epoch in range(
    1,
    EPOCHS + 1
):

    total_loss = 0.0

    for images, labels in loader:

        # ----------------------------------------------------
        # Move data to GPU
        # ----------------------------------------------------

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Encode images into latent space
        # ----------------------------------------------------

        with torch.no_grad():

            mu, logvar = vae.encode(
                images
            )

            latents = mu

        # ----------------------------------------------------
        # Create captions
        # ----------------------------------------------------

        token_list = []

        for label in labels.tolist():

            caption = (
                "a handwritten digit "
                + digit_names[label]
            )

            token_list.append(
                tokenize(caption)
            )

        tokens = torch.cat(
            token_list,
            dim=0
        )

        # Expected:
        # [128, 16]

        # ----------------------------------------------------
        # Text encoding
        # ----------------------------------------------------

        with torch.no_grad():

            text_features = text_encoder(
                tokens
            )

        # Expected:
        # [128, 16, 128]

        # ----------------------------------------------------
        # Random diffusion timestep
        # ----------------------------------------------------

        t = torch.randint(
            0,
            TIMESTEPS,
            (
                images.shape[0],
            ),
            device=DEVICE
        )

        # ----------------------------------------------------
        # Random noise
        # ----------------------------------------------------

        noise = torch.randn_like(
            latents
        )

        # ----------------------------------------------------
        # Alpha bar
        # ----------------------------------------------------

        alpha_bar = alpha_bars[t].view(
            -1,
            1,
            1,
            1
        )

        # ----------------------------------------------------
        # Add noise
        # ----------------------------------------------------

        noisy_latents = (
            torch.sqrt(alpha_bar)
            * latents
            +
            torch.sqrt(
                1.0 - alpha_bar
            )
            * noise
        )

        # ----------------------------------------------------
        # Predict noise
        # ----------------------------------------------------

        predicted_noise = model(
            noisy_latents,
            t,
            text_features
        )

        # ----------------------------------------------------
        # Loss
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

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=GRADIENT_CLIP
        )

        optimizer.step()

        total_loss += loss.item()

    # --------------------------------------------------------
    # Average loss
    # --------------------------------------------------------

    average_loss = (
        total_loss / len(loader)
    )

    print(
        f"Epoch {epoch} completed | "
        f"Average Loss: {average_loss:.4f}"
    )

    # --------------------------------------------------------
    # Save LoRA parameters only
    # --------------------------------------------------------

    lora_state = {}

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            lora_state[name] = (
                parameter.detach()
                .cpu()
            )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    checkpoint_path = (
        f"{OUTPUT_DIR}/"
        f"lora_epoch_{epoch}.pth"
    )

    torch.save(
        lora_state,
        checkpoint_path
    )

    print(
        "LoRA checkpoint saved:",
        checkpoint_path
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("========================================")
print("LoRA TRAINING DONE")
print("========================================")
print()

print(
    "LoRA checkpoints saved in:"
)

print(
    OUTPUT_DIR
)