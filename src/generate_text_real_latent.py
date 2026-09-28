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

VAE_PATH = "checkpoints/spatial_vae_epoch_10.pth"
MODEL_PATH = "checkpoints/text_real_latent_epoch_5.pth"

OUTPUT_PATH = "outputs/text_real_latent_generated.png"


print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# TOKENIZER
# ============================================================

class DeterministicTokenizer:

    def __init__(self):

        self.vocab = {
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

        self.PAD = 0
        self.UNK = 1
        self.START = 2
        self.END = 3

    def encode(self, text):

        words = text.lower().split()

        tokens = [self.START]

        for word in words:
            tokens.append(
                self.vocab.get(word, self.UNK)
            )

        tokens.append(self.END)

        tokens = tokens[:MAX_TOKENS]

        while len(tokens) < MAX_TOKENS:
            tokens.append(self.PAD)

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
                1, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU()
        )

        # IMPORTANT:
        # Must match trained Spatial VAE.

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
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                padding=1
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

    def forward(self, x):

        mu, logvar = self.encode(x)

        return self.decode(mu)


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
# CROSS ATTENTION
# ============================================================

class CrossAttention(nn.Module):

    def __init__(
        self,
        latent_channels=128,
        text_dim=128,
        heads=4
    ):

        super().__init__()

        self.heads = heads

        self.head_dim = (
            latent_channels // heads
        )

        self.scale = self.head_dim ** -0.5

        # IMPORTANT:
        # Names must match training checkpoint.

        self.query = nn.Linear(
            latent_channels,
            latent_channels
        )

        self.key = nn.Linear(
            text_dim,
            latent_channels
        )

        self.value = nn.Linear(
            text_dim,
            latent_channels
        )

        self.output = nn.Linear(
            latent_channels,
            latent_channels
        )

    def forward(
        self,
        x,
        text
    ):

        B, C, H, W = x.shape

        x_flat = x.flatten(
            2
        ).transpose(
            1,
            2
        )

        q = self.query(x_flat)

        k = self.key(text)

        v = self.value(text)

        q = q.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        k = k.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        v = v.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        attention = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        attention = (
            attention * self.scale
        )

        attention = torch.softmax(
            attention,
            dim=-1
        )

        out = torch.matmul(
            attention,
            v
        )

        out = out.transpose(
            1,
            2
        ).contiguous()

        out = out.view(
            B,
            H * W,
            C
        )

        out = self.output(out)

        out = out.transpose(
            1,
            2
        )

        out = out.reshape(
            B,
            C,
            H,
            W
        )

        return out


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

        t = t.float().unsqueeze(-1)

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
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        # IMPORTANT:
        # Names must match training checkpoint.

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
                kernel_size=1
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
# TEXT LATENT DIFFUSION U-NET
# ============================================================

class TextLatentDiffusionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # IMPORTANT:
        # Layer names must match training checkpoint.

        self.input = nn.Conv2d(
            4,
            64,
            kernel_size=3,
            padding=1
        )

        self.res1 = ResBlock(
            64,
            128
        )

        # IMPORTANT:
        # Training used 3x3 here.

        self.down = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.middle1 = ResBlock(
            128,
            128
        )

        self.cross_attention = CrossAttention(
            latent_channels=128,
            text_dim=128,
            heads=4
        )

        self.middle2 = ResBlock(
            128,
            128
        )

        self.up = nn.ConvTranspose2d(
            128,
            128,
            kernel_size=4,
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
            kernel_size=3,
            padding=1
        )

        self.time_embedding = TimeEmbedding()

    def forward(
        self,
        x,
        t,
        text_embeddings
    ):

        time_emb = (
            self.time_embedding(t)
        )

        text_condition = (
            text_embeddings.mean(
                dim=1
            )
        )

        x = self.input(x)

        x1 = self.res1(
            x,
            time_emb,
            text_condition
        )

        x2 = self.down(x1)

        x2 = self.middle1(
            x2,
            time_emb,
            text_condition
        )

        x2 = (
            x2
            + self.cross_attention(
                x2,
                text_embeddings
            )
        )

        x2 = self.middle2(
            x2,
            time_emb,
            text_condition
        )

        x2 = self.up(x2)

        x2 = torch.cat(
            [x2, x1],
            dim=1
        )

        x2 = self.res2(
            x2,
            time_emb,
            text_condition
        )

        x2 = self.res3(
            x2,
            time_emb,
            text_condition
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
# LOAD VAE
# ============================================================

print()
print("Loading Spatial VAE...")

vae = SpatialVAE().to(DEVICE)

vae.load_state_dict(
    torch.load(
        VAE_PATH,
        map_location=DEVICE
    )
)

vae.eval()

for parameter in vae.parameters():

    parameter.requires_grad = False

print(
    "Spatial VAE loaded successfully!"
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "Loading text-conditioned diffusion model..."
)

text_encoder = TextEncoder().to(
    DEVICE
)

model = TextLatentDiffusionUNet().to(
    DEVICE
)

checkpoint = torch.load(
    MODEL_PATH,
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

print(
    "Text-conditioned real latent "
    "diffusion model loaded successfully!"
)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = DeterministicTokenizer()


# ============================================================
# PROMPT
# ============================================================

print()
print("====================================")
print("TEXT TO IMAGE GENERATION")
print("====================================")

prompt = input(
    "Enter prompt: "
)

tokens = tokenizer.encode(
    prompt
)

tokens = tokens.unsqueeze(
    0
).to(DEVICE)


# ============================================================
# TEXT ENCODING
# ============================================================

with torch.no_grad():

    text_embeddings = (
        text_encoder(tokens)
    )


print()
print("Text encoded successfully!")


# ============================================================
# RANDOM LATENT
# ============================================================

z = torch.randn(
    1,
    LATENT_CHANNELS,
    8,
    8,
    device=DEVICE
)


# ============================================================
# REVERSE DIFFUSION
# ============================================================

print()
print(
    "Starting text-conditioned "
    "reverse diffusion..."
)

with torch.no_grad():

    for step in reversed(
        range(TIMESTEPS)
    ):

        t = torch.tensor(
            [step],
            device=DEVICE,
            dtype=torch.long
        )

        predicted_noise = model(
            z,
            t,
            text_embeddings
        )

        alpha = alphas[step]

        alpha_bar = alpha_bars[step]

        beta = betas[step]

        z = (
            1.0 / torch.sqrt(alpha)
        ) * (
            z
            - (
                beta
                / torch.sqrt(
                    1.0 - alpha_bar
                )
            )
            * predicted_noise
        )

        if step > 0:

            noise = torch.randn_like(z)

            z = (
                z
                + torch.sqrt(beta)
                * noise
            )

        if step % 50 == 0:

            print(
                "Diffusion step:",
                step
            )


# ============================================================
# DECODE
# ============================================================

print()
print(
    "Decoding generated latent..."
)

with torch.no_grad():

    generated = vae.decode(z)


# ============================================================
# NORMALIZE IMAGE
# ============================================================

generated = generated.clamp(
    -1,
    1
)

generated = (
    generated + 1.0
) / 2.0

generated = generated.squeeze(
    0
).squeeze(
    0
)

generated = (
    generated.cpu().numpy()
    * 255
).astype("uint8")


# ============================================================
# SAVE
# ============================================================

image = Image.fromarray(
    generated
)

image.save(
    OUTPUT_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("====================================")
print("TEXT GENERATION COMPLETED")
print("====================================")

print(
    "Prompt:",
    prompt
)

print()
print(
    "Generated image saved to:"
)

print(
    OUTPUT_PATH
)