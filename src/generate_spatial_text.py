import torch
import torch.nn as nn
import math
import re

from deterministic_tokenizer import DeterministicTokenizer


# ============================================
# CONFIG
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TIMESTEPS = 300
TEXT_DIM = 64

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================
# VAE
# ============================================

class VAE(nn.Module):

    def __init__(self, latent_dim=16):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(
            128 * 4 * 4,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            128 * 4 * 4,
            latent_dim
        )

        self.fc_decode = nn.Linear(
            latent_dim,
            128 * 4 * 4
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, 4, 2, 1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32, 4, 2, 1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 1, 4, 2, 1
            ),
            nn.Tanh()
        )

    def decode(self, latent):

        x = self.fc_decode(
            latent
        )

        x = x.reshape(
            latent.size(0),
            128,
            4,
            4
        )

        return self.decoder(x)


# ============================================
# TEXT ENCODER
# ============================================

class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            10000,
            TEXT_DIM
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TEXT_DIM,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        return self.encoder(x)


# ============================================
# CROSS ATTENTION
# ============================================

class CrossAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.query = nn.Linear(
            128,
            64
        )

        self.key = nn.Linear(
            64,
            64
        )

        self.value = nn.Linear(
            64,
            64
        )

        self.output = nn.Linear(
            64,
            128
        )

    def forward(
        self,
        latent,
        text
    ):

        B, H, W, C = latent.shape

        latent_flat = latent.reshape(
            B,
            H * W,
            C
        )

        Q = self.query(
            latent_flat
        )

        K = self.key(text)

        V = self.value(text)

        scores = torch.matmul(
            Q,
            K.transpose(
                -2,
                -1
            )
        )

        scores = scores / math.sqrt(64)

        attention = torch.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention,
            V
        )

        output = self.output(
            output
        )

        output = output.reshape(
            B,
            H,
            W,
            128
        )

        return output


# ============================================
# TIME EMBEDDING
# ============================================

class TimeEmbedding(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                1,
                128
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                128
            )
        )

    def forward(self, t):

        t = t.float().unsqueeze(1)

        t = t / TIMESTEPS

        return self.network(t)


# ============================================
# RES BLOCK
# ============================================

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

        self.activation = nn.ReLU()

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
        time_embedding
    ):

        residual = self.skip(x)

        x = self.conv1(x)

        x = self.activation(x)

        time = self.time_projection(
            time_embedding
        )

        time = time[:, :, None, None]

        x = x + time

        x = self.conv2(x)

        x = self.activation(x)

        return x + residual


# ============================================
# SPATIAL TEXT U-NET
# ============================================

class SpatialTextUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.input = nn.Conv2d(
            1,
            64,
            3,
            padding=1
        )

        self.time_embedding = TimeEmbedding()

        self.res1 = ResBlock(
            64,
            128
        )

        self.down = nn.Conv2d(
            128,
            128,
            4,
            2,
            1
        )

        self.middle = ResBlock(
            128,
            128
        )

        self.cross_attention = CrossAttention()

        self.up = nn.ConvTranspose2d(
            128,
            64,
            4,
            2,
            1
        )

        self.res2 = ResBlock(
            64,
            64
        )

        self.output = nn.Conv2d(
            64,
            1,
            3,
            padding=1
        )

    def forward(
        self,
        x,
        t,
        text
    ):

        time = self.time_embedding(t)

        x = self.input(x)

        x = self.res1(
            x,
            time
        )

        x = self.down(x)

        x = self.middle(
            x,
            time
        )

        x_attention = x.permute(
            0,
            2,
            3,
            1
        )

        text_condition = self.cross_attention(
            x_attention,
            text
        )

        text_condition = text_condition.permute(
            0,
            3,
            1,
            2
        )

        x = x + text_condition

        x = self.up(x)

        x = self.res2(
            x,
            time
        )

        return self.output(x)


# ============================================
# LOAD MODELS
# ============================================

vae = VAE(
    latent_dim=16
).to(DEVICE)

vae.load_state_dict(
    torch.load(
        "checkpoints/vae_epoch_5.pth",
        map_location=DEVICE
    )
)

vae.eval()

print()
print("VAE loaded successfully!")


text_encoder = TextEncoder().to(
    DEVICE
)

unet = SpatialTextUNet().to(
    DEVICE
)

checkpoint = torch.load(
    "checkpoints/"
    "spatial_text_deterministic_epoch_5.pth",
    map_location=DEVICE
)

text_encoder.load_state_dict(
    checkpoint["text_encoder"]
)

unet.load_state_dict(
    checkpoint["unet"]
)

text_encoder.eval()
unet.eval()

print(
    "Deterministic spatial text "
    "diffusion model loaded successfully!"
)


# ============================================
# TOKENIZER
# ============================================

tokenizer = DeterministicTokenizer()


# ============================================
# DIFFUSION SCHEDULE
# ============================================

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


# ============================================
# GENERATION FUNCTION
# ============================================

@torch.no_grad()
def generate_image(prompt):

    # ----------------------------------------
    # Tokenize prompt
    # ----------------------------------------

    token_ids = tokenizer.tokenize(
        prompt
    )

    tokens = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )

    print()
    print("Prompt:")
    print(prompt)

    print()
    print("Token IDs:")
    print(token_ids)

    # ----------------------------------------
    # Encode text
    # ----------------------------------------

    text_features = text_encoder(
        tokens
    )

    # ----------------------------------------
    # Start from random noise
    # ----------------------------------------

    latent = torch.randn(
        1,
        1,
        16,
        16,
        device=DEVICE
    )

    print()
    print(
        "Starting spatial reverse diffusion..."
    )

    # ----------------------------------------
    # Reverse diffusion
    # ----------------------------------------

    for t in reversed(
        range(TIMESTEPS)
    ):

        timestep = torch.tensor(
            [t],
            device=DEVICE
        )

        predicted_noise = unet(
            latent,
            timestep,
            text_features
        )

        beta_t = betas[t]

        alpha_t = alphas[t]

        alpha_bar_t = alpha_bars[t]

        # DDPM reverse step

        latent = (
            1
            / torch.sqrt(alpha_t)
        ) * (
            latent
            -
            (
                beta_t
                /
                torch.sqrt(
                    1 - alpha_bar_t
                )
            )
            * predicted_noise
        )

        if t > 0:

            noise = torch.randn_like(
                latent
            )

            latent = (
                latent
                +
                torch.sqrt(beta_t)
                * noise
            )

        if t % 50 == 0:

            print(
                f"Diffusion step: {t}"
            )

    # ----------------------------------------
    # Convert spatial latent back to VAE latent
    # ----------------------------------------

    print()
    print(
        "Decoding latent representation..."
    )

    # Spatial latent:
    # [1,1,16,16]
    #
    # Reduce spatial information back to
    # the original 16-dimensional VAE latent.

    latent_flat = latent.reshape(
        1,
        256
    )

    latent_reduced = torch.nn.functional.adaptive_avg_pool1d(
        latent_flat.unsqueeze(1),
        16
    ).squeeze(1)

    # ----------------------------------------
    # VAE decoding
    # ----------------------------------------

    image = vae.decode(
        latent_reduced
    )

    image = image.clamp(
        -1,
        1
    )

    image = (
        image + 1
    ) / 2

    # ----------------------------------------
    # Save image
    # ----------------------------------------

    from torchvision.utils import save_image

    save_image(
        image,
        "outputs/spatial_text_deterministic.png"
    )

    print()
    print(
        "===================================="
    )

    print(
        "GENERATION COMPLETED"
    )

    print(
        "===================================="
    )

    print()
    print(
        "Saved to:"
    )

    print(
        "outputs/"
        "spatial_text_deterministic.png"
    )


# ============================================
# MAIN
# ============================================

print()
print(
    "===================================="
)

print(
    "DETERMINISTIC SPATIAL TEXT-TO-IMAGE"
)

print(
    "===================================="
)

print()
print(
    "Example:"
)

print(
    "a handwritten digit seven"
)

print()

prompt = input(
    "Enter caption: "
)

generate_image(
    prompt
)