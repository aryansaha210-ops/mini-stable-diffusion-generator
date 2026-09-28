import torch
import torch.nn as nn
from torchvision.utils import save_image


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 16
TEXT_DIM = 64
NUM_TOKENS = 16
TIMESTEPS = 300


print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# TEXT ENCODER
# ============================================================

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


# ============================================================
# CROSS ATTENTION
# ============================================================

class CrossAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.query = nn.Linear(
            LATENT_DIM,
            TEXT_DIM
        )

        self.key = nn.Linear(
            TEXT_DIM,
            TEXT_DIM
        )

        self.value = nn.Linear(
            TEXT_DIM,
            TEXT_DIM
        )

        self.scale = TEXT_DIM ** 0.5

    def forward(
        self,
        latent,
        text_features
    ):

        Q = self.query(
            latent
        ).unsqueeze(1)

        K = self.key(
            text_features
        )

        V = self.value(
            text_features
        )

        scores = torch.matmul(
            Q,
            K.transpose(1, 2)
        )

        scores = scores / self.scale

        weights = torch.softmax(
            scores,
            dim=-1
        )

        attended = torch.matmul(
            weights,
            V
        )

        return attended.squeeze(1)


# ============================================================
# TEXT-CONDITIONED DENOISER
# ============================================================

class TextConditionedDenoiser(nn.Module):

    def __init__(self):
        super().__init__()

        self.text_encoder = TextEncoder()

        self.cross_attention = CrossAttention()

        self.time_embedding = nn.Sequential(

            nn.Linear(
                1,
                TEXT_DIM
            ),

            nn.ReLU(),

            nn.Linear(
                TEXT_DIM,
                TEXT_DIM
            )
        )

        self.network = nn.Sequential(

            nn.Linear(
                LATENT_DIM + TEXT_DIM,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                LATENT_DIM
            )
        )

    def forward(
        self,
        latent,
        timestep,
        tokens
    ):

        text_features = self.text_encoder(
            tokens
        )

        text_condition = self.cross_attention(
            latent,
            text_features
        )

        time_condition = self.time_embedding(
            timestep.float().unsqueeze(1)
            / TIMESTEPS
        )

        condition = (
            text_condition
            + time_condition
        )

        x = torch.cat(
            [
                latent,
                condition
            ],
            dim=1
        )

        return self.network(x)


# ============================================================
# VAE
# ============================================================

class VAE(nn.Module):

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
                4,
                2,
                1
            ),

            nn.ReLU()
        )

        self.fc_mu = nn.Linear(
            128 * 4 * 4,
            LATENT_DIM
        )

        self.fc_logvar = nn.Linear(
            128 * 4 * 4,
            LATENT_DIM
        )

        self.fc_decode = nn.Linear(
            LATENT_DIM,
            128 * 4 * 4
        )

        self.decoder = nn.Sequential(

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

            nn.ConvTranspose2d(
                32,
                1,
                4,
                2,
                1
            ),

            nn.Tanh()
        )

    def encode(self, x):

        h = self.encoder(x)

        h = h.view(
            h.size(0),
            -1
        )

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        return mu, logvar

    def decode(self, z):

        h = self.fc_decode(z)

        h = h.view(
            z.size(0),
            128,
            4,
            4
        )

        return self.decoder(h)


# ============================================================
# LOAD VAE
# ============================================================

vae = VAE().to(DEVICE)

vae.load_state_dict(
    torch.load(
        "checkpoints/vae_epoch_5.pth",
        map_location=DEVICE
    )
)

vae.eval()

print()
print("VAE loaded successfully!")


# ============================================================
# LOAD TEXT-CONDITIONED MODEL
# ============================================================

model = TextConditionedDenoiser().to(
    DEVICE
)

model.load_state_dict(
    torch.load(
        "checkpoints/text_latent_epoch_5.pth",
        map_location=DEVICE
    )
)

model.eval()

print("Text-conditioned model loaded successfully!")


# ============================================================
# DIFFUSION SCHEDULE
# ============================================================

beta = torch.linspace(
    1e-4,
    0.02,
    TIMESTEPS,
    device=DEVICE
)

alpha = 1.0 - beta

alpha_bar = torch.cumprod(
    alpha,
    dim=0
)


# ============================================================
# USER PROMPT
# ============================================================

print()
print("====================================")
print("TEXT-TO-IMAGE GENERATION")
print("====================================")
print()

digit = int(
    input(
        "Enter digit to generate (0-9): "
    )
)

if digit < 0 or digit > 9:

    raise ValueError(
        "Digit must be between 0 and 9."
    )


# ============================================================
# TEXT TOKENS
# ============================================================

tokens = torch.zeros(
    16,
    NUM_TOKENS,
    dtype=torch.long,
    device=DEVICE
)

tokens[:, 0] = digit


# ============================================================
# START FROM RANDOM LATENT
# ============================================================

latent = torch.randn(
    16,
    LATENT_DIM,
    device=DEVICE
)


# ============================================================
# REVERSE DIFFUSION
# ============================================================

print()
print("Starting text-conditioned reverse diffusion...")
print()


with torch.no_grad():

    for t in reversed(
        range(TIMESTEPS)
    ):

        timestep = torch.full(
            (16,),
            t,
            device=DEVICE,
            dtype=torch.long
        )

        predicted_noise = model(
            latent,
            timestep,
            tokens
        )

        alpha_t = alpha[t]

        alpha_bar_t = alpha_bar[t]

        beta_t = beta[t]

        # Reverse diffusion

        latent = (
            1 / torch.sqrt(alpha_t)
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

        # Add noise except at final step

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
                "Reverse diffusion step:",
                t
            )


# ============================================================
# VAE DECODING
# ============================================================

print()
print("Decoding latent representations...")


with torch.no_grad():

    generated_images = vae.decode(
        latent
    )


# ============================================================
# SAVE IMAGE
# ============================================================

generated_images = (
    generated_images + 1
) / 2

save_image(
    generated_images,
    "outputs/text_generated.png",
    nrow=4
)


print()
print("====================================")
print("TEXT-TO-IMAGE GENERATION COMPLETED")
print("====================================")

print()
print("Generated image saved to:")

print(
    "outputs/text_generated.png"
)