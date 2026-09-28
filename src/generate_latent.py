import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =============================
# CONFIGURATION
# =============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 16
TIMESTEPS = 300

CHECKPOINT = "checkpoints/latent_diffusion_epoch_5.pth"

OUTPUT = "outputs/latent_generated.png"


print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =============================
# VAE
# =============================

class VAE(nn.Module):

    def __init__(self):
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

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def decode(self, z):

        x = self.fc_decode(z)

        x = x.view(
            x.size(0),
            128,
            4,
            4
        )

        return self.decoder(x)


# =============================
# LATENT DIFFUSION MODEL
# =============================

class LatentDiffusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.time_embedding = nn.Sequential(

            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.class_embedding = nn.Embedding(
            10,
            64
        )

        self.network = nn.Sequential(

            nn.Linear(
                LATENT_DIM + 64,
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
        z,
        t,
        labels
    ):

        time_emb = self.time_embedding(
            t.float().unsqueeze(1) / TIMESTEPS
        )

        class_emb = self.class_embedding(
            labels
        )

        condition = time_emb + class_emb

        x = torch.cat(
            [
                z,
                condition
            ],
            dim=1
        )

        return self.network(x)


# =============================
# LOAD VAE
# =============================

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


# =============================
# LOAD LATENT DIFFUSION MODEL
# =============================

model = LatentDiffusionModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model.eval()

print("Latent diffusion model loaded successfully!")


# =============================
# DIFFUSION SCHEDULE
# =============================

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


# =============================
# SELECT DIGIT
# =============================

digit = int(
    input("Enter digit to generate (0-9): ")
)

if digit < 0 or digit > 9:

    raise ValueError(
        "Digit must be between 0 and 9."
    )


# Generate 16 images

NUM_IMAGES = 16


labels = torch.full(
    (NUM_IMAGES,),
    digit,
    dtype=torch.long,
    device=DEVICE
)


# =============================
# START FROM RANDOM LATENT
# =============================

z = torch.randn(
    NUM_IMAGES,
    LATENT_DIM,
    device=DEVICE
)


# =============================
# REVERSE DIFFUSION
# =============================

print()
print("Starting latent reverse diffusion...")
print()


with torch.no_grad():

    for t in reversed(
        range(TIMESTEPS)
    ):

        t_batch = torch.full(
            (NUM_IMAGES,),
            t,
            device=DEVICE,
            dtype=torch.long
        )

        # Predict noise

        predicted_noise = model(
            z,
            t_batch,
            labels
        )

        # Current diffusion values

        alpha_t = alpha[t]

        alpha_bar_t = alpha_bar[t]

        beta_t = beta[t]


        # DDPM reverse step

        z = (
            1 / torch.sqrt(alpha_t)
        ) * (
            z
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


        # Add random noise except final step

        if t > 0:

            noise = torch.randn_like(z)

            z = (
                z
                +
                torch.sqrt(beta_t)
                * noise
            )


        if t % 50 == 0:

            print(
                f"Reverse diffusion step: {t}"
            )


# =============================
# LATENT → IMAGE
# =============================

print()
print("Decoding latent representations...")


with torch.no_grad():

    images = vae.decode(z)


# Convert from [-1, 1] to [0, 1]

images = (
    images + 1
) / 2


images = images.clamp(
    0,
    1
)


# =============================
# CREATE GRID
# =============================

fig, axes = plt.subplots(
    4,
    4,
    figsize=(8, 8)
)


for i, ax in enumerate(
    axes.flat
):

    ax.imshow(
        images[i].cpu().squeeze(),
        cmap="gray"
    )

    ax.axis("off")


fig.suptitle(
    f"Latent Diffusion Generated Digit: {digit}"
)


plt.tight_layout()


# =============================
# SAVE IMAGE
# =============================

plt.savefig(
    OUTPUT,
    dpi=150
)

plt.close()


print()
print("====================================")
print("Generation completed!")
print("====================================")
print()
print("Generated image saved to:")
print(OUTPUT)