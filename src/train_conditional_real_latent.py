import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================
# CONFIG
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 5
LR = 2e-4

TIMESTEPS = 300

LATENT_CHANNELS = 4

NUM_CLASSES = 10

CONDITION_DIM = 128


print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================
# SPATIAL VAE
# ============================================

class SpatialVAE(nn.Module):

    def __init__(self, latent_channels=4):

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
                padding=1
            ),
            nn.ReLU()
        )

        self.mu = nn.Conv2d(
            128,
            latent_channels,
            kernel_size=3,
            padding=1
        )

        self.logvar = nn.Conv2d(
            128,
            latent_channels,
            kernel_size=3,
            padding=1
        )

        self.decoder = nn.Sequential(

            nn.Conv2d(
                latent_channels,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32, 1,
                kernel_size=3,
                padding=1
            ),
            nn.Tanh()
        )


    def encode(self, x):

        features = self.encoder(x)

        mu = self.mu(features)

        logvar = self.logvar(features)

        return mu, logvar


    def decode(self, z):

        return self.decoder(z)


# ============================================
# TIME EMBEDDING
# ============================================

class TimeEmbedding(nn.Module):

    def __init__(self, dim=128):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                1,
                dim
            ),

            nn.SiLU(),

            nn.Linear(
                dim,
                dim
            )
        )


    def forward(self, timestep):

        timestep = timestep.float()

        timestep = timestep.unsqueeze(1)

        timestep = timestep / TIMESTEPS

        return self.network(
            timestep
        )


# ============================================
# RESIDUAL BLOCK
# ============================================

class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        condition_dim=128
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

        self.time_projection = nn.Linear(
            128,
            out_channels
        )

        self.condition_projection = nn.Linear(
            condition_dim,
            out_channels
        )

        self.activation = nn.SiLU()

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
        time_embedding,
        condition
    ):

        h = self.conv1(x)

        h = self.activation(h)

        # Time conditioning

        time_condition = self.time_projection(
            time_embedding
        )

        time_condition = time_condition[
            :, :, None, None
        ]

        # Class conditioning

        class_condition = self.condition_projection(
            condition
        )

        class_condition = class_condition[
            :, :, None, None
        ]

        # Combine conditions

        h = (
            h
            +
            time_condition
            +
            class_condition
        )

        h = self.conv2(h)

        h = self.activation(h)

        return h + self.skip(x)


# ============================================
# CONDITIONAL LATENT DIFFUSION U-NET
# ============================================

class ConditionalLatentDiffusionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ------------------------------------
        # Time
        # ------------------------------------

        self.time_embedding = TimeEmbedding(
            128
        )

        # ------------------------------------
        # Class embedding
        # ------------------------------------

        self.class_embedding = nn.Embedding(
            NUM_CLASSES,
            CONDITION_DIM
        )

        # ------------------------------------
        # Input
        # ------------------------------------

        self.input = nn.Conv2d(
            4,
            64,
            kernel_size=3,
            padding=1
        )

        # ------------------------------------
        # Encoder
        # ------------------------------------

        self.res1 = ResBlock(
            64,
            128
        )

        self.down = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # ------------------------------------
        # Middle
        # ------------------------------------

        self.middle1 = ResBlock(
            128,
            128
        )

        self.middle2 = ResBlock(
            128,
            128
        )

        # ------------------------------------
        # Decoder
        # ------------------------------------

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

        # ------------------------------------
        # Output
        # ------------------------------------

        self.output = nn.Conv2d(
            64,
            4,
            kernel_size=3,
            padding=1
        )


    def forward(
        self,
        x,
        timestep,
        labels
    ):

        # Time condition

        time_embedding = self.time_embedding(
            timestep
        )

        # Class condition

        class_condition = self.class_embedding(
            labels
        )

        # Encoder

        x0 = self.input(x)

        x1 = self.res1(
            x0,
            time_embedding,
            class_condition
        )

        x2 = self.down(x1)

        # Middle

        x2 = self.middle1(
            x2,
            time_embedding,
            class_condition
        )

        x2 = self.middle2(
            x2,
            time_embedding,
            class_condition
        )

        # Decoder

        x3 = self.up(x2)

        x3 = torch.cat(
            [x3, x1],
            dim=1
        )

        x3 = self.res2(
            x3,
            time_embedding,
            class_condition
        )

        x3 = self.res3(
            x3,
            time_embedding,
            class_condition
        )

        # Predict noise

        return self.output(x3)


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
# LOAD SPATIAL VAE
# ============================================

vae = SpatialVAE(
    LATENT_CHANNELS
).to(DEVICE)

vae.load_state_dict(
    torch.load(
        "checkpoints/spatial_vae_epoch_10.pth",
        map_location=DEVICE
    )
)

vae.eval()


# Freeze VAE

for parameter in vae.parameters():

    parameter.requires_grad = False


print()
print(
    "Spatial VAE loaded successfully!"
)


# ============================================
# MODEL
# ============================================

model = ConditionalLatentDiffusionUNet().to(
    DEVICE
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)


# ============================================
# DATASET
# ============================================

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
    num_workers=0
)


# ============================================
# TRAIN
# ============================================

print()
print(
    "Starting conditional real latent "
    "diffusion training..."
)


for epoch in range(EPOCHS):

    total_loss = 0.0

    for images, labels in loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        batch_size = images.shape[0]

        # ------------------------------------
        # Encode image
        # ------------------------------------

        with torch.no_grad():

            mu, _ = vae.encode(
                images
            )

            latents = mu

        # ------------------------------------
        # Random timestep
        # ------------------------------------

        timestep = torch.randint(
            0,
            TIMESTEPS,
            (batch_size,),
            device=DEVICE
        )

        # ------------------------------------
        # Random noise
        # ------------------------------------

        noise = torch.randn_like(
            latents
        )

        # ------------------------------------
        # Forward diffusion
        # ------------------------------------

        alpha_bar = alpha_bars[
            timestep
        ]

        alpha_bar = alpha_bar[
            :, None, None, None
        ]

        noisy_latent = (
            torch.sqrt(alpha_bar)
            * latents
            +
            torch.sqrt(
                1.0 - alpha_bar
            )
            * noise
        )

        # ------------------------------------
        # Predict noise
        # ------------------------------------

        predicted_noise = model(
            noisy_latent,
            timestep,
            labels
        )

        # ------------------------------------
        # Loss
        # ------------------------------------

        loss = torch.mean(
            (
                predicted_noise
                -
                noise
            ) ** 2
        )

        # ------------------------------------
        # Backpropagation
        # ------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    average_loss = (
        total_loss
        /
        len(loader)
    )

    print(
        f"Epoch {epoch + 1} completed "
        f"| Average Loss: "
        f"{average_loss:.4f}"
    )


    # ----------------------------------------
    # Save checkpoint
    # ----------------------------------------

    torch.save(
        model.state_dict(),
        f"checkpoints/"
        f"conditional_real_latent_epoch_"
        f"{epoch + 1}.pth"
    )


# ============================================
# COMPLETE
# ============================================

print()

print(
    "===================================="
)

print(
    "CONDITIONAL REAL LATENT TRAINING DONE"
)

print(
    "===================================="
)