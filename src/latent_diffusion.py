import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =============================
# CONFIGURATION
# =============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
LATENT_DIM = 16
TIMESTEPS = 300
EPOCHS = 5
LR = 2e-4


print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =============================
# VAE
# =============================

class VAE(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU()
        )

        # Latent mean
        self.fc_mu = nn.Linear(
            128 * 4 * 4,
            LATENT_DIM
        )

        # Latent log variance
        self.fc_logvar = nn.Linear(
            128 * 4 * 4,
            LATENT_DIM
        )

        # Decoder input
        self.fc_decode = nn.Linear(
            LATENT_DIM,
            128 * 4 * 4
        )

        # Decoder
        self.decoder = nn.Sequential(

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

            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()
        )


    # =============================
    # ENCODER
    # =============================

    def encode(self, x):

        x = self.encoder(x)

        x = x.view(
            x.size(0),
            -1
        )

        mu = self.fc_mu(x)

        logvar = self.fc_logvar(x)

        return mu, logvar


    # =============================
    # DECODER
    # =============================

    def decode(self, z):

        x = self.fc_decode(z)

        x = x.view(
            x.size(0),
            128,
            4,
            4
        )

        x = self.decoder(x)

        return x


# =============================
# LATENT DIFFUSION MODEL
# =============================

class LatentDiffusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Time embedding
        self.time_embedding = nn.Sequential(

            nn.Linear(
                1,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                64
            )
        )

        # Class embedding
        self.class_embedding = nn.Embedding(
            10,
            64
        )

        # Noise prediction network
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


    # =============================
    # FORWARD
    # =============================

    def forward(
        self,
        z,
        t,
        labels
    ):

        # Convert timestep to embedding
        time_emb = self.time_embedding(
            t.float().unsqueeze(1) / TIMESTEPS
        )

        # Convert class label to embedding
        class_emb = self.class_embedding(
            labels
        )

        # Combine conditions
        condition = time_emb + class_emb

        # Combine latent and condition
        x = torch.cat(
            [
                z,
                condition
            ],
            dim=1
        )

        # Predict noise
        predicted_noise = self.network(x)

        return predicted_noise


# =============================
# DATASET
# =============================

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
    root="data",
    train=True,
    download=True,
    transform=transform
)


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =============================
# LOAD TRAINED VAE
# =============================

vae = VAE().to(DEVICE)


vae.load_state_dict(
    torch.load(
        "checkpoints/vae_epoch_5.pth",
        map_location=DEVICE
    )
)


vae.eval()


# Freeze VAE
for parameter in vae.parameters():

    parameter.requires_grad = False


print()
print("VAE loaded successfully!")


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
# CREATE LATENT DIFFUSION MODEL
# =============================

model = LatentDiffusionModel().to(DEVICE)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)


# =============================
# TRAINING
# =============================

print()
print("Starting latent diffusion training...")
print()


for epoch in range(EPOCHS):

    total_loss = 0.0


    for images, labels in dataloader:

        # Move data to GPU
        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        # =============================
        # IMAGE → LATENT
        # =============================

        with torch.no_grad():

            z, _ = vae.encode(images)


        # =============================
        # RANDOM TIMESTEP
        # =============================

        t = torch.randint(
            0,
            TIMESTEPS,
            (
                images.size(0),
            ),
            device=DEVICE
        )


        # =============================
        # RANDOM NOISE
        # =============================

        noise = torch.randn_like(z)


        # =============================
        # GET DIFFUSION VALUES
        # =============================

        sqrt_alpha_bar = torch.sqrt(
            alpha_bar[t]
        ).view(
            -1,
            1
        )


        sqrt_one_minus_alpha_bar = torch.sqrt(
            1 - alpha_bar[t]
        ).view(
            -1,
            1
        )


        # =============================
        # ADD NOISE TO LATENT
        # =============================

        noisy_z = (

            sqrt_alpha_bar * z

            +

            sqrt_one_minus_alpha_bar * noise
        )


        # =============================
        # PREDICT NOISE
        # =============================

        predicted_noise = model(
            noisy_z,
            t,
            labels
        )


        # =============================
        # LOSS
        # =============================

        loss = F.mse_loss(
            predicted_noise,
            noise
        )


        # =============================
        # BACKPROPAGATION
        # =============================

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()


    # =============================
    # AVERAGE LOSS
    # =============================

    average_loss = (
        total_loss / len(dataloader)
    )


    print(
        f"Epoch {epoch + 1} completed | "
        f"Average Loss: {average_loss:.4f}"
    )


    # =============================
    # SAVE CHECKPOINT
    # =============================

    torch.save(
        model.state_dict(),
        f"checkpoints/"
        f"latent_diffusion_epoch_{epoch + 1}.pth"
    )


# =============================
# TRAINING COMPLETE
# =============================

print()
print("====================================")
print("Latent diffusion training completed!")
print("====================================")
print()