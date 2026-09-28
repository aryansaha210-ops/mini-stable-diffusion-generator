import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
IMAGE_SIZE = 32
LATENT_DIM = 16
EPOCHS = 5
LR = 1e-3

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


class VAE(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(128 * 4 * 4, LATENT_DIM)
        self.fc_logvar = nn.Linear(128 * 4 * 4, LATENT_DIM)

        # Decoder
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

    def encode(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)

        return mu + noise * std

    def decode(self, z):

        x = self.fc_decode(z)

        x = x.view(
            x.size(0),
            128,
            4,
            4
        )

        return self.decoder(x)

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar


def vae_loss(
    reconstruction,
    original,
    mu,
    logvar
):

    reconstruction_loss = F.mse_loss(
        reconstruction,
        original,
        reduction="mean"
    )

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return reconstruction_loss + 0.001 * kl_loss


# Dataset

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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


# Model

model = VAE().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# Training

for epoch in range(EPOCHS):

    total_loss = 0

    for images, _ in dataloader:

        images = images.to(DEVICE)

        reconstruction, mu, logvar = model(images)

        loss = vae_loss(
            reconstruction,
            images,
            mu,
            logvar
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch {epoch + 1} completed | "
        f"Average Loss: {average_loss:.4f}"
    )

    torch.save(
        model.state_dict(),
        f"checkpoints/vae_epoch_{epoch + 1}.pth"
    )


print("\nVAE training completed!")