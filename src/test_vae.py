import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 32
LATENT_DIM = 16


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

        self.fc_mu = nn.Linear(128 * 4 * 4, LATENT_DIM)
        self.fc_logvar = nn.Linear(128 * 4 * 4, LATENT_DIM)

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


# Load dataset

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)


# Load model

model = VAE().to(DEVICE)

model.load_state_dict(
    torch.load(
        "checkpoints/vae_epoch_5.pth",
        map_location=DEVICE
    )
)

model.eval()

print("VAE loaded successfully!")


# Get images

images, labels = next(iter(dataloader))

images = images.to(DEVICE)


# Reconstruct

with torch.no_grad():

    reconstructed, _, _ = model(images)


# Convert from [-1, 1] to [0, 1]

images = (images.clamp(-1, 1) + 1) / 2
reconstructed = (reconstructed.clamp(-1, 1) + 1) / 2


# Display original and reconstructed

fig, axes = plt.subplots(
    2,
    8,
    figsize=(12, 4)
)

for i in range(8):

    axes[0, i].imshow(
        images[i].cpu().squeeze(),
        cmap="gray"
    )

    axes[0, i].set_title(
        f"Original: {labels[i].item()}"
    )

    axes[0, i].axis("off")

    axes[1, i].imshow(
        reconstructed[i].cpu().squeeze(),
        cmap="gray"
    )

    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis("off")


plt.tight_layout()

output_path = "outputs/vae_reconstruction.png"

plt.savefig(
    output_path,
    dpi=150
)

plt.close()

print("\nReconstruction completed!")
print("Saved to:")
print(output_path)