import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
IMAGE_SIZE = 32
TIMESTEPS = 300
EPOCHS = 5
LEARNING_RATE = 2e-4

os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


class SimpleUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.mid = nn.Conv2d(128, 128, 3, padding=1)

        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t):

        t = t.float().unsqueeze(1) / TIMESTEPS

        temb = self.time_embedding(t)
        temb = temb[:, :, None, None]

        x1 = F.relu(self.enc1(x))

        x2 = F.relu(
            self.enc2(
                self.pool(x1)
            )
        )

        x3 = F.relu(self.mid(x2))

        x3 = x3 + temb

        x4 = F.interpolate(
            x3,
            scale_factor=2,
            mode="nearest"
        )

        x4 = F.relu(self.dec1(x4))

        output = self.dec2(x4)

        return output


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


def add_noise(x0, t):

    noise = torch.randn_like(x0)

    sqrt_alpha_bar = torch.sqrt(
        alpha_bar[t]
    ).view(-1, 1, 1, 1)

    sqrt_one_minus_alpha_bar = torch.sqrt(
        1 - alpha_bar[t]
    ).view(-1, 1, 1, 1)

    xt = (
        sqrt_alpha_bar * x0
        + sqrt_one_minus_alpha_bar * noise
    )

    return xt, noise


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
    shuffle=True,
    num_workers=0
)


model = SimpleUNet().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


print()
print("Starting diffusion training...")
print()


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):

        images = images.to(DEVICE)

        t = torch.randint(
            0,
            TIMESTEPS,
            (images.size(0),),
            device=DEVICE
        )

        noisy_images, noise = add_noise(
            images,
            t
        )

        predicted_noise = model(
            noisy_images,
            t
        )

        loss = F.mse_loss(
            predicted_noise,
            noise
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Step [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(dataloader)

    print()
    print(
        f"Epoch {epoch + 1} completed "
        f"| Average Loss: {average_loss:.4f}"
    )

    checkpoint_path = (
        f"checkpoints/diffusion_epoch_{epoch + 1}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(f"Checkpoint saved: {checkpoint_path}")
    print()


print("====================================")
print("Training completed successfully!")
print("====================================")
print("Checkpoints saved in checkpoints/")