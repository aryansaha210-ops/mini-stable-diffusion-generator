import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
IMAGE_SIZE = 32
TIMESTEPS = 300
EPOCHS = 5
LR = 2e-4

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------
# Conditional U-Net
# -----------------------------

class ConditionalUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.class_embedding = nn.Embedding(10, 128)

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t, labels):

        time_emb = self.time_embedding(
            t.float().unsqueeze(1) / TIMESTEPS
        )

        class_emb = self.class_embedding(labels)

        condition = time_emb + class_emb

        condition = condition.unsqueeze(-1).unsqueeze(-1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x + condition

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))
        x = self.conv5(x)

        return x


# -----------------------------
# Dataset
# -----------------------------

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


# -----------------------------
# Diffusion schedule
# -----------------------------

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


def add_noise(x, t):

    noise = torch.randn_like(x)

    sqrt_alpha_bar = torch.sqrt(
        alpha_bar[t]
    ).view(-1, 1, 1, 1)

    sqrt_one_minus_alpha_bar = torch.sqrt(
        1 - alpha_bar[t]
    ).view(-1, 1, 1, 1)

    noisy_image = (
        sqrt_alpha_bar * x
        + sqrt_one_minus_alpha_bar * noise
    )

    return noisy_image, noise


# -----------------------------
# Model
# -----------------------------

model = ConditionalUNet().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)


# -----------------------------
# Training
# -----------------------------

for epoch in range(EPOCHS):

    total_loss = 0

    for images, labels in dataloader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

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
            t,
            labels
        )

        loss = F.mse_loss(
            predicted_noise,
            noise
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
        f"checkpoints/conditional_epoch_{epoch + 1}.pth"
    )

print("\nConditional training completed!")