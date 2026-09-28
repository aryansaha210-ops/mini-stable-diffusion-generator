import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 32
TIMESTEPS = 300
NUM_IMAGES = 16

CHECKPOINT = "checkpoints/diffusion_epoch_5.pth"

os.makedirs("outputs", exist_ok=True)

print("Device:", DEVICE)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


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


model = SimpleUNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model.eval()

print()
print("Model loaded successfully!")
print("Checkpoint:", CHECKPOINT)


@torch.no_grad()
def generate_images():

    x = torch.randn(
        NUM_IMAGES,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE
    )

    print()
    print("Starting reverse diffusion...")
    print()

    for t in reversed(range(TIMESTEPS)):

        timestep = torch.full(
            (NUM_IMAGES,),
            t,
            device=DEVICE,
            dtype=torch.long
        )

        predicted_noise = model(
            x,
            timestep
        )

        alpha_t = alpha[t]
        beta_t = beta[t]
        alpha_bar_t = alpha_bar[t]

        x = (
            1 / torch.sqrt(alpha_t)
        ) * (
            x
            -
            (
                beta_t
                /
                torch.sqrt(1 - alpha_bar_t)
            )
            * predicted_noise
        )

        if t > 0:

            noise = torch.randn_like(x)

            x = x + torch.sqrt(beta_t) * noise

        if t % 50 == 0:
            print("Reverse diffusion step:", t)

    x = (x + 1) / 2

    x = torch.clamp(
        x,
        0,
        1
    )

    return x


images = generate_images()

output_path = "outputs/generated_digits.png"

save_image(
    images,
    output_path,
    nrow=4
)

print()
print("====================================")
print("Generation completed!")
print("====================================")
print()
print("Generated image saved to:")
print(output_path)