import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 32
TIMESTEPS = 300


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


# Diffusion schedule

beta = torch.linspace(
    1e-4,
    0.02,
    TIMESTEPS,
    device=DEVICE
)

alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


# Load model

model = ConditionalUNet().to(DEVICE)

checkpoint = "checkpoints/conditional_epoch_5.pth"

model.load_state_dict(
    torch.load(checkpoint, map_location=DEVICE)
)

model.eval()

print("Model loaded successfully!")
print("Checkpoint:", checkpoint)


# Choose digit

digit = int(input("\nEnter digit to generate (0-9): "))

if digit < 0 or digit > 9:
    raise ValueError("Digit must be between 0 and 9.")


# Generate 16 samples

num_images = 16

x = torch.randn(
    num_images,
    1,
    IMAGE_SIZE,
    IMAGE_SIZE,
    device=DEVICE
)

labels = torch.full(
    (num_images,),
    digit,
    dtype=torch.long,
    device=DEVICE
)


print("\nStarting conditional reverse diffusion...")


with torch.no_grad():

    for t in reversed(range(TIMESTEPS)):

        t_tensor = torch.full(
            (num_images,),
            t,
            dtype=torch.long,
            device=DEVICE
        )

        predicted_noise = model(
            x,
            t_tensor,
            labels
        )

        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (
            1 / torch.sqrt(alpha_t)
        ) * (
            x
            - ((1 - alpha_t) /
               torch.sqrt(1 - alpha_bar_t))
            * predicted_noise
        ) + torch.sqrt(1 - alpha_t) * noise

        if t % 50 == 0:
            print("Reverse diffusion step:", t)


# Convert images

x = (x.clamp(-1, 1) + 1) / 2

fig, axes = plt.subplots(4, 4, figsize=(6, 6))

for i, ax in enumerate(axes.flat):

    ax.imshow(
        x[i].cpu().squeeze(),
        cmap="gray"
    )

    ax.axis("off")

plt.suptitle(
    f"Conditional Diffusion - Digit {digit}"
)

plt.tight_layout()

output_path = f"outputs/conditional_digit_{digit}.png"

plt.savefig(
    output_path,
    dpi=150
)

plt.close()

print("\n====================================")
print("Conditional generation completed!")
print("====================================")
print("\nGenerated image saved to:")
print(output_path)