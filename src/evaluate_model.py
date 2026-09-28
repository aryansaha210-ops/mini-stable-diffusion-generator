import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Device:", DEVICE)
print()
print("Starting model evaluation...")
print()


# --------------------------------------------------
# 1. Load MNIST
# --------------------------------------------------

transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("MNIST test samples:", len(dataset))
print()


# --------------------------------------------------
# 2. Select samples
# --------------------------------------------------

samples = []

for digit in range(10):

    for image, label in dataset:

        if label == digit:
            samples.append((image, label))
            break


# --------------------------------------------------
# 3. Create comparison grid
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    5,
    figsize=(12, 5)
)

for i, (image, label) in enumerate(samples):

    axes[i // 5, i % 5].imshow(
        image.squeeze(),
        cmap="gray"
    )

    axes[i // 5, i % 5].set_title(
        f"Digit {label}"
    )

    axes[i // 5, i % 5].axis("off")


plt.suptitle(
    "MNIST Reference Images",
    fontsize=16
)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "mnist_reference_grid.png"
)

plt.savefig(path, dpi=200)

plt.close()

print("Reference grid saved:")
print(path)
print()


# --------------------------------------------------
# 4. Dataset statistics
# --------------------------------------------------

digit_counts = [0] * 10

for _, label in dataset:
    digit_counts[label] += 1


print("Dataset statistics:")
print()

for digit in range(10):

    print(
        f"Digit {digit}: {digit_counts[digit]} images"
    )

print()


# --------------------------------------------------
# 5. Basic pixel statistics
# --------------------------------------------------

all_pixels = torch.cat(
    [
        image.flatten()
        for image, _ in dataset
    ]
)

mean_pixel = all_pixels.mean().item()
std_pixel = all_pixels.std().item()

print("Pixel statistics:")
print("Mean:", mean_pixel)
print("Standard deviation:", std_pixel)
print()


# --------------------------------------------------
# 6. Plot digit distribution
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    range(10),
    digit_counts
)

plt.xlabel("Digit")
plt.ylabel("Number of Images")
plt.title("MNIST Test Dataset Distribution")

plt.xticks(range(10))

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "mnist_distribution.png"
)

plt.savefig(path, dpi=200)

plt.close()

print("Distribution plot saved:")
print(path)
print()


# --------------------------------------------------
# DONE
# --------------------------------------------------

print("EVALUATION AND VISUALIZATION DONE")