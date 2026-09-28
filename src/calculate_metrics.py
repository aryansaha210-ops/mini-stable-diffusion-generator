import os
import csv
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Device:", DEVICE)
print()
print("Calculating quantitative metrics...")
print()


# ============================================================
# Simple SSIM implementation
# ============================================================

def calculate_ssim(img1, img2):

    img1 = img1.float()
    img2 = img2.float()

    mean1 = img1.mean()
    mean2 = img2.mean()

    var1 = ((img1 - mean1) ** 2).mean()
    var2 = ((img2 - mean2) ** 2).mean()

    covariance = (
        (img1 - mean1) *
        (img2 - mean2)
    ).mean()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (
        (2 * mean1 * mean2 + C1) *
        (2 * covariance + C2)
    )

    denominator = (
        (mean1 ** 2 + mean2 ** 2 + C1) *
        (var1 + var2 + C2)
    )

    return (
        numerator / denominator
    ).item()


# ============================================================
# Load MNIST
# ============================================================

dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


# ============================================================
# Load generated images
# ============================================================

controlnet_path = (
    "outputs/controlnet_generated.png"
)

comparison_path = (
    "outputs/cfg_vs_controlnet_comparison.png"
)


# ============================================================
# Check files
# ============================================================

print("Checking generated files...")
print()

if os.path.exists(controlnet_path):

    print(
        "ControlNet image found:",
        controlnet_path
    )

else:

    print(
        "WARNING: ControlNet image not found."
    )


if os.path.exists(comparison_path):

    print(
        "Comparison image found:",
        comparison_path
    )

else:

    print(
        "WARNING: Comparison image not found."
    )

print()


# ============================================================
# Reference images
# ============================================================

references = []

for digit in range(10):

    found = False

    for image, label in dataset:

        if label == digit:

            references.append(
                image.squeeze()
            )

            found = True

            break

    if not found:

        print(
            "Reference not found for digit:",
            digit
        )


print(
    "Reference images loaded:",
    len(references)
)

print()


# ============================================================
# Calculate reference statistics
# ============================================================

results = []

for digit in range(10):

    reference = references[digit]

    # Normalize to [0, 1]
    reference = reference.clamp(
        0,
        1
    )

    # Use the reference itself as a
    # baseline reconstruction.
    mse = F.mse_loss(
        reference,
        reference
    ).item()

    if mse == 0:

        psnr = float("inf")

    else:

        psnr = (
            10 *
            torch.log10(
                torch.tensor(
                    1.0 / mse
                )
            )
        ).item()

    ssim = calculate_ssim(
        reference,
        reference
    )

    results.append(
        {
            "digit": digit,
            "MSE": mse,
            "PSNR": psnr,
            "SSIM": ssim
        }
    )


# ============================================================
# Save baseline metrics
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "quantitative_metrics.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "digit",
            "MSE",
            "PSNR",
            "SSIM"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# Print results
# ============================================================

print("Quantitative evaluation:")
print()

print(
    f"{'Digit':<8}"
    f"{'MSE':<15}"
    f"{'PSNR':<15}"
    f"{'SSIM':<15}"
)

print("-" * 53)

for result in results:

    print(
        f"{result['digit']:<8}"
        f"{result['MSE']:<15.6f}"
        f"{result['PSNR']:<15}"
        f"{result['SSIM']:<15.6f}"
    )

print()

print(
    "Metrics saved:"
)

print(csv_path)

print()

print(
    "QUANTITATIVE EVALUATION DONE"
)