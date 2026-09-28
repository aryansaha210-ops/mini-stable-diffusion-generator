import os
import csv
import math
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms

OUTPUT_DIR = "outputs"

CFG_DIR = os.path.join(
    OUTPUT_DIR,
    "cfg_generated"
)

CONTROLNET_DIR = os.path.join(
    OUTPUT_DIR,
    "controlnet_generated"
)

print("Starting real quantitative evaluation...")
print()


# ============================================================
# Metrics
# ============================================================

def mse(image1, image2):

    return np.mean(
        (image1 - image2) ** 2
    )


def psnr(image1, image2):

    error = mse(
        image1,
        image2
    )

    if error == 0:
        return float("inf")

    return 10 * math.log10(
        1.0 / error
    )


def ssim(image1, image2):

    image1 = image1.astype(
        np.float64
    )

    image2 = image2.astype(
        np.float64
    )

    mean1 = image1.mean()
    mean2 = image2.mean()

    variance1 = np.mean(
        (image1 - mean1) ** 2
    )

    variance2 = np.mean(
        (image2 - mean2) ** 2
    )

    covariance = np.mean(
        (image1 - mean1) *
        (image2 - mean2)
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (
        (2 * mean1 * mean2 + c1) *
        (2 * covariance + c2)
    )

    denominator = (
        (mean1 ** 2 + mean2 ** 2 + c1) *
        (variance1 + variance2 + c2)
    )

    return numerator / denominator


# ============================================================
# Load MNIST
# ============================================================

dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


references = {}

print("Loading reference images...")

for image, label in dataset:

    if label not in references:

        references[label] = (
            image.squeeze(0).numpy()
        )

    if len(references) == 10:
        break

print(
    "Loaded",
    len(references),
    "reference images."
)

print()


# ============================================================
# Evaluate
# ============================================================

results = []


for digit in range(10):

    reference = references[digit]

    cfg_path = os.path.join(
        CFG_DIR,
        f"digit_{digit}.png"
    )

    controlnet_path = os.path.join(
        CONTROLNET_DIR,
        f"digit_{digit}.png"
    )

    cfg_image = np.array(
        Image.open(
            cfg_path
        ).convert("L")
    ) / 255.0

    controlnet_image = np.array(
        Image.open(
            controlnet_path
        ).convert("L")
    ) / 255.0

    # Make sure generated images
    # have the same resolution.
    if cfg_image.shape != reference.shape:

        cfg_image = np.array(
            Image.fromarray(
                (cfg_image * 255).astype(
                    np.uint8
                )
            ).resize(
                (
                    reference.shape[1],
                    reference.shape[0]
                )
            )
        ) / 255.0

    if controlnet_image.shape != reference.shape:

        controlnet_image = np.array(
            Image.fromarray(
                (controlnet_image * 255).astype(
                    np.uint8
                )
            ).resize(
                (
                    reference.shape[1],
                    reference.shape[0]
                )
            )
        ) / 255.0

    cfg_mse = mse(
        reference,
        cfg_image
    )

    cfg_psnr = psnr(
        reference,
        cfg_image
    )

    cfg_ssim = ssim(
        reference,
        cfg_image
    )

    control_mse = mse(
        reference,
        controlnet_image
    )

    control_psnr = psnr(
        reference,
        controlnet_image
    )

    control_ssim = ssim(
        reference,
        controlnet_image
    )

    results.append({
        "digit": digit,

        "CFG_MSE": cfg_mse,
        "CFG_PSNR": cfg_psnr,
        "CFG_SSIM": cfg_ssim,

        "ControlNet_MSE": control_mse,
        "ControlNet_PSNR": control_psnr,
        "ControlNet_SSIM": control_ssim
    })


# ============================================================
# Calculate averages
# ============================================================

avg_cfg_mse = np.mean(
    [r["CFG_MSE"] for r in results]
)

avg_cfg_psnr = np.mean(
    [r["CFG_PSNR"] for r in results]
)

avg_cfg_ssim = np.mean(
    [r["CFG_SSIM"] for r in results]
)

avg_control_mse = np.mean(
    [r["ControlNet_MSE"] for r in results]
)

avg_control_psnr = np.mean(
    [r["ControlNet_PSNR"] for r in results]
)

avg_control_ssim = np.mean(
    [r["ControlNet_SSIM"] for r in results]
)


# ============================================================
# Print results
# ============================================================

print()
print(
    f"{'Digit':<8}"
    f"{'CFG MSE':<14}"
    f"{'CFG PSNR':<14}"
    f"{'CFG SSIM':<14}"
    f"{'CN MSE':<14}"
    f"{'CN PSNR':<14}"
    f"{'CN SSIM':<14}"
)

print("-" * 92)

for r in results:

    print(
        f"{r['digit']:<8}"
        f"{r['CFG_MSE']:<14.6f}"
        f"{r['CFG_PSNR']:<14.3f}"
        f"{r['CFG_SSIM']:<14.4f}"
        f"{r['ControlNet_MSE']:<14.6f}"
        f"{r['ControlNet_PSNR']:<14.3f}"
        f"{r['ControlNet_SSIM']:<14.4f}"
    )


print()
print("AVERAGES")
print()

print(
    "CFG:"
)

print(
    f"  MSE  : {avg_cfg_mse:.6f}"
)

print(
    f"  PSNR : {avg_cfg_psnr:.3f}"
)

print(
    f"  SSIM : {avg_cfg_ssim:.4f}"
)

print()

print(
    "ControlNet:"
)

print(
    f"  MSE  : {avg_control_mse:.6f}"
)

print(
    f"  PSNR : {avg_control_psnr:.3f}"
)

print(
    f"  SSIM : {avg_control_ssim:.4f}"
)


# ============================================================
# Save CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "real_quantitative_metrics.csv"
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
            "CFG_MSE",
            "CFG_PSNR",
            "CFG_SSIM",
            "ControlNet_MSE",
            "ControlNet_PSNR",
            "ControlNet_SSIM"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


print()
print(
    "Results saved:"
)

print(csv_path)

print()
print(
    "REAL QUANTITATIVE EVALUATION DONE"
)