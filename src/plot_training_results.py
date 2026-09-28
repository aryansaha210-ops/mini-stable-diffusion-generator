import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Training losses recorded during the project
# ============================================================

results = {

    "Spatial VAE": [
        0.4069,
        0.4033,
        0.4033,
        0.0644,
        0.0058,
        0.0044,
        0.0036,
        0.0030,
        0.0027,
        0.0024
    ],

    "Real Latent Diffusion": [
        0.3517,
        0.2124,
        0.1909,
        0.1766,
        0.1668
    ],

    "Conditional Real Latent": [
        0.3551,
        0.2105,
        0.1871,
        0.1742,
        0.1667
    ],

    "Text Real Latent": [
        0.3670,
        0.2162,
        0.1964,
        0.1809,
        0.1703
    ],

    "CFG": [
        0.1670,
        0.1590,
        0.1547,
        0.1529,
        0.1497
    ],

    "ControlNet": [
        3499.1957,
        607.4456,
        465.4615,
        301.6387,
        202.8449
    ]
}


# ============================================================
# Plot 1: All training curves
# ============================================================

plt.figure(figsize=(11, 7))

for name, losses in results.items():

    epochs = range(
        1,
        len(losses) + 1
    )

    plt.plot(
        epochs,
        losses,
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title(
    "Training Loss Across Model Stages"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "training_loss_all_models.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(
    "Saved:",
    path
)


# ============================================================
# Plot 2: Diffusion models excluding ControlNet
# ============================================================

diffusion_models = {

    "Real Latent Diffusion":
        results["Real Latent Diffusion"],

    "Conditional Real Latent":
        results["Conditional Real Latent"],

    "Text Real Latent":
        results["Text Real Latent"],

    "CFG":
        results["CFG"]
}


plt.figure(figsize=(10, 6))

for name, losses in diffusion_models.items():

    epochs = range(
        1,
        len(losses) + 1
    )

    plt.plot(
        epochs,
        losses,
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Training Loss")

plt.title(
    "Diffusion Model Training Comparison"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "diffusion_training_comparison.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(
    "Saved:",
    path
)


# ============================================================
# Plot 3: ControlNet training
# ============================================================

controlnet_losses = results["ControlNet"]

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 6),
    controlnet_losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")

plt.title(
    "ControlNet Training Loss"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "controlnet_training_loss.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(
    "Saved:",
    path
)


# ============================================================
# Done
# ============================================================

print()
print(
    "TRAINING VISUALIZATION DONE"
)