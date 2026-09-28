import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = r".\outputs"

PER_DIGIT_ACCURACY = [
    60,
    90,
    30,
    90,
    80,
    60,
    100,
    80,
    30,
    60
]

CONFUSION_MATRIX = np.array([
    [6, 1, 0, 1, 0, 1, 1, 0, 0, 0],
    [0, 9, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 3, 1, 1, 3, 0, 1, 0, 1],
    [0, 0, 0, 9, 0, 1, 0, 0, 0, 0],
    [0, 0, 2, 0, 8, 0, 0, 0, 0, 0],
    [1, 2, 0, 1, 0, 6, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 10, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 4, 0, 3, 0, 0, 3, 0],
    [1, 0, 0, 0, 0, 0, 0, 3, 0, 6]
])

DIGITS = np.arange(10)


# ============================================================
# TRAINING LOSS
# ============================================================

TRAINING_LOSS = [
    0.1670,
    0.1590,
    0.1547,
    0.1529,
    0.1497,
    0.2305,
    0.2000,
    0.1930,
    0.1901,
    0.1859,
    0.1826,
    0.1811,
    0.1806,
    0.1775,
    0.1759,
    0.1750,
    0.1746,
    0.1732,
    0.1713,
    0.1709,
    0.1650,
    0.1628,
    0.1611,
    0.1599,
    0.1592,
    0.1597,
    0.1586,
    0.1580,
    0.1566,
    0.1574
]

EPOCHS = np.arange(
    1,
    len(TRAINING_LOSS) + 1
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PLOT 1 — PER-DIGIT ACCURACY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    DIGITS,
    PER_DIGIT_ACCURACY
)

plt.xlabel(
    "Target Digit"
)

plt.ylabel(
    "Generation Accuracy (%)"
)

plt.title(
    "CFG Generation Accuracy by Digit"
)

plt.xticks(
    DIGITS
)

plt.ylim(
    0,
    110
)

for digit, accuracy in zip(
    DIGITS,
    PER_DIGIT_ACCURACY
):

    plt.text(
        digit,
        accuracy + 2,
        f"{accuracy}%",
        ha="center"
    )

plt.tight_layout()

path_accuracy = os.path.join(
    OUTPUT_DIR,
    "evaluation_per_digit_accuracy.png"
)

plt.savefig(
    path_accuracy,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 2 — CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(9, 8)
)

plt.imshow(
    CONFUSION_MATRIX,
    interpolation="nearest"
)

plt.title(
    "CFG Generation Confusion Matrix"
)

plt.xlabel(
    "Predicted Digit"
)

plt.ylabel(
    "True Digit"
)

plt.xticks(
    DIGITS
)

plt.yticks(
    DIGITS
)

plt.colorbar(
    label="Number of Samples"
)


# Add values inside cells

for i in range(10):

    for j in range(10):

        plt.text(
            j,
            i,
            str(
                CONFUSION_MATRIX[i, j]
            ),
            ha="center",
            va="center"
        )


plt.tight_layout()

path_confusion = os.path.join(
    OUTPUT_DIR,
    "evaluation_confusion_matrix.png"
)

plt.savefig(
    path_confusion,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 3 — TRAINING LOSS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    EPOCHS,
    TRAINING_LOSS,
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Training MSE Loss"
)

plt.title(
    "CFG Text-Conditioned Latent Diffusion Training Loss"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

path_loss = os.path.join(
    OUTPUT_DIR,
    "cfg_training_loss_curve.png"
)

plt.savefig(
    path_loss,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("EVALUATION PLOTS CREATED")
print("=" * 70)

print()

print(
    "Per-digit accuracy:"
)

print(
    os.path.abspath(
        path_accuracy
    )
)

print()

print(
    "Confusion matrix:"
)

print(
    os.path.abspath(
        path_confusion
    )
)

print()

print(
    "Training loss:"
)

print(
    os.path.abspath(
        path_loss
    )
)

print()

print("=" * 70)
print("DONE")
print("=" * 70)