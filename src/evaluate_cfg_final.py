import os
import numpy as np
from PIL import Image
from torchvision import datasets


# ============================================================
# CONFIG
# ============================================================

GENERATED_DIR = r".\outputs\cfg_safe_all_digits"
DATA_DIR = r".\data"

DIGIT_NAMES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]


print("=" * 65)
print("FINAL CFG MODEL - QUANTITATIVE GENERATION EVALUATION")
print("=" * 65)


# ============================================================
# LOAD GENERATED IMAGES
# ============================================================

print("\nLoading generated images...")

generated_images = []
true_labels = []

for digit in range(10):

    possible_files = [
        os.path.join(GENERATED_DIR, f"digit_{digit}.png"),
        os.path.join(GENERATED_DIR, f"{digit}.png"),
        os.path.join(GENERATED_DIR, f"generated_digit_{digit}.png"),
    ]

    filename = None

    for path in possible_files:
        if os.path.exists(path):
            filename = path
            break

    if filename is None:
        print(f"WARNING: Could not find image for digit {digit}")
        continue

    image = Image.open(filename).convert("L")
    image = image.resize((32, 32))

    image = np.array(image).astype(np.float32) / 255.0

    generated_images.append(image)
    true_labels.append(digit)

    print(f"Loaded digit {digit}: {filename}")


generated_images = np.array(generated_images)
true_labels = np.array(true_labels)


print("\nGenerated image tensor:")
print(generated_images.shape)


# ============================================================
# LOAD MNIST TEST DATA
# ============================================================

print("\nLoading MNIST test dataset...")

mnist = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True
)

print(f"MNIST test samples: {len(mnist)}")


# ============================================================
# PREPARE MNIST REFERENCE IMAGES
# ============================================================

print("\nPreparing MNIST reference images...")

reference_images = []
reference_labels = []

for i in range(len(mnist)):

    image, label = mnist[i]

    image = np.array(image).astype(np.float32) / 255.0

    image = Image.fromarray(
        (image * 255).astype(np.uint8)
    )

    image = image.resize((32, 32))

    image = np.array(image).astype(np.float32) / 255.0

    reference_images.append(image)
    reference_labels.append(label)


reference_images = np.array(reference_images)
reference_labels = np.array(reference_labels)


print(f"Reference tensor: {reference_images.shape}")


# ============================================================
# NEAREST-NEIGHBOR CLASSIFICATION
# ============================================================

print("\nRunning nearest-neighbor classification...")
print("Comparing generated images against MNIST test images.")

predicted_labels = []
nearest_distances = []


for i, generated in enumerate(generated_images):

    # Mean squared error against every MNIST image
    distances = np.mean(
        (reference_images - generated) ** 2,
        axis=(1, 2)
    )

    nearest_index = np.argmin(distances)

    predicted_digit = reference_labels[nearest_index]
    nearest_distance = distances[nearest_index]

    predicted_labels.append(predicted_digit)
    nearest_distances.append(nearest_distance)

    print(
        f"Requested: {true_labels[i]} "
        f"| Predicted: {predicted_digit} "
        f"| Distance: {nearest_distance:.6f}"
    )


predicted_labels = np.array(predicted_labels)
nearest_distances = np.array(nearest_distances)


# ============================================================
# OVERALL ACCURACY
# ============================================================

correct = np.sum(
    predicted_labels == true_labels
)

total = len(true_labels)

accuracy = correct / total if total > 0 else 0.0


print("\n" + "=" * 65)
print("GENERATION ACCURACY")
print("=" * 65)

print(f"\nCorrect: {correct}/{total}")

print(
    f"Overall generation accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# PER-DIGIT ACCURACY
# ============================================================

print("\nPer-digit accuracy:")

per_digit_accuracy = {}

for digit in range(10):

    mask = true_labels == digit

    count = np.sum(mask)

    if count == 0:
        continue

    correct_digit = np.sum(
        predicted_labels[mask] == true_labels[mask]
    )

    digit_accuracy = correct_digit / count

    per_digit_accuracy[digit] = digit_accuracy

    print(
        f"Digit {digit}: "
        f"{digit_accuracy * 100:.2f}%"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print("\nRows = requested digit")
print("Columns = predicted digit\n")


confusion = np.zeros(
    (10, 10),
    dtype=int
)


for actual, predicted in zip(
    true_labels,
    predicted_labels
):

    confusion[actual, predicted] += 1


print("      Predicted")
print("       " + " ".join(
    f"{i:3d}" for i in range(10)
))

for digit in range(10):

    print(
        f"{digit:3d} | " +
        " ".join(
            f"{value:3d}"
            for value in confusion[digit]
        )
    )


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

print("\n" + "=" * 65)
print("PER-DIGIT PRECISION / RECALL / F1")
print("=" * 65)

print(
    f"\n{'Digit':<8}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1':<15}"
)


f1_scores = []


for digit in range(10):

    true_positive = confusion[digit, digit]

    false_positive = (
        np.sum(confusion[:, digit])
        - true_positive
    )

    false_negative = (
        np.sum(confusion[digit, :])
        - true_positive
    )

    precision = (
        true_positive /
        (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive /
        (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    if precision + recall > 0:

        f1 = (
            2 * precision * recall /
            (precision + recall)
        )

    else:

        f1 = 0.0

    f1_scores.append(f1)

    print(
        f"{digit:<8}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{f1:<15.4f}"
    )


macro_f1 = np.mean(f1_scores)


print(
    f"\nMacro F1: {macro_f1:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = r".\outputs\cfg_final_evaluation.txt"


with open(results_file, "w") as f:

    f.write(
        "FINAL CFG MODEL - QUANTITATIVE EVALUATION\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Correct: {correct}/{total}\n"
    )

    f.write(
        f"Overall accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    f.write("Per-digit accuracy:\n")

    for digit in range(10):

        if digit in per_digit_accuracy:

            f.write(
                f"Digit {digit}: "
                f"{per_digit_accuracy[digit] * 100:.2f}%\n"
            )

    f.write("\nConfusion Matrix:\n")

    f.write(
        "Rows = requested digit\n"
        "Columns = predicted digit\n\n"
    )

    f.write(str(confusion))

    f.write("\n\nPrecision / Recall / F1:\n\n")

    for digit in range(10):

        true_positive = confusion[digit, digit]

        false_positive = (
            np.sum(confusion[:, digit])
            - true_positive
        )

        false_negative = (
            np.sum(confusion[digit, :])
            - true_positive
        )

        precision = (
            true_positive /
            (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )

        recall = (
            true_positive /
            (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

        if precision + recall > 0:

            f1 = (
                2 * precision * recall /
                (precision + recall)
            )

        else:

            f1 = 0.0

        f.write(
            f"Digit {digit}: "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}\n"
        )

    f.write(
        f"\nMacro F1: {macro_f1:.4f}\n"
    )

    f.write(
        "\nNearest-neighbor results:\n"
    )

    for i in range(len(true_labels)):

        f.write(
            f"Requested {true_labels[i]} -> "
            f"Predicted {predicted_labels[i]} | "
            f"Distance {nearest_distances[i]:.6f}\n"
        )


print("\nResults saved to:")
print(os.path.abspath(results_file))


print("\n" + "=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)