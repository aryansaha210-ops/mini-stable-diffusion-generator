import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENERATED_DIR = r".\outputs\cfg_safe_all_digits"
CHECKPOINT_PATH = r".\checkpoints\mnist_classifier.pth"
RESULTS_FILE = r".\outputs\cfg_cnn_evaluation.txt"

DIGIT_NAMES = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("FINAL CFG MODEL - CNN GENERATION EVALUATION")
print("=" * 65)

print(f"\nDevice: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# CNN MODEL
# ============================================================

class MNISTClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# LOAD CNN
# ============================================================

print("\nLoading trained MNIST CNN...")

model = MNISTClassifier().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model"])
model.eval()

classifier_accuracy = checkpoint.get(
    "test_accuracy",
    None
)

print("CNN checkpoint loaded.")

if classifier_accuracy is not None:
    print(
        f"CNN test accuracy: "
        f"{classifier_accuracy * 100:.2f}%"
    )


# ============================================================
# LOAD GENERATED IMAGES
# ============================================================

print("\nLoading generated images...")

images = []
true_labels = []

for digit in range(10):

    filename = os.path.join(
        GENERATED_DIR,
        f"digit_{digit}.png"
    )

    if not os.path.exists(filename):

        print(
            f"WARNING: Image for digit {digit} "
            f"not found: {filename}"
        )

        continue

    image = Image.open(
        filename
    ).convert("L")

    # Convert 32x32 generated image to 28x28 MNIST size
    image = image.resize((28, 28))

    image = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0

    # Same normalization used during CNN training
    image = (image - 0.5) / 0.5

    images.append(image)
    true_labels.append(digit)

    print(
        f"Loaded digit {digit}: {filename}"
    )


# ============================================================
# CHECK IMAGES
# ============================================================

if len(images) == 0:

    raise RuntimeError(
        "No generated images were found."
    )


images = np.array(images)
true_labels = np.array(true_labels)


# ============================================================
# CREATE TENSOR
# ============================================================

images_tensor = torch.tensor(
    images,
    dtype=torch.float32
).unsqueeze(1)

images_tensor = images_tensor.to(DEVICE)

print(
    "\nInput tensor shape: "
    f"{tuple(images_tensor.shape)}"
)


# ============================================================
# CNN PREDICTION
# ============================================================

print("\nRunning CNN evaluation...")

with torch.no_grad():

    outputs = model(images_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidences, predictions = torch.max(
        probabilities,
        dim=1
    )


predicted_labels = predictions.cpu().numpy()
confidence_scores = confidences.cpu().numpy()


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("PREDICTIONS")
print("=" * 65)

print(
    f"\n{'Requested':<12}"
    f"{'Predicted':<12}"
    f"{'Confidence':<15}"
    f"Result"
)

for i in range(len(true_labels)):

    requested = true_labels[i]
    predicted = predicted_labels[i]
    confidence = confidence_scores[i]

    result = (
        "CORRECT"
        if requested == predicted
        else "WRONG"
    )

    print(
        f"{requested:<12}"
        f"{predicted:<12}"
        f"{confidence * 100:>6.2f}%"
        f"{'':<8}"
        f"{result}"
    )


# ============================================================
# ACCURACY
# ============================================================

correct = np.sum(
    predicted_labels == true_labels
)

total = len(true_labels)

accuracy = (
    correct / total
    if total > 0
    else 0.0
)

print("\n" + "=" * 65)
print("CNN GENERATION ACCURACY")
print("=" * 65)

print(f"\nCorrect: {correct}/{total}")

print(
    f"Generation accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

confusion = np.zeros(
    (10, 10),
    dtype=int
)

for actual, predicted in zip(
    true_labels,
    predicted_labels
):

    confusion[
        actual,
        predicted
    ] += 1


print("\nRows = requested digit")
print("Columns = predicted digit\n")

print(
    "      " +
    " ".join(
        f"{i:3d}"
        for i in range(10)
    )
)

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
print("PRECISION / RECALL / F1")
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
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive /
        (true_positive + false_negative)
        if true_positive + false_negative > 0
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
# AVERAGE CONFIDENCE
# ============================================================

average_confidence = np.mean(
    confidence_scores
)

print(
    f"Average CNN confidence: "
    f"{average_confidence * 100:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

print("\nSaving evaluation results...")

with open(
    RESULTS_FILE,
    "w"
) as f:

    f.write(
        "FINAL CFG MODEL - CNN EVALUATION\n"
    )

    f.write("=" * 60 + "\n\n")

    if classifier_accuracy is not None:

        f.write(
            f"CNN test accuracy: "
            f"{classifier_accuracy * 100:.2f}%\n\n"
        )

    f.write(
        f"Generated digit accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Correct: {correct}/{total}\n\n"
    )

    f.write("Predictions:\n")

    for i in range(len(true_labels)):

        f.write(
            f"Requested {true_labels[i]} -> "
            f"Predicted {predicted_labels[i]} | "
            f"Confidence "
            f"{confidence_scores[i] * 100:.2f}%\n"
        )

    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion))

    f.write("\n\nPrecision / Recall / F1:\n")

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
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive /
            (true_positive + false_negative)
            if true_positive + false_negative > 0
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
        f"Average CNN confidence: "
        f"{average_confidence * 100:.2f}%\n"
    )


print("\nResults saved to:")
print(os.path.abspath(RESULTS_FILE))

print("\n" + "=" * 65)
print("CNN EVALUATION COMPLETE")
print("=" * 65)