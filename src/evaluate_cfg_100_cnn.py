import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = r".\outputs\cfg_100_eval"

CLASSIFIER_CHECKPOINT = (
    r".\checkpoints\mnist_classifier.pth"
)

OUTPUT_FILE = (
    r".\outputs\cfg_100_cnn_evaluation.txt"
)

NUM_CLASSES = 10


# ============================================================
# CNN CLASSIFIER
# EXACT ARCHITECTURE USED DURING TRAINING
# ============================================================

class MNISTClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# START
# ============================================================

print("=" * 70)
print("100-SAMPLE CFG CNN EVALUATION")
print("=" * 70)

print()

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD CNN
# ============================================================

print()
print("Loading CNN classifier...")

model = MNISTClassifier().to(DEVICE)


checkpoint = torch.load(
    CLASSIFIER_CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)


# IMPORTANT:
# The checkpoint contains:
#
# {
#     "model": ...,
#     "test_accuracy": ...
# }
#
# Therefore we load checkpoint["model"].

model.load_state_dict(
    checkpoint["model"]
)

model.eval()


classifier_test_accuracy = (
    checkpoint.get(
        "test_accuracy",
        None
    )
)


print("CNN classifier loaded.")

if classifier_test_accuracy is not None:

    print(
        f"Classifier test accuracy: "
        f"{classifier_test_accuracy * 100:.2f}%"
    )


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (28, 28)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


# ============================================================
# RESULT STORAGE
# ============================================================

all_predictions = []

all_targets = []

all_confidences = []


per_digit_correct = [
    0 for _ in range(NUM_CLASSES)
]

per_digit_total = [
    0 for _ in range(NUM_CLASSES)
]


confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=int
)


# ============================================================
# EVALUATE 100 IMAGES
# ============================================================

print()
print("=" * 70)
print("EVALUATING GENERATED IMAGES")
print("=" * 70)


with torch.no_grad():

    for digit in range(NUM_CLASSES):

        digit_dir = os.path.join(
            IMAGE_DIR,
            str(digit)
        )

        print()
        print(
            f"Digit {digit}:"
        )

        for sample_id in range(10):

            filename = (
                f"sample_{sample_id:02d}.png"
            )

            image_path = os.path.join(
                digit_dir,
                filename
            )


            # ------------------------------------------------
            # Check image exists
            # ------------------------------------------------

            if not os.path.exists(
                image_path
            ):

                print(
                    f"  WARNING: Missing "
                    f"{image_path}"
                )

                continue


            # ------------------------------------------------
            # Load image
            # ------------------------------------------------

            image = Image.open(
                image_path
            ).convert("L")


            # ------------------------------------------------
            # Transform
            # ------------------------------------------------

            image = transform(
                image
            )


            # Add batch dimension

            image = image.unsqueeze(
                0
            ).to(DEVICE)


            # ------------------------------------------------
            # CNN prediction
            # ------------------------------------------------

            logits = model(
                image
            )


            probabilities = torch.softmax(
                logits,
                dim=1
            )


            confidence, prediction = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )


            prediction = (
                prediction.item()
            )

            confidence = (
                confidence.item()
            )


            target = digit


            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            all_predictions.append(
                prediction
            )

            all_targets.append(
                target
            )

            all_confidences.append(
                confidence
            )


            per_digit_total[
                digit
            ] += 1


            if prediction == target:

                per_digit_correct[
                    digit
                ] += 1


            confusion_matrix[
                target,
                prediction
            ] += 1


            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            status = (
                "CORRECT"
                if prediction == target
                else "WRONG"
            )


            print(
                f"  sample {sample_id:02d}: "
                f"{target} -> {prediction} "
                f"{confidence * 100:.2f}% "
                f"{status}"
            )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

all_predictions = np.array(
    all_predictions
)

all_targets = np.array(
    all_targets
)

all_confidences = np.array(
    all_confidences
)


# ============================================================
# OVERALL ACCURACY
# ============================================================

total = len(
    all_targets
)

correct = np.sum(
    all_predictions == all_targets
)

incorrect = (
    total - correct
)


if total > 0:

    accuracy = (
        correct / total
    )

else:

    accuracy = 0.0


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

precision_values = []

recall_values = []

f1_values = []


for digit in range(
    NUM_CLASSES
):

    tp = confusion_matrix[
        digit,
        digit
    ]


    fp = (
        np.sum(
            confusion_matrix[:, digit]
        )
        - tp
    )


    fn = (
        np.sum(
            confusion_matrix[digit, :]
        )
        - tp
    )


    # Precision

    if (
        tp + fp
    ) > 0:

        precision = (
            tp / (tp + fp)
        )

    else:

        precision = 0.0


    # Recall

    if (
        tp + fn
    ) > 0:

        recall = (
            tp / (tp + fn)
        )

    else:

        recall = 0.0


    # F1

    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0


    precision_values.append(
        precision
    )

    recall_values.append(
        recall
    )

    f1_values.append(
        f1
    )


macro_precision = np.mean(
    precision_values
)

macro_recall = np.mean(
    recall_values
)

macro_f1 = np.mean(
    f1_values
)


# ============================================================
# AVERAGE CONFIDENCE
# ============================================================

if len(
    all_confidences
) > 0:

    average_confidence = np.mean(
        all_confidences
    )

else:

    average_confidence = 0.0


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print()

print(
    f"Total images evaluated: "
    f"{total}"
)

print(
    f"Correct predictions:    "
    f"{correct}"
)

print(
    f"Incorrect predictions:  "
    f"{incorrect}"
)

print(
    f"Generation accuracy:    "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro Precision:         "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall:            "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1:                "
    f"{macro_f1:.4f}"
)

print(
    f"Average CNN confidence:  "
    f"{average_confidence * 100:.2f}%"
)


# ============================================================
# PER-DIGIT RESULTS
# ============================================================

print()
print("=" * 70)
print("PER-DIGIT RESULTS")
print("=" * 70)

for digit in range(
    NUM_CLASSES
):

    if per_digit_total[digit] > 0:

        digit_accuracy = (
            per_digit_correct[digit]
            / per_digit_total[digit]
        )

    else:

        digit_accuracy = 0.0


    print(
        f"Digit {digit}: "
        f"{per_digit_correct[digit]}/"
        f"{per_digit_total[digit]} "
        f"({digit_accuracy * 100:.2f}%) "
        f"F1={f1_values[digit]:.4f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(
    "Rows = true digit"
)

print(
    "Columns = predicted digit"
)

print()


header = (
    "       "
    + " ".join(
        f"{i:4d}"
        for i in range(NUM_CLASSES)
    )
)

print(header)


for digit in range(
    NUM_CLASSES
):

    row = " ".join(
        f"{confusion_matrix[digit, j]:4d}"
        for j in range(NUM_CLASSES)
    )

    print(
        f"{digit:4d}   {row}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "100-SAMPLE CFG CNN EVALUATION\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )


    if classifier_test_accuracy is not None:

        f.write(
            f"Independent CNN test accuracy: "
            f"{classifier_test_accuracy * 100:.2f}%\n\n"
        )


    f.write(
        f"Total images evaluated: "
        f"{total}\n"
    )

    f.write(
        f"Correct predictions: "
        f"{correct}\n"
    )

    f.write(
        f"Incorrect predictions: "
        f"{incorrect}\n"
    )

    f.write(
        f"Generation accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Macro Precision: "
        f"{macro_precision:.4f}\n"
    )

    f.write(
        f"Macro Recall: "
        f"{macro_recall:.4f}\n"
    )

    f.write(
        f"Macro F1: "
        f"{macro_f1:.4f}\n"
    )

    f.write(
        f"Average CNN confidence: "
        f"{average_confidence * 100:.2f}%\n"
    )


    # --------------------------------------------------------
    # Per-digit
    # --------------------------------------------------------

    f.write("\n")

    f.write(
        "=" * 70
        + "\n"
    )

    f.write(
        "PER-DIGIT RESULTS\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )


    for digit in range(
        NUM_CLASSES
    ):

        if per_digit_total[digit] > 0:

            digit_accuracy = (
                per_digit_correct[digit]
                / per_digit_total[digit]
            )

        else:

            digit_accuracy = 0.0


        f.write(
            f"Digit {digit}: "
            f"{per_digit_correct[digit]}/"
            f"{per_digit_total[digit]} "
            f"({digit_accuracy * 100:.2f}%) "
            f"Precision={precision_values[digit]:.4f} "
            f"Recall={recall_values[digit]:.4f} "
            f"F1={f1_values[digit]:.4f}\n"
        )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    f.write("\n")

    f.write(
        "=" * 70
        + "\n"
    )

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        "Rows = true digit\n"
    )

    f.write(
        "Columns = predicted digit\n\n"
    )

    f.write(
        header
        + "\n"
    )


    for digit in range(
        NUM_CLASSES
    ):

        row = " ".join(
            f"{confusion_matrix[digit, j]:4d}"
            for j in range(NUM_CLASSES)
        )

        f.write(
            f"{digit:4d}   {row}\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print()

print(
    "Results saved to:"
)

print(
    os.path.abspath(
        OUTPUT_FILE
    )
)

print()