import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = r".\data"
CHECKPOINT_DIR = r".\checkpoints"

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3

MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "mnist_classifier.pth"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("MNIST CNN CLASSIFIER TRAINING")
print("=" * 65)

print(f"\nDevice: {DEVICE}")

if DEVICE == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# DATA TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


# ============================================================
# DATASET
# ============================================================

print("\nLoading MNIST dataset...")

train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

print(
    f"Training samples: {len(train_dataset)}"
)

print(
    f"Test samples: {len(test_dataset)}"
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CNN MODEL
# ============================================================

class MNISTClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # 1 x 28 x 28
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # 32 x 28 x 28
            nn.MaxPool2d(2),

            # 32 x 14 x 14
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # 64 x 14 x 14
            nn.MaxPool2d(2),

            # 64 x 7 x 7
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
# CREATE MODEL
# ============================================================

model = MNISTClassifier().to(DEVICE)

print("\nModel:")
print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 65)
print("TRAINING")
print("=" * 65)


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            test_correct += (
                predictions == labels
            ).sum().item()

            test_total += labels.size(0)


    test_accuracy = (
        test_correct / test_total
    )


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training Loss: {train_loss:.6f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

torch.save(
    {
        "model": model.state_dict(),
        "test_accuracy": test_accuracy,
    },
    MODEL_PATH
)


print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)

print(
    f"\nFinal test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nClassifier saved to:"
)

print(
    os.path.abspath(MODEL_PATH)
)