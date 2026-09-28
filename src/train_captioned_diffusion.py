import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import re


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 16
TEXT_DIM = 64
NUM_TOKENS = 16
TIMESTEPS = 300
VOCAB_SIZE = 10000

BATCH_SIZE = 128
EPOCHS = 5
LR = 2e-4


print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# TOKENIZER
# ============================================================

class SimpleTokenizer:

    def __init__(self):

        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3
        }

    def tokenize(self, text):

        text = text.lower()

        return re.findall(
            r"[a-z0-9]+",
            text
        )

    def word_to_id(self, word):

        if word in self.special_tokens:
            return self.special_tokens[word]

        value = 0

        for character in word:

            value = (
                value * 31
                + ord(character)
            ) % (
                VOCAB_SIZE - 4
            )

        return value + 4

    def encode(
        self,
        text,
        max_length=NUM_TOKENS
    ):

        words = self.tokenize(text)

        tokens = [
            self.special_tokens["<START>"]
        ]

        for word in words:

            tokens.append(
                self.word_to_id(word)
            )

        tokens.append(
            self.special_tokens["<END>"]
        )

        tokens = tokens[:max_length]

        while len(tokens) < max_length:

            tokens.append(
                self.special_tokens["<PAD>"]
            )

        return tokens


tokenizer = SimpleTokenizer()


# ============================================================
# CAPTIONS
# ============================================================

digit_names = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine"
]


def create_caption(label):

    return (
        "a handwritten digit "
        + digit_names[label]
    )


# ============================================================
# TEXT ENCODER
# ============================================================

class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            TEXT_DIM
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TEXT_DIM,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        return self.encoder(x)


# ============================================================
# CROSS ATTENTION
# ============================================================

class CrossAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.query = nn.Linear(
            LATENT_DIM,
            TEXT_DIM
        )

        self.key = nn.Linear(
            TEXT_DIM,
            TEXT_DIM
        )

        self.value = nn.Linear(
            TEXT_DIM,
            TEXT_DIM
        )

        self.scale = TEXT_DIM ** 0.5

    def forward(
        self,
        latent,
        text_features
    ):

        Q = self.query(
            latent
        ).unsqueeze(1)

        K = self.key(
            text_features
        )

        V = self.value(
            text_features
        )

        scores = torch.matmul(
            Q,
            K.transpose(1, 2)
        )

        scores = scores / self.scale

        weights = torch.softmax(
            scores,
            dim=-1
        )

        attended = torch.matmul(
            weights,
            V
        )

        return attended.squeeze(1)


# ============================================================
# TEXT-CONDITIONED DENOISER
# ============================================================

class TextConditionedDenoiser(nn.Module):

    def __init__(self):

        super().__init__()

        self.text_encoder = TextEncoder()

        self.cross_attention = CrossAttention()

        self.time_embedding = nn.Sequential(

            nn.Linear(
                1,
                TEXT_DIM
            ),

            nn.ReLU(),

            nn.Linear(
                TEXT_DIM,
                TEXT_DIM
            )
        )

        self.network = nn.Sequential(

            nn.Linear(
                LATENT_DIM + TEXT_DIM,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                LATENT_DIM
            )
        )

    def forward(
        self,
        latent,
        timestep,
        tokens
    ):

        text_features = self.text_encoder(
            tokens
        )

        text_condition = self.cross_attention(
            latent,
            text_features
        )

        time_condition = self.time_embedding(
            timestep.float().unsqueeze(1)
            / TIMESTEPS
        )

        condition = (
            text_condition
            + time_condition
        )

        x = torch.cat(
            [
                latent,
                condition
            ],
            dim=1
        )

        return self.network(x)


# ============================================================
# VAE
# ============================================================

class VAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                4,
                2,
                1
            ),

            nn.ReLU()
        )

        self.fc_mu = nn.Linear(
            128 * 4 * 4,
            LATENT_DIM
        )

        self.fc_logvar = nn.Linear(
            128 * 4 * 4,
            LATENT_DIM
        )

        self.fc_decode = nn.Linear(
            LATENT_DIM,
            128 * 4 * 4
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                1,
                4,
                2,
                1
            ),

            nn.Tanh()
        )

    def encode(self, x):

        h = self.encoder(x)

        h = h.view(
            h.size(0),
            -1
        )

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        return mu, logvar


# ============================================================
# LOAD VAE
# ============================================================

vae = VAE().to(DEVICE)

vae.load_state_dict(
    torch.load(
        "checkpoints/vae_epoch_5.pth",
        map_location=DEVICE
    )
)

vae.eval()

for parameter in vae.parameters():
    parameter.requires_grad = False


print()
print("VAE loaded successfully!")


# ============================================================
# DATASET
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (32, 32)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# DIFFUSION SCHEDULE
# ============================================================

beta = torch.linspace(
    1e-4,
    0.02,
    TIMESTEPS,
    device=DEVICE
)

alpha = 1.0 - beta

alpha_bar = torch.cumprod(
    alpha,
    dim=0
)


# ============================================================
# MODEL
# ============================================================

model = TextConditionedDenoiser().to(
    DEVICE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

loss_function = nn.MSELoss()


# ============================================================
# TRAINING
# ============================================================

print()
print("Starting caption-conditioned training...")
print()


for epoch in range(EPOCHS):

    total_loss = 0.0

    for images, labels in loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        # ----------------------------------------------------
        # IMAGE -> LATENT
        # ----------------------------------------------------

        with torch.no_grad():

            mu, logvar = vae.encode(
                images
            )

            latent = mu

        # ----------------------------------------------------
        # CREATE CAPTIONS
        # ----------------------------------------------------

        captions = []

        for label in labels:

            captions.append(
                create_caption(
                    label.item()
                )
            )

        # ----------------------------------------------------
        # TOKENIZE CAPTIONS
        # ----------------------------------------------------

        token_ids = []

        for caption in captions:

            token_ids.append(
                tokenizer.encode(
                    caption
                )
            )

        tokens = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=DEVICE
        )

        # ----------------------------------------------------
        # RANDOM TIMESTEP
        # ----------------------------------------------------

        t = torch.randint(
            0,
            TIMESTEPS,
            (
                images.size(0),
            ),
            device=DEVICE
        )

        # ----------------------------------------------------
        # RANDOM NOISE
        # ----------------------------------------------------

        noise = torch.randn_like(
            latent
        )

        # ----------------------------------------------------
        # FORWARD DIFFUSION
        # ----------------------------------------------------

        sqrt_alpha_bar = torch.sqrt(
            alpha_bar[t]
        ).unsqueeze(1)

        sqrt_one_minus_alpha_bar = torch.sqrt(
            1.0 - alpha_bar[t]
        ).unsqueeze(1)

        noisy_latent = (
            sqrt_alpha_bar * latent
            +
            sqrt_one_minus_alpha_bar * noise
        )

        # ----------------------------------------------------
        # PREDICT NOISE
        # ----------------------------------------------------

        predicted_noise = model(
            noisy_latent,
            t,
            tokens
        )

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = loss_function(
            predicted_noise,
            noise
        )

        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    # --------------------------------------------------------
    # EPOCH RESULT
    # --------------------------------------------------------

    average_loss = (
        total_loss / len(loader)
    )

    print(
        f"Epoch {epoch + 1} completed "
        f"| Average Loss: {average_loss:.4f}"
    )

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        f"checkpoints/"
        f"captioned_diffusion_epoch_"
        f"{epoch + 1}.pth"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("====================================")
print("CAPTION-CONDITIONED TRAINING DONE")
print("====================================")