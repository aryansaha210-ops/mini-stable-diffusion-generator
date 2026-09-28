import torch
import torch.nn as nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 16
TEXT_DIM = 64
NUM_TOKENS = 16
TIMESTEPS = 300


print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =============================
# TEXT ENCODER
# =============================

class TextEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            10000,
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


# =============================
# CROSS ATTENTION
# =============================

class CrossAttention(nn.Module):

    def __init__(self):
        super().__init__()

        # Latent → Query
        self.query = nn.Linear(
            LATENT_DIM,
            TEXT_DIM
        )

        # Text → Key
        self.key = nn.Linear(
            TEXT_DIM,
            TEXT_DIM
        )

        # Text → Value
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

        # Query
        Q = self.query(
            latent
        ).unsqueeze(1)

        # Key
        K = self.key(
            text_features
        )

        # Value
        V = self.value(
            text_features
        )

        # Attention scores
        scores = torch.matmul(
            Q,
            K.transpose(1, 2)
        )

        scores = scores / self.scale

        # Attention weights
        weights = torch.softmax(
            scores,
            dim=-1
        )

        # Weighted text
        attended = torch.matmul(
            weights,
            V
        )

        # [batch, 1, 64] → [batch, 64]
        attended = attended.squeeze(1)

        return attended


# =============================
# TEXT-CONDITIONED DENOISER
# =============================

class TextConditionedDenoiser(nn.Module):

    def __init__(self):
        super().__init__()

        self.text_encoder = TextEncoder()

        self.cross_attention = CrossAttention()

        # Time → 64
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

        # Latent + condition
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

        # =============================
        # TEXT
        # =============================

        text_features = self.text_encoder(
            tokens
        )

        # =============================
        # CROSS ATTENTION
        # =============================

        text_condition = self.cross_attention(
            latent,
            text_features
        )

        # =============================
        # TIME
        # =============================

        time_condition = self.time_embedding(
            timestep.float().unsqueeze(1)
            / TIMESTEPS
        )

        # Both are now [batch, 64]

        condition = (
            text_condition
            + time_condition
        )

        # =============================
        # NOISE PREDICTION
        # =============================

        x = torch.cat(
            [
                latent,
                condition
            ],
            dim=1
        )

        return self.network(x)


# =============================
# TEST
# =============================

model = TextConditionedDenoiser().to(
    DEVICE
)

batch_size = 4


tokens = torch.randint(
    0,
    10000,
    (batch_size, NUM_TOKENS),
    device=DEVICE
)


latent = torch.randn(
    batch_size,
    LATENT_DIM,
    device=DEVICE
)


timesteps = torch.randint(
    0,
    TIMESTEPS,
    (batch_size,),
    device=DEVICE
)


# Forward pass

predicted_noise = model(
    latent,
    timesteps,
    tokens
)


# =============================
# RESULTS
# =============================

print()
print("====================================")
print("TEXT-CONDITIONED DENOISER WORKING")
print("====================================")

print()
print("Latent shape:")
print(latent.shape)

print()
print("Text token shape:")
print(tokens.shape)

print()
print("Predicted noise shape:")
print(predicted_noise.shape)

print()
print("Cross-attention condition shape:")
print(
    torch.Size(
        [batch_size, TEXT_DIM]
    )
)

print()
print(
    "Text → Cross-Attention → "
    "Time Conditioning → Diffusion successful!"
)