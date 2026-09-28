import torch
import torch.nn as nn


# =============================
# CONFIGURATION
# =============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 16
TEXT_DIM = 64
NUM_TOKENS = 16

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =============================
# SIMPLE TEXT ENCODER
# =============================

class TextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size=10000,
        embedding_dim=64
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=128,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        x = self.encoder(x)

        return x


# =============================
# CROSS ATTENTION
# =============================

class CrossAttention(nn.Module):

    def __init__(
        self,
        latent_dim=16,
        text_dim=64,
        attention_dim=64
    ):
        super().__init__()

        self.query = nn.Linear(
            latent_dim,
            attention_dim
        )

        self.key = nn.Linear(
            text_dim,
            attention_dim
        )

        self.value = nn.Linear(
            text_dim,
            attention_dim
        )

        self.output = nn.Linear(
            attention_dim,
            latent_dim
        )

        self.scale = attention_dim ** 0.5


    def forward(
        self,
        latent,
        text_features
    ):

        # Latent → Query
        Q = self.query(
            latent
        ).unsqueeze(1)

        # Text → Key
        K = self.key(
            text_features
        )

        # Text → Value
        V = self.value(
            text_features
        )

        # Attention scores
        attention_scores = torch.matmul(
            Q,
            K.transpose(
                1,
                2
            )
        ) / self.scale

        # Attention probabilities
        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        # Weighted text representation
        attended_text = torch.matmul(
            attention_weights,
            V
        )

        attended_text = attended_text.squeeze(1)

        # Project back to latent dimension
        output = self.output(
            attended_text
        )

        return output


# =============================
# TEXT-CONDITIONED MODEL
# =============================

class TextConditionedModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.text_encoder = TextEncoder(
            vocab_size=10000,
            embedding_dim=TEXT_DIM
        )

        self.cross_attention = CrossAttention(
            latent_dim=LATENT_DIM,
            text_dim=TEXT_DIM,
            attention_dim=TEXT_DIM
        )

        self.network = nn.Sequential(

            nn.Linear(
                LATENT_DIM,
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
        tokens
    ):

        # Encode text
        text_features = self.text_encoder(
            tokens
        )

        # Cross attention
        attended = self.cross_attention(
            latent,
            text_features
        )

        # Add conditioning
        conditioned_latent = (
            latent + attended
        )

        # Process conditioned latent
        output = self.network(
            conditioned_latent
        )

        return output


# =============================
# CREATE MODEL
# =============================

model = TextConditionedModel().to(
    DEVICE
)


# =============================
# TEST
# =============================

batch_size = 4

# Simulated token IDs
tokens = torch.randint(
    0,
    10000,
    (
        batch_size,
        NUM_TOKENS
    ),
    device=DEVICE
)


# Simulated latent vectors
latent = torch.randn(
    batch_size,
    LATENT_DIM,
    device=DEVICE
)


# Forward pass
output = model(
    latent,
    tokens
)


print()
print("====================================")
print("Text conditioning test successful!")
print("====================================")

print()
print("Input latent shape:")
print(latent.shape)

print()
print("Text token shape:")
print(tokens.shape)

print()
print("Output shape:")
print(output.shape)