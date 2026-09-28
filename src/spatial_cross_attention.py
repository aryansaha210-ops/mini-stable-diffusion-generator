import torch
import torch.nn as nn
import math


# ============================================
# CROSS ATTENTION
# ============================================

class CrossAttention(nn.Module):

    def __init__(
        self,
        latent_channels=64,
        text_dim=64,
        heads=4
    ):
        super().__init__()

        self.heads = heads
        self.head_dim = latent_channels // heads

        # Image/latent → Query
        self.to_q = nn.Linear(
            latent_channels,
            latent_channels
        )

        # Text → Key
        self.to_k = nn.Linear(
            text_dim,
            latent_channels
        )

        # Text → Value
        self.to_v = nn.Linear(
            text_dim,
            latent_channels
        )

        self.output = nn.Linear(
            latent_channels,
            latent_channels
        )

    def forward(self, latent, text):

        # latent:
        # [batch, height, width, channels]

        # text:
        # [batch, tokens, text_dim]

        batch, height, width, channels = latent.shape

        # Flatten spatial dimensions
        latent_flat = latent.reshape(
            batch,
            height * width,
            channels
        )

        # Query
        q = self.to_q(latent_flat)

        # Key
        k = self.to_k(text)

        # Value
        v = self.to_v(text)

        # Split into attention heads
        q = q.view(
            batch,
            -1,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch,
            -1,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch,
            -1,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        # Weighted text information
        output = torch.matmul(
            attention,
            v
        )

        # Merge heads
        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            batch,
            height * width,
            channels
        )

        output = self.output(output)

        # Restore spatial dimensions
        output = output.view(
            batch,
            height,
            width,
            channels
        )

        return output


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # Simulated spatial latent
    latent = torch.randn(
        4,
        16,
        16,
        64
    ).to(device)

    # Simulated text embeddings
    text = torch.randn(
        4,
        16,
        64
    ).to(device)

    model = CrossAttention().to(device)

    output = model(
        latent,
        text
    )

    print()
    print("Latent shape:")
    print(latent.shape)

    print()
    print("Text shape:")
    print(text.shape)

    print()
    print("Attention output shape:")
    print(output.shape)

    print()
    print("Cross-attention test successful!")