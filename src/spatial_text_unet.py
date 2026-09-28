import torch
import torch.nn as nn
import math


# ============================================
# TIME EMBEDDING
# ============================================

class TimeEmbedding(nn.Module):

    def __init__(self, dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, timestep):

        timestep = timestep.float().unsqueeze(1) / 300.0

        return self.network(timestep)


# ============================================
# CROSS ATTENTION
# ============================================

class CrossAttention(nn.Module):

    def __init__(
        self,
        channels=64,
        text_dim=64,
        heads=4
    ):
        super().__init__()

        self.heads = heads
        self.head_dim = channels // heads

        self.to_q = nn.Linear(
            channels,
            channels
        )

        self.to_k = nn.Linear(
            text_dim,
            channels
        )

        self.to_v = nn.Linear(
            text_dim,
            channels
        )

        self.output = nn.Linear(
            channels,
            channels
        )

    def forward(self, x, text):

        batch, channels, height, width = x.shape

        # [B,C,H,W] → [B,H*W,C]
        x_flat = x.flatten(2).transpose(1, 2)

        q = self.to_q(x_flat)
        k = self.to_k(text)
        v = self.to_v(text)

        # Multi-head attention
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

        # Attention
        attention = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        attention = attention / math.sqrt(
            self.head_dim
        )

        attention = torch.softmax(
            attention,
            dim=-1
        )

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

        # Restore spatial format
        output = output.transpose(
            1,
            2
        ).reshape(
            batch,
            channels,
            height,
            width
        )

        return output


# ============================================
# RESIDUAL BLOCK
# ============================================

class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_dim=128
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1
        )

        self.norm1 = nn.GroupNorm(
            8,
            out_channels
        )

        self.norm2 = nn.GroupNorm(
            8,
            out_channels
        )

        self.time_projection = nn.Linear(
            time_dim,
            out_channels
        )

        self.activation = nn.SiLU()

        if in_channels != out_channels:

            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                1
            )

        else:

            self.skip = nn.Identity()

    def forward(self, x, time_embedding):

        residual = self.skip(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        time = self.time_projection(
            time_embedding
        )

        time = time[:, :, None, None]

        x = x + time

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual

        return self.activation(x)


# ============================================
# TEXT-CONDITIONED SPATIAL U-NET
# ============================================

class SpatialTextUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.time_embedding = TimeEmbedding(
            128
        )

        # ------------------------------------
        # INPUT
        # ------------------------------------

        self.input_conv = nn.Conv2d(
            1,
            64,
            3,
            padding=1
        )

        # ------------------------------------
        # ENCODER
        # ------------------------------------

        self.down1 = ResBlock(
            64,
            64
        )

        self.down2 = ResBlock(
            64,
            128
        )

        self.downsample = nn.Conv2d(
            128,
            128,
            4,
            stride=2,
            padding=1
        )

        # ------------------------------------
        # MIDDLE
        # ------------------------------------

        self.middle = ResBlock(
            128,
            128
        )

        # Convert 128 → 64 for attention
        self.attention_in = nn.Conv2d(
            128,
            64,
            1
        )

        self.cross_attention = CrossAttention(
            channels=64,
            text_dim=64,
            heads=4
        )

        self.attention_out = nn.Conv2d(
            64,
            128,
            1
        )

        self.middle2 = ResBlock(
            128,
            128
        )

        # ------------------------------------
        # DECODER
        # ------------------------------------

        self.upsample = nn.ConvTranspose2d(
            128,
            128,
            4,
            stride=2,
            padding=1
        )

        self.up1 = ResBlock(
            128,
            64
        )

        self.up2 = ResBlock(
            64,
            64
        )

        # ------------------------------------
        # OUTPUT
        # ------------------------------------

        self.output = nn.Conv2d(
            64,
            1,
            3,
            padding=1
        )

    def forward(
        self,
        x,
        timestep,
        text
    ):

        # Time
        time_embedding = self.time_embedding(
            timestep
        )

        # Input
        x = self.input_conv(x)

        # Encoder
        x = self.down1(
            x,
            time_embedding
        )

        x = self.down2(
            x,
            time_embedding
        )

        # Downsample
        x = self.downsample(x)

        # Middle
        x = self.middle(
            x,
            time_embedding
        )

        # ------------------------------------
        # CROSS ATTENTION
        # ------------------------------------

        attention_x = self.attention_in(x)

        attention_output = self.cross_attention(
            attention_x,
            text
        )

        attention_x = attention_x + attention_output

        attention_x = self.attention_out(
            attention_x
        )

        x = x + attention_x

        # Middle
        x = self.middle2(
            x,
            time_embedding
        )

        # Upsample
        x = self.upsample(x)

        # Decoder
        x = self.up1(
            x,
            time_embedding
        )

        x = self.up2(
            x,
            time_embedding
        )

        # Noise prediction
        return self.output(x)


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

    # Noisy spatial latent
    latent = torch.randn(
        4,
        1,
        16,
        16
    ).to(device)

    # Timesteps
    timestep = torch.randint(
        0,
        300,
        (4,)
    ).to(device)

    # Simulated text embeddings
    text = torch.randn(
        4,
        16,
        64
    ).to(device)

    # Model
    model = SpatialTextUNet().to(device)

    output = model(
        latent,
        timestep,
        text
    )

    print()
    print("Latent shape:")
    print(latent.shape)

    print()
    print("Text embedding shape:")
    print(text.shape)

    print()
    print("Output shape:")
    print(output.shape)

    print()
    print(
        "Spatial Text U-Net test successful!"
    )