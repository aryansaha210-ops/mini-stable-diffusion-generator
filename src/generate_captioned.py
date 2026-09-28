import torch
import torch.nn as nn
import torch.nn.functional as F


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
# RESIDUAL BLOCK
# ============================================

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim=128):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.time_projection = nn.Linear(
            time_dim,
            out_channels
        )

        self.norm1 = nn.GroupNorm(
            8,
            out_channels
        )

        self.norm2 = nn.GroupNorm(
            8,
            out_channels
        )

        self.activation = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x, time_embedding):

        residual = self.skip(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        time = self.time_projection(time_embedding)
        time = time.unsqueeze(-1).unsqueeze(-1)

        x = x + time

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual

        x = self.activation(x)

        return x


# ============================================
# SPATIAL LATENT U-NET
# ============================================

class SpatialUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.time_embedding = TimeEmbedding(128)

        # ------------------------------------
        # INPUT
        # ------------------------------------

        self.input_conv = nn.Conv2d(
            1,
            64,
            kernel_size=3,
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
            kernel_size=4,
            stride=2,
            padding=1
        )

        # ------------------------------------
        # MIDDLE
        # ------------------------------------

        self.middle1 = ResBlock(
            128,
            128
        )

        self.middle2 = ResBlock(
            128,
            128
        )

        # ------------------------------------
        # UPSAMPLE
        # ------------------------------------

        self.upsample = nn.ConvTranspose2d(
            128,
            128,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # ------------------------------------
        # DECODER
        # ------------------------------------

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
            kernel_size=3,
            padding=1
        )

    def forward(self, x, timestep):

        # Time embedding
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
        x = self.middle1(
            x,
            time_embedding
        )

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

        # Output predicted noise
        x = self.output(x)

        return x


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # Simulated spatial latent
    x = torch.randn(
        4,
        1,
        16,
        16
    ).to(device)

    timestep = torch.randint(
        0,
        300,
        (4,)
    ).to(device)

    model = SpatialUNet().to(device)

    output = model(
        x,
        timestep
    )

    print()
    print("Input shape:")
    print(x.shape)

    print()
    print("Output shape:")
    print(output.shape)

    print()
    print("Spatial U-Net test successful!")
    