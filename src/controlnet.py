import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CONTROLNET-STYLE CONDITIONING NETWORK
# ============================================================

class ControlNetCondition(nn.Module):

    def __init__(self):

        super().__init__()

        # Input:
        # 4-channel spatial latent
        #
        # Condition:
        # 4-channel spatial latent
        #
        # Output:
        # 128-channel controlled feature

        self.condition_encoder = nn.Sequential(

            nn.Conv2d(
                4,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.SiLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.SiLU(),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            )
        )

        # Zero-initialized output layer.
        #
        # This is important for ControlNet-style
        # conditioning because initially the
        # conditioning branch should have almost
        # no effect on the pretrained model.

        self.zero_conv = nn.Conv2d(
            128,
            128,
            kernel_size=1
        )

        nn.init.zeros_(
            self.zero_conv.weight
        )

        nn.init.zeros_(
            self.zero_conv.bias
        )

    def forward(self, condition):

        x = self.condition_encoder(
            condition
        )

        x = self.zero_conv(x)

        return x


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("Testing ControlNet-style network...")
    print()

    model = ControlNetCondition().to(
        DEVICE
    )

    # Fake spatial latent
    condition = torch.randn(
        4,
        4,
        8,
        8
    ).to(DEVICE)

    output = model(
        condition
    )

    print(
        "Input condition shape:",
        condition.shape
    )

    print(
        "Control output shape:",
        output.shape
    )

    print()

    print(
        "ControlNet-style network test successful!"
    )