import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(
        self,
        original_layer,
        rank=4,
        alpha=1.0
    ):
        super().__init__()

        self.original = original_layer

        self.rank = rank
        self.alpha = alpha

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(
            torch.randn(
                rank,
                in_features
            ) * 0.01
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                out_features,
                rank
            )
        )

        # Freeze original layer
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):

        original_output = self.original(x)

        lora_output = (
            x @ self.lora_A.T
        )

        lora_output = (
            lora_output @ self.lora_B.T
        )

        scaling = self.alpha / self.rank

        return (
            original_output
            + scaling * lora_output
        )
