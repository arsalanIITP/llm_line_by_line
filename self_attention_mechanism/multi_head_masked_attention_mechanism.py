import sys
from pathlib import Path

import torch
import torch.nn as nn

# causal_masked_am.py sits in the project root, one level up from this file, so
# put that directory on sys.path to allow running this script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from causal_masked_attention_mechanism import CausalAttention_v3  # noqa: E402


class MultiHeadAttentionWrapper(nn.Module):
    """Runs several causal attention heads and concatenates their outputs.

    Each head returns (batch, num_tokens, d_out), so the wrapper returns
    (batch, num_tokens, d_out * num_heads).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # ModuleList, not a plain list, so each head's parameters are registered
        # with this module and picked up by .parameters() and .to(device)
        self.heads = nn.ModuleList(
            [
                CausalAttention_v3(
                    d_in, d_out, context_length, dropout, qkv_bias
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # every head sees the same input; the results are joined along the
        # feature dimension, which is why d_out multiplies by num_heads
        # this for loop is squential and should be optimised for parallel computing
        return torch.cat([head(x) for head in self.heads], dim=-1)


if __name__ == "__main__":
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your
            [0.55, 0.87, 0.66],  # journey
            [0.57, 0.85, 0.64],  # starts
            [0.22, 0.58, 0.33],  # with
            [0.77, 0.25, 0.10],  # one
            [0.05, 0.80, 0.55],  # step
        ]
    )
    # stack the same sentence twice to simulate a batch of 2 -> (2, 6, 3)
    batch = torch.stack((inputs, inputs), dim=0)

    torch.manual_seed(123)
    context_length = batch.shape[1]
    mha = MultiHeadAttentionWrapper(
        d_in=3, d_out=2, context_length=context_length, dropout=0.0, num_heads=2
    )
    context_vecs = mha(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
