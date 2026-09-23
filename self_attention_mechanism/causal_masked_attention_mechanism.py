import torch
import torch.nn as nn


class CausalAttention_v3(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        drop_out: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(drop_out)
        # causal attention mask: 1s strictly above the diagonal are the "future"
        # positions each token must not be allowed to look at.
        # register_buffer keeps it in state_dict and moves it with .to(device),
        # but it is not a trainable parameter.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, embedding_input: torch.Tensor) -> torch.Tensor:
        # embedding_input is batched now: (batch, num_tokens, d_in)
        num_tokens = embedding_input.shape[1]
        keys = self.W_key(embedding_input)
        queries = self.W_query(embedding_input)
        values = self.W_value(embedding_input)

        # transpose only the last two dims so the batch dimension is preserved
        attn_scores = queries @ keys.transpose(1, 2)
        # mask is built for the max context_length, so slice it to this sequence.
        # In PyTorch, operations with a trailing underscore are performed
        # in-place, avoiding unnecessary memory copies.
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # dropout is applied to the weights so the model cannot lean on any
        # single token; it is a no-op at eval time
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (person 0)
        [0.55, 0.87, 0.66],  # journey  (person 1)
        [0.57, 0.85, 0.64],  # starts   (person 2)
        [0.22, 0.58, 0.33],  # with     (person 3)
        [0.77, 0.25, 0.10],  # one      (person 4)
        [0.05, 0.80, 0.55],
    ]  # step     (person 5)
)

# stack the same sentence twice to simulate a batch of 2 -> (2, 6, 3)
batch = torch.stack((inputs, inputs), dim=0)

torch.manual_seed(789)
context_length = batch.shape[1]
ca = CausalAttention_v3(3, 2, context_length, drop_out=0.0)
print(ca(batch))
