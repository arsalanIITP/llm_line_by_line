from typing import TypedDict


class GPTConfig(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool


GPT_CONFIG_124M: GPTConfig = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}


import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # create embedding layer for token
        self.token_embedding_layer = torch.nn.Embedding(
            config["vocab_size"], config["emb_dim"]
        )
        # create embedding layer for context positional embedding
        self.position_embedding = torch.nn.Embedding(
            config["context_length"], config["emb_dim"]
        )
        # add dropout for masked causal attention
        self.drop_emb = nn.Dropout(config["drop_rate"])
        # Use a placeholder transformerBlock
        self.transformer_block = nn.Sequential(
            *[DummyTransformerBlock(config) for _ in range(config["n_layers"])]
        )
        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, token_ids: torch.Tensor):
        batch_size, seq_len = token_ids.shape
        token_embedding = self.token_embedding_layer(token_ids)
        position_embedding = self.position_embedding(
            torch.arange(seq_len, device=token_ids.device)
        )
        x = token_embedding + position_embedding
        x = self.drop_emb(x)
        x = self.transformer_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        # last layer in NN called logits
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps=1e-5):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


import tiktoken


def get_batch():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    return batch


torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(get_batch())
print(logits.shape)
print(logits)
