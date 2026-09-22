import torch
import torch.nn as nn
import tiktoken


# Step 1 :- Create tokenizer word -> integer
def tokenize(sentence: str) -> list[int]:
    encoding = tiktoken.get_encoding("gpt2")
    encoded_text = encoding.encode(sentence)
    return encoded_text


# Step 2 :- Create embedding layer tokens -> N dimentinal embeddings
class EmbeddingLayer:
    def __init__(self):
        self._vocab_size = 50257
        self._embedding_dim = 3
        self._embedding_layer = torch.nn.Embedding(
            self._vocab_size, self._embedding_dim
        )

    def get_embeddings(self, token_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor(token_ids)
        return self._embedding_layer(ids)


input_text = "Your journey starts with one step"

tokenize_input_text = tokenize(input_text)
embeddingLayer = EmbeddingLayer()
embedded_input_text = embeddingLayer.get_embeddings(tokenize_input_text)

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

# Previously, you used raw embeddings directly to compute similarity.
# But the model couldn't learn what to look for


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        # nn.linear have weights and biases we dont need biases here qkv_bias: bool = False
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, embedding_input: torch.Tensor) -> torch.Tensor:
        # this is a research are why 3 wts why not 2 or 7
        # 3 wt was mention in OG paper Attention all you need and imperically better
        keys = self.W_query(embedding_input)
        queries = self.W_query(embedding_input)
        values = self.W_value(embedding_input)
        # Attention scores measure how much each token should "look at" every other token,
        # computed as the dot-product similarity between what each token is asking for (query)
        # and what each token advertises about itself (key).
        attn_scores = queries @ keys.T
        # normalization of scores to make more training efficient and probabilistic
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # context vector is actually context aware where context is list of n dimentional embeddings
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(3, 2)
print(sa_v2(inputs))
