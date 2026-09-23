Building an LLM from Scratch
A ground-up implementation guide following the architecture from Attention Is All You Need.

Module 1 — The Engine: Attention Mechanism
The self-attention mechanism from Attention Is All You Need is the core engine of every transformer-based LLM. Rather than learning it as one monolithic block, we build it in four progressive stages — each adding one concept on top of the last, until we arrive at multihead masked attention, the mechanism used in GPT and similar architectures.

Learning Path
Stage 1 — Simple Attention
Compute dot-product similarity between every pair of token embeddings, normalize with softmax, and produce context vectors. No learnable parameters — the embeddings themselves determine attention.

You build: SimpleAttentionMechanism — attention scores from raw embeddings, softmax weights, context vectors.

Stage 2 — Trainable Weighted Attention
Introduce three weight matrices — W_query, W_key, W_value — that project embeddings into separate query, key, and value spaces. The model now learns what to attend to.

You build: SelfAttention_v1 → SelfAttention_v2 using nn.Linear, inheriting from nn.Module for automatic gradient tracking.

Stage 3 — Causal (Masked) Attention with Dropout
Apply a causal mask so each token can only attend to itself and earlier tokens — this prevents information leaking from the future during training. Add dropout to attention weights for regularization.

You build: CausalAttention — masked softmax + dropout, registered buffers for the mask.

Stage 4 — Multihead Attention
Run n parallel attention heads, each with its own Q/K/V projections over a smaller dimension (d_model / n_heads). Concatenate their outputs and project back. Multiple heads let the model attend to different relationship types simultaneously.

You build: MultiHeadAttentionWrapper → MultiHeadAttention (fused, single-matrix version for efficiency).

Quick Reference
Stage	Class	What it adds	Key concept
1	SimpleAttentionMechanism	Dot-product attention on raw embeddings	Context = weighted sum of all tokens
2	SelfAttention_v2	Learnable W_q, W_k, W_v via nn.Linear	Separate query/key/value spaces
3	CausalAttention	Upper-triangle mask + dropout	Tokens can't see the future
4	MultiHeadAttention	n parallel heads, concat + project	Attend to multiple patterns at once
