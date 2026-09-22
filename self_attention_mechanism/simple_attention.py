from typing import Any

import torch


class SimpleAttentionMechanism:
    def __init__(self, embedded_input: torch.Tensor):
        self._embedded_input = embedded_input

    def forward(self):
        attention_score = self._embedded_input @ self._embedded_input.T
        attention_weights = torch.softmax(attention_score, dim=-1)
        context_vector = attention_weights @ self._embedded_input
        return context_vector

    # making this callable instance
    def __call__(self) -> Any:
        return self.forward()


embedding_inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (person 0)
        [0.55, 0.87, 0.66],  # journey  (person 1)
        [0.57, 0.85, 0.64],  # starts   (person 2)
        [0.22, 0.58, 0.33],  # with     (person 3)
        [0.77, 0.25, 0.10],  # one      (person 4)
        [0.05, 0.80, 0.55],
    ]  # step     (person 5)
)

sam = SimpleAttentionMechanism(embedding_inputs)
print(sam())
