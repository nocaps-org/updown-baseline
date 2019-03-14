from typing import Optional

import torch
from torch import nn
from allennlp.nn.util import masked_softmax


class LinearAttentionWithProjection(nn.Module):

    def __init__(self,
                 vector_input_size: int,
                 matrix_input_size: int,
                 projection_size: int):
        super().__init__()

        self._vector_projection_layer = nn.Linear(vector_input_size, projection_size, bias=False)
        self._matrix_projection_layer = nn.Linear(matrix_input_size, projection_size, bias=False)
        self._attention_layer = nn.Linear(projection_size, 1, bias=False)

    def forward(self,
                vector: torch.Tensor,
                matrix: torch.Tensor,
                matrix_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # shape: (batch_size, projection_size)
        projected_vector = self._vector_projection_layer(vector)

        # shape: (batch_size, num_candidates, projection_size)
        projected_matrix = self._matrix_projection_layer(matrix)

        # Broadcast vector as matrix for addition.
        # shape: (batch_size, num_candidates, projection_size)
        projected_vector = projected_vector.unsqueeze(1).repeat(1, projected_matrix.size(1), 1)

        # shape: (batch_size, num_candidates, 1)
        attention_logits = self._attention_layer(torch.tanh(projected_vector + projected_matrix))

        # shape: (batch_size, num_candidates)
        attention_logits = attention_logits.squeeze(-1)

        # `\alpha`s as importance weights for candidates (rows) in the `matrix`.
        # shape: (batch_size, num_candidates)
        if matrix_mask is not None:
            attention_weights = masked_softmax(attention_logits, matrix_mask, dim=-1)
        else:
            attention_weights = torch.softmax(attention_logits, dim=-1)

        return attention_weights
