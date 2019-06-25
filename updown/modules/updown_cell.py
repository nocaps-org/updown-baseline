from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from allennlp.nn.util import masked_mean

from updown.modules.attention import BottomUpTopDownAttention


class UpDownCell(nn.Module):
    r"""
    The basic computation unit of :class:`~updown.models.updown_captioner.UpDownCaptioner`.

    Extended Summary
    ----------------
    The architecture (`Anderson et al. 2017 (Fig. 3) <https://arxiv.org/abs/1707.07998>`_)
    is as follows:

    .. code-block:: text

                                        h2 (t)
                                         .^.
                                          |
                           +--------------------------------+
            h2 (t-1) ----> |         Language LSTM          | ----> h2 (t)
                           +--------------------------------+
                             .^.         .^.
                              |           |
        bottom-up     +----------------+  |
        features  --> | BUTD Attention |  |
                      +----------------+  |
                             .^.          |
                              |___________|
                                          |
                           +--------------------------------+
            h1 (t-1) ----> |         Attention LSTM         | ----> h1 (t)
                           +--------------------------------+
                                         .^.
                        __________________|__________________
                        |                 |                  |
                        |             mean pooled        input token
                    h2 (t-1)           features           embedding

    If :class:`~updown.models.updown_captioner.UpDownCaptioner` is analogous to an
    :class:`~torch.nn.LSTM`, then this class would be analogous to :class:`~torch.nn.LSTMCell`.

    Parameters
    ----------
    image_feature_size: int
        Size of the bottom-up image features.
    embedding_size: int
        Size of the word embedding input to the captioner.
    hidden_size: int
        Size of the hidden / cell states of attention LSTM and language LSTM of the captioner.
    attention_projection_size: int
        Size of the projected image and textual features before computing bottom-up top-down
        attention weights.
    """

    def __init__(
        self,
        image_feature_size: int,
        embedding_size: int,
        hidden_size: int,
        attention_projection_size: int,
    ):
        super().__init__()

        self.image_feature_size = image_feature_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_projection_size = attention_projection_size

        self._attention_lstm_cell = nn.LSTMCell(
            self.embedding_size + self.image_feature_size + 2 * self.hidden_size, self.hidden_size
        )
        self._butd_attention = BottomUpTopDownAttention(
            self.hidden_size, self.image_feature_size, self.attention_projection_size
        )
        self._language_lstm_cell = nn.LSTMCell(
            self.image_feature_size + 2 * self.hidden_size, self.hidden_size
        )

    def forward(
        self,
        image_features: torch.Tensor,
        token_embedding: torch.Tensor,
        states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Given image features, input token embeddings of current time-step and LSTM states,
        predict output token embeddings for next time-step and update states. This behaves
        very similar to :class:`~torch.nn.LSTMCell`.

        Parameters
        ----------
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes, image_feature_size)``. ``num_boxes`` for
            each instance in a batch might be different. Instances with lesser boxes are padded
            with zeros up to ``num_boxes``.
        token_embedding: torch.Tensor
            A tensor of shape ``(batch_size, embedding_size)`` containing token embeddings for a
            particular time-step.
        states: Dict[str, torch.Tensor], optional (default = None)
            A dict with keys ``{"h1", "c1", "h2", "c2"}`` of LSTM states: (h1, c1) for Attention
            LSTM and (h2, c2) for Language LSTM. If not provided (at first time-step), these are
            initialized as zeros.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tensor of shape ``(batch_size, hidden_state)`` with output token embedding, which
            is the updated state "h2", and updated states (h1, c1), (h2, c2).
        """

        batch_size = image_features.size(0)

        # Average pooling of image features happens only at the first timestep. LRU cache
        # saves compute by not executing the function call in subsequent timesteps.
        # shape: (batch_size, image_feature_size), (batch_size, num_boxes)
        averaged_image_features, image_features_mask = self._average_image_features(image_features)

        # Initialize (h1, c1), (h2, c2) if not passed.
        if states is None:
            state = image_features.new_zeros((batch_size, self.hidden_size))
            states = {
                "h1": state.clone(),
                "c1": state.clone(),
                "h2": state.clone(),
                "c2": state.clone(),
            }

        # shape: (batch_size, embedding_size + image_feature_size + 2 * hidden_size)
        attention_lstm_cell_input = torch.cat(
            [token_embedding, averaged_image_features, states["h1"], states["h2"]], dim=1
        )
        states["h1"], states["c1"] = self._attention_lstm_cell(
            attention_lstm_cell_input, (states["h1"], states["c1"])
        )

        # shape: (batch_size, num_boxes)
        attention_weights = self._butd_attention(
            states["h1"], image_features, image_features_mask=image_features_mask
        )

        # shape: (batch_size, image_feature_size)
        attended_image_features = torch.sum(
            attention_weights.unsqueeze(-1) * image_features, dim=1
        )

        # shape: (batch_size, image_feature_size + 2 * hidden_size)
        language_lstm_cell_input = torch.cat(
            [attended_image_features, states["h1"], states["h2"]], dim=1
        )
        states["h2"], states["c2"] = self._language_lstm_cell(
            language_lstm_cell_input, (states["h2"], states["c2"])
        )

        return states["h2"], states

    @lru_cache(maxsize=10)
    def _average_image_features(
        self, image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Perform mean pooling of bottom-up image features, while taking care of variable
        ``num_boxes`` in case of adaptive features.

        Extended Summary
        ----------------
        For a single training/evaluation instance, the image features remain the same from first
        time-step to maximum decoding steps. To keep a clean API, we use LRU cache -- which would
        maintain a cache of last 10 return values because on call signature, and not actually
        execute itself if it is called with the same image features seen at least once in last
        10 calls. This saves some computation.

        Parameters
        ----------
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes, image_feature_size)``. ``num_boxes`` for
            each instance in a batch might be different. Instances with lesser boxes are padded
            with zeros up to ``num_boxes``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Averaged image features of shape ``(batch_size, image_feature_size)`` and a binary
            mask of shape ``(batch_size, num_boxes)`` which is zero for padded features.
        """

        # shape: (batch_size, num_boxes)
        image_features_mask = torch.sum(torch.abs(image_features), dim=-1) > 0

        # shape: (batch_size, image_feature_size)
        averaged_image_features = masked_mean(
            image_features, image_features_mask.unsqueeze(-1), dim=1
        )

        return averaged_image_features, image_features_mask
