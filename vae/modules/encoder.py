# pylint: disable=arguments-differ, no-self-use

from typing import Dict
import torch
from allennlp.modules import FeedForward
from allennlp.nn.util import get_final_encoder_states, masked_mean
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
from allennlp.common import Registrable


class Encoder(Registrable, torch.nn.Module):

    default_implementation = 'bow'

    def forward(self,
                embedded_text: torch.Tensor,
                mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


@Encoder.register("bow")
class BowEncoder(Encoder):

    def __init__(self, hidden_dim: int) -> None:
        super(BowEncoder, self).__init__()
        self._hidden_dim = hidden_dim

    # pylint: disable=attribute-defined-outside-init
    def initialize_encoder_architecture(self, input_dim: int) -> None:
        self.architecture = FeedForward(input_dim=input_dim,
                                        num_layers=1,
                                        hidden_dims=self._hidden_dim,
                                        activations=lambda x: x,
                                        dropout=0.2)

    def forward(self,
                embedded_text: torch.Tensor,
                mask: torch.Tensor = None  # pylint: disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        onehot_proj = self.architecture(embedded_text)
        return {"encoded_docs": embedded_text,
                "encoder_output": onehot_proj}
