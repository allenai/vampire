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


@Encoder.register("seq2vec")
class Seq2VecEncoderEnhanced(Encoder):

    def __init__(self, architecture: Seq2VecEncoder) -> None:
        super(Seq2VecEncoderEnhanced, self).__init__()
        self.architecture = architecture

    def initialize_encoder_architecture(self,
                                        input_dim: int  # pylint: disable=unused-argument
                                       ) -> None:
        return

    def forward(self,
                embedded_text: torch.Tensor,
                mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        encoder_output = self.architecture(embedded_text, mask)
        return {"encoded_docs": encoder_output,
                "encoder_output": encoder_output}


@Encoder.register("seq2seq__final_state")
class Seq2SeqEncoderFinalState(Encoder):

    def __init__(self, architecture: Seq2SeqEncoder) -> None:
        super(Seq2SeqEncoderFinalState, self).__init__()
        self.architecture = architecture

    def initialize_encoder_architecture(self,
                                        input_dim: int  # pylint: disable=unused-argument
                                       ) -> None:
        return

    def forward(self,
                embedded_text: torch.Tensor,
                mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        encoded_docs = self.architecture(embedded_text, mask)
        encoder_output = get_final_encoder_states(encoded_docs,
                                                  mask,
                                                  self.architecture.is_bidirectional())
        return {"encoded_docs": encoded_docs,
                "encoder_output": encoder_output}


@Encoder.register("seq2seq__avg")
class Seq2SeqEncoderAvg(Encoder):

    def __init__(self, architecture: Seq2SeqEncoder) -> None:
        super(Seq2SeqEncoderAvg, self).__init__()
        self.architecture = architecture

    def initialize_encoder_architecture(self,
                                        input_dim: int  # pylint: disable=unused-argument
                                       ) -> None:
        return

    def forward(self,
                embedded_text: torch.Tensor,
                mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        encoded_docs = self.architecture(embedded_text, mask)
        broadcast_mask = mask.unsqueeze(-1).float()
        encoded_docs = encoded_docs * broadcast_mask
        encoder_output = masked_mean(encoded_docs,
                                     broadcast_mask,
                                     dim=1,
                                     keepdim=False)
        return {"encoded_docs": encoded_docs,
                "encoder_output": encoder_output}


@Encoder.register("seq2seq__maxpool")
class Seq2SeqEncoderMaxPool(Encoder):

    def __init__(self, architecture: Seq2SeqEncoder) -> None:
        super(Seq2SeqEncoderMaxPool, self).__init__()
        self.architecture = architecture

    def initialize_encoder_architecture(self,
                                        input_dim: int  # pylint: disable=unused-argument
                                       ) -> None:
        return

    def forward(self,
                embedded_text: torch.Tensor,
                mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        encoded_docs = self.architecture(embedded_text, mask)
        broadcast_mask = mask.unsqueeze(-1).float()
        one_minus_mask = (1.0 - broadcast_mask).byte()
        replaced = encoded_docs.masked_fill(one_minus_mask, -1e-7)
        encoder_output, _ = replaced.max(dim=1, keepdim=False)
        return {"encoded_docs": encoded_docs,
                "encoder_output": encoder_output}
