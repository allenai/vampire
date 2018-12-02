from common.util import compute_bow
from allennlp.modules import FeedForward
from allennlp.nn.util import get_text_field_mask
from allennlp.modules import Seq2SeqEncoder
from allennlp.common import Registrable
import torch
from typing import Dict


class Encoder(Registrable, torch.nn.Module):

    default_implementation='bow'

    def forward(self, **kwargs):
        raise notImplementedError

@Encoder.register("bow")
class BowEncoder(Encoder):

    def __init__(self, hidden_dim: int):
        super(BowEncoder, self).__init__()
        self._hidden_dim = hidden_dim

    def forward(self, embedded_text, mask=None) -> Dict[str, torch.Tensor]:
        
        projection = FeedForward(input_dim=onehot_repr.shape[1],
                                 num_layers=1,
                                 hidden_dims=self._hidden_dim,
                                 activations=torch.nn.Linear)
        onehot_proj = projection(onehot_repr)
        return {"encoded_docs": onehot_repr,
                "encoder_output": onehot_proj}

@Encoder.register("seq2seq")
class Seq2SeqEncoder(Encoder):
    
    def __init__(self, architecture: Seq2SeqEncoder, aggregate: str = "maxpool"):
        super(Seq2SeqEncoder, self).__init__()
        self._encoder = architecture
        self._aggregate = aggregate

    def forward(self, embedded_text, mask) -> Dict[str, torch.Tensor]:
        encoded_docs = self._encoder(embedded_text, mask)
        if self._aggregate == 'maxpool':
            encoder_output = torch.max(encoded_docs, 1)[0]
        return {"encoded_docs": encoded_docs,
                "encoder_output": encoder_output}