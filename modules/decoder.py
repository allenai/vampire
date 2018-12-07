from common.util import compute_bow
from allennlp.modules import FeedForward
from allennlp.nn.util import get_text_field_mask
from allennlp.modules import Seq2SeqEncoder
from allennlp.common import Registrable
import torch
from typing import Dict


class Decoder(Registrable, torch.nn.Module):

    default_implementation='bow'

    def forward(self, **kwargs):
        raise notImplementedError

@Decoder.register("seq2seq")
class Seq2Seq(Decoder):

    def __init__(self, architecture: Seq2SeqEncoder):
        super(Seq2Seq, self).__init__()
        self._architecture = architecture
        
    
    def _initialize_theta_projection(self, latent_dim, hidden_dim, embedding_dim):
        self._theta_projection_e = torch.nn.Linear(latent_dim, embedding_dim)
        self._theta_projection_h = torch.nn.Linear(latent_dim, hidden_dim)
        self._theta_projection_c = torch.nn.Linear(latent_dim, hidden_dim)

    def _initialize_decoder_out(self, vocab_dim):
        self._decoder_out = torch.nn.Linear(self._architecture.get_output_dim(),
                                            vocab_dim)
    def forward(self,
                embedded_text: torch.Tensor,
                mask: torch.Tensor,
                theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # reconstruct input
        n_layers = 2 if self._architecture.is_bidirectional() else 1
        theta_projection_h = self._theta_projection_h(theta)
        theta_projection_h = (theta_projection_h.view(embedded_text.shape[0], n_layers, -1)
                                                .permute(1, 0, 2)
                                                .contiguous())

        lat_to_cat = (theta.unsqueeze(0).expand(embedded_text.shape[1], embedded_text.shape[0], -1)
                                        .permute(1, 0, 2)
                                        .contiguous())
        embedded_text = torch.cat([embedded_text, lat_to_cat], dim=2)
        
        if n_layers == 2:
            theta_projection_c = self._theta_projection_c(theta)
            theta_projection_c = (theta_projection_c.view(embedded_text.shape[0], n_layers, -1)
                                                    .permute(1, 0, 2)
                                                    .contiguous())
            decoder_output = self._architecture(embedded_text,
                                                mask,
                                                (theta_projection_h, theta_projection_c))
        else:
            decoder_output = self._architecture(embedded_text,
                                                mask,
                                                theta_projection_h)
        
        flattened_decoder_output = decoder_output.view(decoder_output.size(0) * decoder_output.size(1),
                                                       decoder_output.size(2))
        flattened_decoder_output = self._decoder_out(flattened_decoder_output)
        return {"decoder_output": decoder_output,
                "flattened_decoder_output": flattened_decoder_output}


@Decoder.register("bow")
class Bow(Decoder):

    def __init__(self, hidden_dim: int):
        super(Bow, self).__init__()
        self._hidden_dim = hidden_dim        

    def _initialize_decoder_architecture(self, input_dim: int):
        softplus = torch.nn.Softplus()
        self._architecture = FeedForward(input_dim=input_dim,
                                         num_layers=1,
                                         hidden_dims=self._hidden_dim,
                                         activations=softplus,
                                         dropout=0.2)

    def _initialize_decoder_out(self, vocab_dim: int):
        self._decoder_out = torch.nn.Linear(self._architecture.get_output_dim(),
                                            vocab_dim)
    def forward(self,
                theta: torch.Tensor,
                embedded_text: torch.Tensor=None,
                mask: torch.Tensor=None) -> Dict[str, torch.Tensor]:        
        decoder_output = self._architecture(theta)
        decoder_output = self._decoder_out(decoder_output)
        return {"decoder_output": decoder_output, "flattened_decoder_output": decoder_output}