# pylint: disable=arguments-differ

from typing import Dict
import torch
from allennlp.common import Registrable
from vae.common.util import schedule


class Decoder(Registrable, torch.nn.Module):

    default_implementation = 'bow'

    def forward(self,
                theta: torch.Tensor,
                bow: torch.Tensor = None,
                bg: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


@Decoder.register("bow")
class Bow(Decoder):

    def __init__(self, apply_batchnorm: bool = False, batchnorm_annealing: str = None) -> None:
        super(Bow, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.batchnorm_annealing = batchnorm_annealing
        if self.batchnorm_annealing is not None:
            self.batchnorm_scheduler = lambda x: schedule(x, batchnorm_annealing)
        else:
            self.batchnorm_scheduler = None
        self.batch_num = 0

    # pylint: disable=attribute-defined-outside-init
    def initialize_decoder_out(self, latent_dim: int, vocab_dim: int):
        self.decoder_out = torch.nn.Linear(latent_dim, vocab_dim)
        if self.apply_batchnorm:
            self.output_bn = torch.nn.BatchNorm1d(vocab_dim, eps=0.001, momentum=0.001, affine=True)
            self.output_bn.weight.requires_grad = False

    def forward(self,
                theta: torch.Tensor,
                bow: torch.Tensor = None,
                bg: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        if bow is not None:
            theta = torch.cat([theta, bow], 1)
        decoder_output = self.decoder_out(theta)
        if bg is not None:
            decoder_output += bg
        if self.apply_batchnorm:
            decoder_output_bn = self.output_bn(decoder_output)
            if self.batchnorm_scheduler is not None:
                decoder_output = (self.batchnorm_scheduler(self.batch_num) * decoder_output_bn +
                                  (1.0 - self.batchnorm_scheduler(self.batch_num)) * decoder_output)
            else:
                decoder_output = decoder_output_bn
        self.batch_num += 1
        return {"decoder_output": decoder_output, "flattened_decoder_output": decoder_output}
