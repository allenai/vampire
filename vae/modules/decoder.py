from vae.common.util import compute_bow, schedule
from allennlp.modules import FeedForward
from allennlp.nn.util import get_text_field_mask
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
from allennlp.common import Registrable
import torch
from typing import Dict


class Decoder(Registrable, torch.nn.Module):

    default_implementation = 'bow'

    def forward(self, **kwargs):
        raise NotImplementedError

@Decoder.register("seq2seq")
class Seq2Seq(Decoder):

    def __init__(self,
                 architecture: Seq2SeqEncoder,
                 apply_batchnorm: bool = False,
                 batchnorm_annealing: str = None):
        super(Seq2Seq, self).__init__()
        self.architecture = architecture
        self.apply_batchnorm = apply_batchnorm
        self.batchnorm_annealing = batchnorm_annealing
        if self.batchnorm_annealing is not None:
            self.batchnorm_scheduler = lambda x: schedule(x, batchnorm_annealing)
        else:
            self.batchnorm_scheduler = None
        self.batch_num = 0


    def initializedecoder_out(self, vocab_dim):
        self.decoder_out = torch.nn.Linear(self.architecture.get_output_dim(),
                                            vocab_dim)
        if self.apply_batchnorm:
            self.output_bn = torch.nn.BatchNorm1d(vocab_dim, eps=0.001, momentum=0.001, affine=True)
            self.output_bn.weight.data.copy_(torch.ones(vocab_dim))
            self.output_bn.weight.requires_grad = False
    
    def forward(self,
                embedded_text: torch.Tensor,
                mask: torch.Tensor,
                theta: torch.Tensor=None,
                bow: torch.Tensor=None,
                bg: torch.Tensor=None) -> Dict[str, torch.Tensor]:

        if theta is not None:
            lat_to_cat = (theta.unsqueeze(0).expand(embedded_text.shape[1], embedded_text.shape[0], -1)
                                            .permute(1, 0, 2)
                                            .contiguous())
            
            embedded_text = torch.cat([embedded_text, lat_to_cat], dim=2)
        
        if bow is not None:
            bow_to_cat = (bow.unsqueeze(0).expand(embedded_text.shape[1], embedded_text.shape[0], -1)
                                            .permute(1, 0, 2)
                                            .contiguous())
            
            embedded_text = torch.cat([embedded_text, bow_to_cat], dim=2)

        decoder_output = self.architecture(embedded_text, mask)
                                        
        flatteneddecoder_output = decoder_output.view(decoder_output.size(0) * decoder_output.size(1),
                                                       decoder_output.size(2))

        flatteneddecoder_output = self.decoder_out(flatteneddecoder_output)

        if bg is not None:
            flatteneddecoder_output += bg
        
        if self.apply_batchnorm:
            flatteneddecoder_output_bn = self.output_bn(flatteneddecoder_output)
        
            if self.batchnorm_annealing is not None:
                flatteneddecoder_output = (self.batchnorm_scheduler(self.batch_num) * flatteneddecoder_output_bn 
                                            + (1.0 - self.batchnorm_scheduler(self.batch_num)) * flatteneddecoder_output)
            else:
                flatteneddecoder_output = flatteneddecoder_output_bn
        self.batch_num += 1

        return {"decoder_output": decoder_output, "flatteneddecoder_output": flatteneddecoder_output}


@Decoder.register("seq2vec")
class Seq2Vec(Decoder):

    def __init__(self, architecture: Seq2VecEncoder, apply_batchnorm: bool = False, batchnorm_annealing: str = None):
        super(Seq2Vec, self).__init__()
        self.architecture = architecture
        self.apply_batchnorm = apply_batchnorm
        self.batchnorm_annealing = batchnorm_annealing
        if self.batchnorm_annealing is not None:
            self.batchnorm_scheduler = lambda x: schedule(x, batchnorm_annealing)
        else:
            self.batchnorm_scheduler = None
        self.batch_num = 0


    def initializedecoder_out(self, vocab_dim):
        self.decoder_out = torch.nn.Linear(self.architecture.get_output_dim(),
                                            vocab_dim)
        if self.apply_batchnorm:
            self.output_bn = torch.nn.BatchNorm1d(vocab_dim, eps=0.001, momentum=0.001, affine=True)
            self.output_bn.weight.data.copy_(torch.ones(vocab_dim))
            self.output_bn.weight.requires_grad = False
    
    def forward(self,
                embedded_text: torch.Tensor,
                mask: torch.Tensor,
                theta: torch.Tensor=None,
                bow: torch.Tensor=None,
                bg: torch.Tensor=None) -> Dict[str, torch.Tensor]:

        if theta is not None:
            lat_to_cat = (theta.unsqueeze(0).expand(embedded_text.shape[1], embedded_text.shape[0], -1)
                                            .permute(1, 0, 2)
                                            .contiguous())
            
            embedded_text = torch.cat([embedded_text, lat_to_cat], dim=2)
        
        if bow is not None:
            bow_to_cat = (bow.unsqueeze(0).expand(embedded_text.shape[1], embedded_text.shape[0], -1)
                                            .permute(1, 0, 2)
                                            .contiguous())
            
            embedded_text = torch.cat([embedded_text, bow_to_cat], dim=2)

        decoder_output = self.architecture(embedded_text, mask)

        decoder_output = self.decoder_out(decoder_output)

        if bg is not None:
            decoder_output += bg
        
        if self.apply_batchnorm:
            decoder_output_bn = self.output_bn(decoder_output)
        
            if self.batchnorm_annealing is not None:
                decoder_output = (self.batchnorm_scheduler(self.batch_num) * decoder_output_bn 
                                            + (1.0 - self.batchnorm_scheduler(self.batch_num)) * decoder_output)
            else:
                decoder_output = decoder_output_bn
        self.batch_num += 1
        return {"decoder_output": decoder_output, "flatteneddecoder_output": decoder_output}

@Decoder.register("bow")
class Bow(Decoder):

    def __init__(self, apply_batchnorm: bool = False, batchnorm_annealing: str = None):
        super(Bow, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.batchnorm_annealing = batchnorm_annealing
        if self.batchnorm_annealing is not None:
            self.batchnorm_scheduler = lambda x: schedule(x, batchnorm_annealing)
        else:
            self.batchnorm_scheduler = None
        self.batch_num = 0

    def initializedecoder_out(self, latent_dim: int, vocab_dim: int):
        self.decoder_out = torch.nn.Linear(latent_dim,
                                            vocab_dim)
        if self.apply_batchnorm:
            self.output_bn = torch.nn.BatchNorm1d(vocab_dim, eps=0.001, momentum=0.001, affine=True)
            self.output_bn.weight.requires_grad = False

    def forward(self,
                theta: torch.Tensor,
                bow: torch.Tensor=None,
                bg: torch.Tensor=None) -> Dict[str, torch.Tensor]:        

        if bow is not None:
            theta = torch.cat([theta, bow], 1)
        decoder_output = self.decoder_out(theta)
        if bg is not None:
            decoder_output += bg
        if self.apply_batchnorm:
            decoder_output_bn = self.output_bn(decoder_output)
            if self.batchnorm_scheduler is not None:
                decoder_output = (self.batchnorm_scheduler(self.batch_num) * decoder_output_bn 
                                  + (1.0 - self.batchnorm_scheduler(self.batch_num)) * decoder_output)
            else:
                decoder_output = decoder_output_bn
        self.batch_num += 1
        return {"decoder_output": decoder_output, "flatteneddecoder_output": decoder_output}