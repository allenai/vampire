import torch
import numpy as np
import os
from allennlp.nn.util import get_text_field_mask
from typing import Dict, Optional, List, Any, Tuple
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder, FeedForward
from allennlp.nn.util import get_text_field_mask, get_device_of
from allennlp.models.archival import load_archive, Archive
from allennlp.nn import InitializerApplicator
from overrides import overrides
from modules.vae import VAE
from modules.distribution import Distribution
from modules.encoder import Encoder
from modules.decoder import Decoder
from common.util import schedule, compute_bow
from typing import Dict

@VAE.register("M1")
class M1(VAE):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 latent_dim: int,
                 hidden_dim: int,
                 encoder: Encoder,
                 decoder: Decoder,
                 distribution: Distribution,
                 kl_weight_annealing: str = 'sigmoid',
                 word_dropout: float = 0.5,
                 pretrained_file: str = None,
                 freeze_pretrained_weights: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(M1, self).__init__()
        self.pad_idx = vocab.get_token_to_index_vocabulary("full")["@@PADDING@@"]
        self.unk_idx = vocab.get_token_to_index_vocabulary("full")["@@UNKNOWN@@"]
        self.sos_idx = vocab.get_token_to_index_vocabulary("full")["@@start@@"]
        self._embedder = text_field_embedder
        self._masker = get_text_field_mask
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self._dist = distribution
        self._encoder = encoder
        self._decoder = decoder
        self.batch_num = 0
        # we initialize parts of the decoder and distribution here so we don't have to repeat
        # dimensions in the config, which can be cumbersome.
        self._dist._initialize_params(self._encoder._encoder.get_output_dim(), latent_dim)
        self._decoder._initialize_theta_projection(latent_dim, hidden_dim * 2)
        self._decoder._initialize_decoder_out(vocab.get_vocab_size("full"))
        self.word_dropout = word_dropout
        self.weight_scheduler = lambda x: schedule(x, kl_weight_annealing)
        self._reconstruction_loss = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx,
                                                              reduction='sum')
        if pretrained_file is not None:
            if os.path.isfile(pretrained_file):
                archive = load_archive(pretrained_file)
                self._initialize_weights_from_archive(archive, freeze_pretrained_weights)
        else:
            initializer(self)
    
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                epoch_num: int,
                label: torch.IntTensor=None,  
                metadata: torch.IntTensor=None) -> Dict[str, torch.Tensor]:
        """
        Run one step of VAE with RNN decoder
        """

        batch_size = tokens['tokens'].shape[0]


        embedded_text = self._embedder(tokens)
        
        mask = self._masker(tokens)

        # encode tokens
        encoder_output = self._encoder(embedded_text=embedded_text, mask=mask)

        # use parameterized distribution to compute latent code and KL divergence
        _, kld, theta = self._dist.generate_latent_code(encoder_output['encoder_output'],
                                                        n_sample=1)

        decoder_input = self.drop_words(tokens)
        embedded_text = self._embedder(decoder_input)


        # decode using the latent code.
        decoder_output = self._decoder(embedded_text=embedded_text,
                                       mask=mask,
                                       theta=theta)

        reconstruction_loss = self._reconstruction_loss(decoder_output['flattened_decoder_output'],
                                                        tokens['tokens'].view(-1))

        # compute marginal likelihood
        nll = reconstruction_loss

        # add in the KLD to compute the ELBO
        kld = kld.to(nll.device) * self.weight_scheduler(self.batch_num)
        elbo = nll + kld

        output = {'elbo': elbo  / batch_size,
                  'nll': nll / batch_size,
                  'kld': kld / batch_size,
                  'encoded_docs': encoder_output['encoded_docs'],
                  'theta': theta,
                  'decoder_output': decoder_output} 
        self.batch_num += 1
        return output