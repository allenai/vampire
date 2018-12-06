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
from common.util import schedule, compute_bow, log_standard_categorical
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
                 kl_weight_annealing: str = None,
                 word_dropout: float = 0.5,
                 embedding_dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(M1, self).__init__()
        self.pad_idx = vocab.get_token_to_index_vocabulary("full")["@@PADDING@@"]
        self.unk_idx = vocab.get_token_to_index_vocabulary("full")["@@UNKNOWN@@"]
        self.sos_idx = vocab.get_token_to_index_vocabulary("full")["@@start@@"]
        self.eos_idx = vocab.get_token_to_index_vocabulary("full")["@@end@@"]
        self.vocab = vocab
        self._embedder = text_field_embedder
        self._masker = get_text_field_mask
        self._dist = distribution
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self._encoder = encoder
        self._decoder = decoder
        self.batch_num = 0
        # we initialize parts of the decoder, classifier, and distribution here so we don't have to repeat
        # dimensions in the config, which can be cumbersome.
        param_input_dim = self._encoder._architecture.get_output_dim()
        self._dist._initialize_params(param_input_dim, latent_dim)
        hidden_factor = 2 if self._decoder._architecture.is_bidirectional() else 1
        embedding_dim = text_field_embedder.token_embedder_tokens.output_dim
        self._decoder._initialize_theta_projection(latent_dim, hidden_dim * hidden_factor, embedding_dim)
        self._decoder._initialize_decoder_out(vocab.get_vocab_size("full"))

        self.word_dropout = word_dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)
        if kl_weight_annealing is not None:
            self.weight_scheduler = lambda x: schedule(x, kl_weight_annealing)
        else:
            self.weight_scheduler = lambda x: 1
        self._reconstruction_loss = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx,
                                                              reduction='sum')
        initializer(self)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                epoch_num: int,
                metadata: torch.IntTensor=None) -> Dict[str, torch.Tensor]:
        """
        Run one step of VAE with RNN decoder
        """
        batch_size = tokens['tokens'].shape[0]

        embedded_text = self._embedder(tokens)
        
        mask = self._masker(tokens)

        # encode tokens
        encoder_output = self._encoder(embedded_text=embedded_text, mask=mask)

        # concatenate generated labels and continuous document vecs as input representation
        input_repr = torch.cat([encoder_output['encoder_output']], 1)

        
        # use parameterized distribution to compute latent code and KL divergence
        _, kld, theta = self._dist.generate_latent_code(input_repr, n_sample=1)

        decoder_input = self.drop_words(tokens, self.word_dropout)
        embedded_text = self._embedder(decoder_input)
        embedded_text = self.embedding_dropout(embedded_text)
        # decode using the latent code.
        decoder_output = self._decoder(embedded_text=embedded_text,
                                       mask=mask,
                                       theta=theta)
        
        reconstruction_loss = self._reconstruction_loss(decoder_output['flattened_decoder_output'],
                                                        tokens['tokens'].view(-1))
        
        # compute marginal likelihood
        nll = reconstruction_loss
        
        kld_weight = self.weight_scheduler(self.batch_num)

        # add in the KLD to compute the ELBO
        kld = kld.to(nll.device)
        
        elbo = nll + kld * kld_weight
        
        output = {
                  'elbo': elbo / batch_size,
                  'nll': nll / batch_size,
                  'kld': kld / batch_size,
                  'kld_weight': kld_weight,
                  'encoded_docs': encoder_output['encoded_docs'],
                  'theta': theta, 
                  'decoder_output': decoder_output}

        self.batch_num += 1
        return output