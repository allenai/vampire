import torch
import numpy as np
import os
from allennlp.nn.util import get_text_field_mask
from typing import Dict, Optional, List, Any
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder, FeedForward, Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask, get_device_of
from allennlp.models.archival import load_archive, Archive
from modules.vae import VAE
from common.util import compute_bow
from modules.distribution import Distribution
from allennlp.nn import InitializerApplicator
from overrides import overrides
from modules.distribution import Normal, VMF

@VAE.register("bag_of_words_vae")
class BowVAE(VAE):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder: FeedForward,
                 distribution: str = "normal",
                 mode: str = "supervised", 
                 hidden_dim: int = 128,
                 latent_dim: int = 50,
                 kl_weight: float = 1.0,
                 dropout: float = 0.2,
                 pretrained_file: str = None, 
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(BowVAE, self).__init__()
        self.name = 'bow_vae'
        self.vocab = vocab
        self._mode = mode
        self._num_labels = vocab.get_vocab_size("labels")
        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._decoder = decoder
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.dropout = dropout
        self.pretrained_file = pretrained_file
        if distribution == 'normal':
            self._dist = Normal(hidden_dim=self._encoder.get_output_dim() + self._num_labels,
                                latent_dim=self.latent_dim,
                                func_mean=FeedForward(input_dim=self._encoder.get_output_dim() + self._num_labels,
                                                      num_layers=1,
                                                      hidden_dims=self.latent_dim,
                                                      activations=torch.nn.Softplus()),
                                func_logvar=FeedForward(input_dim=self._encoder.get_output_dim() + self._num_labels,
                                                        num_layers=1,
                                                        hidden_dims=self.latent_dim,
                                                        activations=torch.nn.Softplus()))
        elif distribution == "vmf":
            self._dist = VMF(hidden_dim=self._encoder.get_output_dim() + self._num_labels,
                             latent_dim=self.latent_dim,
                             func_mean=FeedForward(input_dim=self._encoder.get_output_dim() + self._num_labels,
                                                   num_layers=1,
                                                   hidden_dims=self.latent_dim,
                                                   activations=torch.nn.Softplus()))
        self.stopword_indicator = torch.zeros(self.vocab.get_vocab_size("full"))
        indices = [self.vocab.get_token_to_index_vocabulary('full')[x]
                   for x in self.vocab.get_token_to_index_vocabulary('full').keys()
                   if x in ('@@PADDING@@', '@@UNKNOWN@@')]
        self.stopword_indicator[indices] = 1
        self._latent_dropout = torch.nn.Dropout(dropout)
        if pretrained_file is not None:
            if os.path.isfile(pretrained_file):
                archive = load_archive(pretrained_file)
                self._initialize_weights_from_archive(archive)
            else:
                logger.error("model file for initializing weights is passed, but does not exist.")
        else:
            initializer(self)

    @overrides
    def _initialize_weights_from_archive(self, archive: Archive) -> None:
        # logger.info("Initializing weights from pre-trained SCHOLAR model.")
        model_parameters = dict(self.named_parameters())
        archived_parameters = dict(archive.model.named_parameters())
        for name, weights in archived_parameters.items():
            if name in model_parameters:
                if "mu" in name or "var" or "encoder" or "x_recon" in name:
                    new_weights = weights.data
                    model_parameters[name].data.copy_(new_weights)
    
    @overrides
    def _encode(self, tokens, label):
        batch_size = tokens['tokens'].size(0)
        onehot_repr = compute_bow(tokens, self.vocab.get_index_to_token_vocabulary("full"), self.stopword_indicator)
        if self._mode == 'supervised':
            label_onehot = onehot_repr.new_zeros(batch_size, self._num_labels).float()
            label_onehot = label_onehot.scatter_(1, label.reshape(-1, 1), 1)
        else:
            label_onehot = None
        
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()
        
        encoded_docs = self._encoder(embedded_text, mask)
        cont_repr = torch.max(encoded_docs, 1)[0]

        input_repr = {'cont_repr': cont_repr,
                      'label_repr': label_onehot}
    
        return onehot_repr, input_repr

    @overrides
    def _reparameterize(self, posteriors):
        """
        reparametrization trick
        """
        mu = posteriors['mean']
        logvar = posteriors['logvar']
        eps = torch.randn(mu.shape).to(mu.device)
        latent = mu + logvar.exp().sqrt() * eps
        latent = self._latent_dropout(latent)
        theta = torch.nn.functional.softmax(latent, dim=1)
        return theta

    @overrides
    def _decode(self, theta: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Decode theta into reconstruction of input
        """
        # reconstruct input
        x_recon = self._decoder(theta)
        # x_recon = self._batch_norm_xrecon(x_recon)
        x_recon = torch.nn.functional.softmax(x_recon, dim=1)
        return x_recon

    @overrides
    def _reconstruction_loss(self, x_onehot: torch.FloatTensor, x_recon: torch.FloatTensor):
        return -torch.sum(x_onehot * (x_recon + 1e-10).log(), dim=-1)

    

    @overrides
    def forward(self, tokens, label):
        cuda_device = get_device_of(tokens['tokens'])
        batch_size = tokens['tokens'].size(0)
        
        x_onehot, input_repr = self._encode(tokens=tokens,
                                            label=label)

        if self._mode == 'unsupervised':
            artificial_label = logits.max(1)[1]
            label_onehot = x_onehot.new_zeros(batch_size, self._num_labels).float()
            label_onehot = label_onehot.scatter_(1, artificial_label.reshape(-1, 1), 1)
            input_repr['label_repr'] = label_onehot

        input_repr_ = torch.cat(list(input_repr.values()), 1)

        params, kld, theta = self._dist.generate_latent_repr(input_repr_, n_sample=1)
        
        x_recon = self._decode(theta=theta)

        reconstruction_loss = self._reconstruction_loss(x_onehot,
                                                        x_recon)

        nll = reconstruction_loss

        elbo = nll + kld.to(nll.device)
        
        # set output_dict
        output_dict = {}
        output_dict['cont_repr'] = input_repr['cont_repr']
        output_dict['x_recon'] = x_recon
        output_dict['theta'] = theta
        output_dict['elbo'] = elbo.mean()
        output_dict['kld'] = kld.mean().data.cpu().numpy()
        output_dict['nll'] = nll.mean().data.cpu().numpy()
        output_dict['reconstruction'] = reconstruction_loss.mean().data.cpu().numpy()

        return output_dict