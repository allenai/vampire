import torch
import numpy as np
import os
from allennlp.nn.util import get_text_field_mask
from vae.util import compute_bow
from typing import Dict, Optional, List, Any
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder, FeedForward, Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy, Average
from allennlp.nn.util import get_text_field_mask, get_device_of
from allennlp.models.archival import load_archive, Archive
from vae.vae import VAE
from vae.distribution import Distribution
from allennlp.nn import InitializerApplicator
from overrides import overrides


@VAE.register("bag_of_words_vae")
class BowVAE(VAE):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder: FeedForward,
                 distribution: Distribution,
                 mode: str = "supervised", 
                 hidden_dim: int = 128,
                 latent_dim: int = 50,
                 kl_weight: float = 1.0,
                 dropout: float = 0.2,
                 pretrained_file: str = None, 
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(BowVAE, self).__init__()
        self.metrics = {
            'kld': Average(),
            'reconstruction': Average(),
            'nll': Average(),
            'accuracy': CategoricalAccuracy(),
            'elbo': Average()
        }
        self.vocab = vocab
        self._mode = mode
        self._num_labels = vocab.get_vocab_size("labels")
        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._decoder = encoder
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.dropout = dropout
        self.pretrained_file = pretrained_file
        self._dist = distribution    
        self._projection_feedforward = torch.nn.Linear(self.vocab.get_vocab_size("stopless"), hidden_dim)
        self._encoder_dropout = torch.nn.Dropout(dropout)
        self._latent_dropout = torch.nn.Dropout(dropout)
        self._x_recon = torch.nn.Linear(latent_dim, self.vocab.get_vocab_size("stopless"))
        self._y_recon = torch.nn.Linear(self._encoder.get_output_dim(), self._num_labels)
        self._discriminator_loss = torch.nn.CrossEntropyLoss()
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
    def _encode(self, full_tokens, stopless_tokens, label):
        batch_size = full_tokens['tokens'].size(0)
        onehot_repr = compute_bow(stopless_tokens, self.vocab.get_index_to_token_vocabulary("stopless"))
        onehot_proj = self._projection_feedforward(onehot_repr)
        onehot_proj = self._encoder_dropout(onehot_proj)
        if self._mode == 'supervised':
            label_onehot = onehot_repr.new_zeros(batch_size, self._num_labels).float()
            label_onehot = label_onehot.scatter_(1, label.reshape(-1, 1), 1)
        else:
            label_onehot = None
        
        embedded_text = self._text_field_embedder(full_tokens)
        mask = get_text_field_mask(full_tokens).float()
        
        encoded_docs = self._encoder(embedded_text, mask)
        cont_repr = torch.max(encoded_docs, 1)[0]

        input_repr = {'onehot_repr': onehot_proj,
                      'cont_repr': cont_repr,
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
        x_recon = self._x_recon(theta)
        # x_recon = self._batch_norm_xrecon(x_recon)
        x_recon = torch.nn.functional.softmax(x_recon, dim=1)
        return x_recon

    @overrides
    def _reconstruction_loss(self, x_onehot: torch.FloatTensor, x_recon: torch.FloatTensor):
        return -torch.sum(x_onehot * (x_recon + 1e-10).log(), dim=-1)

    @overrides
    def _discriminator(self, cont_repr: torch.Tensor):
        """
        Given the instances, labelled or unlabelled, selects the correct input
        to use and classifies it.
        """
        logits = self._y_recon(cont_repr)
        return logits

    @overrides
    def forward(self, full_tokens, stopless_tokens, label):
        cuda_device = get_device_of(full_tokens['tokens'])
        batch_size = full_tokens['tokens'].size(0)

        x_onehot, input_repr = self._encode(full_tokens=full_tokens,
                                            stopless_tokens=stopless_tokens,
                                            label=label)
        
        logits = self._discriminator(input_repr['cont_repr'])
        
        if self._mode == 'unsupervised':
            artificial_label = logits.max(1)[1]
            label_onehot = x_onehot.new_zeros(batch_size, self._num_labels).float()
            label_onehot = label_onehot.scatter_(1, artificial_label.reshape(-1, 1), 1)
            input_repr['label_repr'] = label_onehot
        input_repr = torch.cat(list(input_repr.values()), 1)

        params, kld, theta = self._dist.generate_latent_repr(input_repr, n_sample=1)
        
        x_recon = self._decode(theta=theta)

        reconstruction_loss = self._reconstruction_loss(x_onehot,
                                                        x_recon)

        nll = reconstruction_loss

        if self._mode == 'supervised': 
            discriminator_loss = self._discriminator_loss(logits, label)
            nll += discriminator_loss

        elbo = nll + kld.to(nll.device)

        # set metrics
        self.metrics['accuracy'](logits, label)
        self.metrics["reconstruction"](reconstruction_loss.mean())
        self.metrics["elbo"](elbo.mean())
        self.metrics["kld"](kld.mean())
        self.metrics["nll"](nll.mean())
        
        # set output_dict
        output_dict = {}
        output_dict['x_recon'] = x_recon
        output_dict['theta'] = theta
        output_dict['loss'] = elbo.mean()
        output_dict['logits'] = logits
        output_dict['kld'] = kld.mean().data.cpu().numpy()
        output_dict['nll'] = nll.mean().data.cpu().numpy()
        output_dict['reconstruction'] = reconstruction_loss.mean().data.cpu().numpy()

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}
