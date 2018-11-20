import torch
import numpy as np
import os
from allennlp.nn.util import get_text_field_mask
from typing import Dict, Optional, List, Any
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder, FeedForward, Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask, get_device_of, sequence_cross_entropy_with_logits
from allennlp.models.archival import load_archive, Archive
from modules.vae import VAE
from modules.distribution import Distribution
from common.util import compute_bow
from allennlp.nn import InitializerApplicator
from overrides import overrides


@VAE.register("rnn_vae")
class RNNVAE(VAE):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder: Seq2SeqEncoder,
                 distribution: Distribution,
                 mode: str = "supervised", 
                 hidden_dim: int = 128,
                 latent_dim: int = 50,
                 kl_weight: float = 1.0,
                 dropout: float = 0.2,
                 pretrained_file: str = None, 
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(RNNVAE, self).__init__()
        self.name = 'rnn_vae'
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
        self._dist = distribution
        # self.stopword_indicator = torch.zeros(self.vocab.get_vocab_size("full"))
        # indices = [self.vocab.get_token_to_index_vocabulary('full')[x]
        #            for x in self.vocab.get_token_to_index_vocabulary('full').keys()
        #            if self.vocab.get_token_to_index_vocabulary('stopless').get(x) is None]
        # self.stopword_indicator[indices] = 1
        self._projection_feedforward = torch.nn.Linear(vocab.get_vocab_size("full"), hidden_dim)
        self._encoder_dropout = torch.nn.Dropout(dropout)
        self._latent_dropout = torch.nn.Dropout(dropout)
        self._decoder_dropout = torch.nn.Dropout(dropout)
        self._theta_projection_h = torch.nn.Linear(self.latent_dim, self.hidden_dim * 2 if self._encoder.is_bidirectional else self.hidden_dim)
        self._theta_projection_c = torch.nn.Linear(self.latent_dim, self.hidden_dim * 2 if self._encoder.is_bidirectional else self.hidden_dim)
        self._x_recon = torch.nn.Linear(self._decoder.get_output_dim(), self.vocab.get_vocab_size("full"))
        self._reconstruction_criterion = torch.nn.CrossEntropyLoss()
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
        onehot_repr = compute_bow(tokens, self.vocab.get_index_to_token_vocabulary("full"))
        onehot_proj = self._projection_feedforward(onehot_repr)
        onehot_proj = self._encoder_dropout(onehot_proj)
        if self._mode == 'supervised':
            label_onehot = onehot_repr.new_zeros(batch_size, self._num_labels).float()
            label_onehot = label_onehot.scatter_(1, label.reshape(-1, 1), 1)
        else:
            label_onehot = None
        
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()
        
        encoded_docs = self._encoder(embedded_text, mask)
        cont_repr = torch.max(encoded_docs, 1)[0]

        input_repr = {'encoded_docs': encoded_docs,
                      'mask': mask,
                      'onehot_repr': onehot_proj,
                      'cont_repr': cont_repr,
                      'label_repr': label_onehot}
    
        return embedded_text, mask, input_repr

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
    def _decode(self, encoded_docs: torch.Tensor, mask: torch.Tensor, theta: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Decode theta into reconstruction of input
        """
        # reconstruct input
        n_layers = 2 if self._encoder.is_bidirectional else 1
        theta_projection_h = self._theta_projection_h(theta).view(encoded_docs.shape[0], n_layers, -1).permute(1, 0, 2).contiguous()
        theta_projection_c = self._theta_projection_c(theta).view(encoded_docs.shape[0], n_layers, -1).permute(1, 0, 2).contiguous()
        x_recon = self._decoder(encoded_docs, mask, (theta_projection_h, theta_projection_c))
        x_recon = self._decoder_dropout(x_recon)
        x_recon_flattened = x_recon.view(x_recon.size(0) * x_recon.size(1), x_recon.size(2))
        logits = self._x_recon(x_recon_flattened)
        downstream_projection = torch.max(x_recon, 1)[0]
        # x_recon = self._batch_norm_xrecon(x_recon)
        # x_recon = torch.nn.functional.softmax(x_recon, dim=1)
        return logits, downstream_projection

    @overrides
    def _reconstruction_loss(self, tokens: torch.LongTensor, logits: torch.FloatTensor):
        return self._reconstruction_criterion(logits, tokens['tokens'].view(-1))
    
    @overrides
    def forward(self, tokens, label):
        cuda_device = get_device_of(tokens['tokens'])
        batch_size = tokens['tokens'].size(0)

        embedded_text, mask, input_repr = self._encode(tokens=tokens, label=label)

        if self._mode == 'unsupervised':
            artificial_label = logits.max(1)[1]
            label_onehot = x_onehot.new_zeros(batch_size, self._num_labels).float()
            label_onehot = label_onehot.scatter_(1, artificial_label.reshape(-1, 1), 1)
            input_repr['label_repr'] = label_onehot

        input_repr_ = torch.cat([input_repr['cont_repr'], input_repr['label_repr'], input_repr['onehot_repr']], 1)

        params, kld, theta = self._dist.generate_latent_repr(input_repr_, n_sample=1)
        
        logits, downstream_projection = self._decode(encoded_docs=input_repr['encoded_docs'], mask=input_repr['mask'], theta=theta)
        reconstruction_loss = self._reconstruction_loss(tokens, logits)

        nll = reconstruction_loss

        elbo = nll + kld.to(nll.device)
        
        # set output_dict
        output_dict = {}
        output_dict['downstream_projection'] = downstream_projection
        output_dict['cont_repr'] = input_repr['cont_repr']
        output_dict['theta'] = theta
        output_dict['elbo'] = elbo.mean()
        output_dict['kld'] = kld.mean().data.cpu().numpy()
        output_dict['nll'] = nll.mean().data.cpu().numpy()
        output_dict['reconstruction'] = reconstruction_loss.mean().data.cpu().numpy()

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}
