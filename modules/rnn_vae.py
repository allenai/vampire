import torch
import numpy as np
import os
from allennlp.nn.util import get_text_field_mask
from typing import Dict, Optional, List, Any
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder, FeedForward, Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask, get_device_of
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.models.archival import load_archive, Archive
from allennlp.nn import InitializerApplicator
from overrides import overrides
from modules.distribution import Normal, VMF
from modules.vae import VAE
from common.util import compute_bow


@VAE.register("rnn_vae")
class RNN_VAE(VAE):
    """
    Implementation of a VAE with an RNN-based decoder.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder: Seq2SeqEncoder,
                 classifier: FeedForward,
                 mode: str = "supervised",
                 distribution: str = "normal",
                 hidden_dim: int = 128,
                 latent_dim: int = 50,
                 kl_weight: float = 1.0,
                 dropout: float = 0.2,
                 pretrained_file: str = None,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(RNN_VAE, self).__init__()
        self.name = 'rnn_vae'
        self.vocab = vocab
        self._mode = mode
        self._num_labels = vocab.get_vocab_size("labels")
        self._text_field_embedder = text_field_embedder
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self._encoder = encoder
        self._decoder = decoder
        self._classifier = classifier
        self._classifier_logits = torch.nn.Linear(self._classifier.get_output_dim(),
                                                  self._num_labels)
        self._classifier_loss = torch.nn.CrossEntropyLoss()
        self.dropout = dropout
        self.pretrained_file = pretrained_file

        # Initialize distribution parameter networks and priors
        param_input_dim = self._encoder.get_output_dim() + self._num_labels
        softplus = torch.nn.Softplus()

        if distribution == 'normal':
            self._dist = Normal(hidden_dim=param_input_dim,
                                latent_dim=self.latent_dim,
                                func_mean=FeedForward(input_dim=param_input_dim,
                                                      num_layers=1,
                                                      hidden_dims=self.latent_dim,
                                                      activations=softplus),
                                func_logvar=FeedForward(input_dim=param_input_dim,
                                                        num_layers=1,
                                                        hidden_dims=self.latent_dim,
                                                        activations=softplus))
        elif distribution == "vmf":
            # VAE with VMF prior. kappa is fixed at 100, based on empirical findings.
            self._dist = VMF(kappa=100,
                             hidden_dim=param_input_dim,
                             latent_dim=self.latent_dim,
                             func_mean=FeedForward(input_dim=param_input_dim,
                                                   num_layers=1,
                                                   hidden_dims=self.latent_dim,
                                                   activations=softplus))
        else:
            logger.error("{} is not a distribution that is not supported.".format(distribution))

        self._encoder_dropout = torch.nn.Dropout(dropout)
        self._decoder_dropout = torch.nn.Dropout(dropout)

        # create theta projections into initial states for RNN decoder
        h_dim = self.hidden_dim * 2 if self._encoder.is_bidirectional else self.hidden_dim
        self._theta_projection_h = torch.nn.Linear(self.latent_dim, h_dim)
        self._theta_projection_c = torch.nn.Linear(self.latent_dim, h_dim)

        # projection to reconstruct input
        self._x_recon = torch.nn.Linear(self._decoder.get_output_dim(),
                                        self.vocab.get_vocab_size("full"))
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
        """
        Initialize weights (theta?) from a model archive.

        Params
        ______
        archive : `Archive`
            pretrained model archive
        """
        model_parameters = dict(self.named_parameters())
        archived_parameters = dict(archive.model.named_parameters())
        new_weights = archived_parameters["theta"].data
        model_parameters["theta"].data.copy_(new_weights)

    @overrides
    def _encode(self, tokens: Dict[str, torch.Tensor]):
        """
        Encode the tokens into embeddings

        Params
        ______

        tokens: ``Dict[str, torch.Tensor]`` - tokens to embed

        Returns
        _______

        input_repr: ``Dict[str, torch.Tensor]``
            Dictionary containing:
                - encoded document as vector sequences
                - mask on tokens
                - document vectors
        """
        batch_size = tokens['tokens'].size(0)
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()
        encoded_docs = self._encoder(embedded_text, mask)
        cont_repr = torch.max(encoded_docs, 1)[0]
        cont_repr = self._encoder_dropout(cont_repr)
        input_repr = {
                      'encoded_docs': encoded_docs,
                      'mask': mask,
                      'cont_repr': cont_repr
                    }
        return input_repr

    @overrides
    def _decode(self,
                encoded_docs: torch.Tensor,
                mask: torch.Tensor,
                theta: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Decode theta into reconstruction of input using an RNN.

        Params
        ______

        encoded_docs: ``torch.Tensor``
            encoded documents from input representation
        mask : ``torch.Tensor``
            token mask from input representation
        theta: ``torch.Tensor``
            latent code

        Returns
        _______

        decoded_output: ``torch.Tensor``
            output of decoder

        flattened_decoded_output: ``torch.Tensor``
            output of decoder, flattened. we do this so we can use straight cross entropy
            in the reconstruction loss.
        """
        # reconstruct input
        n_layers = 2 if self._encoder.is_bidirectional else 1
        theta_projection_h = self._theta_projection_h(theta)
        theta_projection_h = (theta_projection_h.view(encoded_docs.shape[0], n_layers, -1)
                                                .permute(1, 0, 2)
                                                .contiguous())
        theta_projection_c = self._theta_projection_c(theta)
        theta_projection_c = (theta_projection_c.view(encoded_docs.shape[0], n_layers, -1)
                                                .permute(1, 0, 2)
                                                .contiguous())
        decoded_output = self._decoder(encoded_docs, mask, (theta_projection_h, theta_projection_c))
        decoded_output = self._decoder_dropout(decoded_output)
        flattened_decoded_output = decoded_output.view(decoded_output.size(0) * decoded_output.size(1),
                                                       decoded_output.size(2))
        flattened_decoded_output = self._x_recon(flattened_decoded_output)
        # x_recon = self._batch_norm_xrecon(x_recon)
        # x_recon = torch.nn.functional.softmax(x_recon, dim=1)
        return decoded_output, flattened_decoded_output

    @overrides
    def _reconstruction_loss(self, tokens: torch.LongTensor, decoder_output: torch.FloatTensor):
        """
        Calculate reconstruction loss between input tokens and output of decoder

        Params
        ______

        tokens: ``Dict[str, torch.Tensor]``
            input tokens

        decoder_output: ``torch.FloatTensor``
            output of decoder, a projection to the vocabulary
        """
        return self._reconstruction_criterion(decoder_output, tokens['tokens'].view(-1))

    def _discriminate(tokens, label):
        """
        Generate labels from the input, and use supervision to compute a loss.

        Params
        ______

        tokens: ``Dict[str, torch.Tensor]``
            input tokens

        label: ``torch.IntTensor``
            gold labels

        Returns
        _______

        loss: ``torch.FloatTensor``
            Cross entropy loss on gold label

        label_onehot: ``torch.IntTensor``
            onehot representation of generated labels
        """
        clf_out = self._classifier(input_repr['cont_repr'])
        logits = self._classifier_logits(clf_out)
        artificial_label = logits.max(1)[1]
        label_onehot = clf_out.new_zeros(tokens['tokens'].size(0), self._num_labels).float()
        label_onehot = label_onehot.scatter_(1, artificial_label.reshape(-1, 1), 1)
        if self._mode == 'supervised':
            generative_clf_loss = self._classifier_loss(logits, label)
        else:
            generative_clf_loss = 0

        return generative_clf_loss, label_onehot

    @overrides
    def forward(self, tokens, label):
        """
        Run one step of VAE with RNN decoder
        """

        # encode tokens
        input = self._encode(tokens=tokens)

        # generate labels
        generative_clf_loss, label_onehot = self._discriminate(tokens, label)
        input['label_repr'] = label_onehot

        # concatenate generated labels and continuous document vecs as input representation
        input_repr = torch.cat([input['cont_repr'], input['label_repr']], 1)

        # use parameterized distribution to compute latent code and KL divergence
        _, kld, theta = self._dist.generate_latent_repr(input_repr, n_sample=1)

        # decode using the latent code.
        decoded_output, flattened_decoded_output = self._decode(encoded_docs=input['encoded_docs'],
                                                                mask=input['mask'],
                                                                theta=theta)

        # compute a reconstruction loss
        reconstruction_loss = self._reconstruction_loss(tokens, flattened_decoded_output)

        # compute marginal likelihood
        nll = reconstruction_loss + generative_clf_loss

        # add in the KLD to compute the ELBO
        elbo = nll + kld.to(nll.device)

        # set output_dict
        output_dict = {}
        output_dict['decoded_output'] = decoded_output
        output_dict['cont_repr'] = input_repr['cont_repr']
        output_dict['theta'] = theta
        output_dict['elbo'] = elbo.mean()
        output_dict['kld'] = kld.mean().data.cpu().numpy()
        output_dict['nll'] = nll.mean().data.cpu().numpy()
        output_dict['reconstruction'] = reconstruction_loss.mean().data.cpu().numpy()
        return output_dict
