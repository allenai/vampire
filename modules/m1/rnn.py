import torch
import numpy as np
import os
from allennlp.nn.util import get_text_field_mask
from typing import Dict, Optional, List, Any, Tuple
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
from common.util import compute_bow, split_instances, log_standard_categorical, schedule


@VAE.register("rnn_vae")
class RNN_VAE(VAE):
    """
    Implementation of a VAE with an RNN-based decoder.

    Params
    ______

    vocab: ``Vocabulary``
        Vocabulary to use
    text_field_embedder: ``TextFieldEmbedder``
        text field embedder
    encoder: ``Seq2SeqEncoder``
        VAE encoder
    decoder: ``FeedForward``
        Feedforward decoder to vocabulary
    mode: ``str``
        mode to run VAE in (supervised or unsupervised)
    distribution: ``str``
        distribution type
    hidden_dim: ``int``
        hidden dimension of VAE
    latent_dim: ``int``
        latent code dimension of VAE
    kl_weight: ``float``
        weight to apply to KL divergence
    dropout: ``float``
        dropout applied at various layers of VAE
    pretrained_file: ``str``
        pretrained VAE file
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder: Seq2SeqEncoder,
                 num_labels: int,
                 distribution: str="gaussian",
                 vmf_kappa: int = None,
                 hidden_dim: int = 128,
                 latent_dim: int = 50,
                 kl_weight: float = 1.0,
                 kl_weight_annealing: str = 'sigmoid',
                 dropout: float = 0.2,
                 pretrained_file: str = None,
                 freeze_pretrained_weights: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(RNN_VAE, self).__init__()
        self.vocab = vocab
        self._num_labels = num_labels
        self._unlabel_index = self.vocab.get_token_to_index_vocabulary("labels").get("-1")
        self._text_field_embedder = text_field_embedder
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self._encoder = encoder
        self._decoder = decoder
        self.dropout = dropout
        self.pretrained_file = pretrained_file
        self.weight_scheduler = lambda x: schedule(x, kl_weight_annealing)
        # Initialize distribution parameter networks and priors
        param_input_dim = self._encoder.get_output_dim()
        self._dist = distribution
        if distribution == 'gaussian':
            self._dist = Normal(hidden_dim=param_input_dim,
                                latent_dim=self.latent_dim)
        elif distribution == "vmf":
            assert vmf_kappa is not None
            # VAE with VMF prior.
            self._dist = VMF(kappa=vmf_kappa,
                             hidden_dim=param_input_dim,
                             latent_dim=self.latent_dim)
        else:
            logger.error("{} is not a distribution that is not supported.".format(distribution))

        self._encoder_dropout = torch.nn.Dropout(dropout)
        self._decoder_dropout = torch.nn.Dropout(dropout)

        # create theta projections into initial states for RNN decoder
        h_dim = self.hidden_dim * 2 if self._encoder.is_bidirectional else self.hidden_dim
        self._theta_projection_h = torch.nn.Linear(self.latent_dim, h_dim)
        self._theta_projection_c = torch.nn.Linear(self.latent_dim, h_dim)

        # projection to reconstruct input
        self._decoder_out = torch.nn.Linear(self._decoder.get_output_dim(),
                                            self.vocab.get_vocab_size("full"))
        self._reconstruction_criterion = torch.nn.CrossEntropyLoss()

        if pretrained_file is not None:
            if os.path.isfile(pretrained_file):
                archive = load_archive(pretrained_file)
                self._initialize_weights_from_archive(archive, freeze_pretrained_weights)
            else:
                logger.error("model file for initializing weights is passed, but does not exist.")
        else:
            initializer(self)

    @overrides
    def _initialize_weights_from_archive(self,
                                         archive: Archive,
                                         freeze_weights: bool = False) -> None:
        """
        Initialize weights (theta?) from a model archive.

        Params
        ______
        archive : `Archive`
            pretrained model archive
        """
        model_parameters = dict(self.named_parameters())
        archived_parameters = dict(archive.model.named_parameters())
        for item, val in archived_parameters.items():
            new_weights = val.data
            item_sub = ".".join(item.split('.')[1:])
            model_parameters[item_sub].data.copy_(new_weights)
            if freeze_weights:
                item_sub = ".".join(item.split('.')[1:])
                model_parameters[item_sub].requires_grad = False
    
    def _freeze_weights(self) -> None:
        model_parameters = dict(self.named_parameters())
        for item in model_parameters:
            model_parameters[item].requires_grad = False

    @overrides
    def _encode(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        encoded_input = {
                      'embedded_text': embedded_text,
                      'encoder_output': encoded_docs,
                      'mask': mask,
                      'cont_repr': cont_repr
                    }
        return encoded_input

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
        flattened_decoded_output = self._decoder_out(flattened_decoded_output)
        # x_recon = self._batch_norm_xrecon(x_recon)
        # x_recon = torch.nn.functional.softmax(x_recon, dim=1)
        return decoded_output, flattened_decoded_output

    @overrides
    def _reconstruction_loss(self,
                             tokens: torch.LongTensor,
                             decoder_output: torch.FloatTensor,
                             mask: torch.LongTensor) -> torch.FloatTensor:
        """
        Calculate reconstruction loss between input tokens and output of decoder

        Params
        ______

        tokens: ``Dict[str, torch.Tensor]``
            input tokens

        decoder_output: ``torch.FloatTensor``
            output of decoder, a projection to the vocabulary
        """
        return self._reconstruction_criterion(decoder_output[mask.view(-1).byte(), :],
                                              torch.masked_select(tokens['tokens'].view(-1),
                                                                  mask.view(-1).byte()))

    def _run(self,  
             tokens: Dict[str, torch.Tensor],
             epoch_num: int,
             metadata: torch.IntTensor=None,
             embedded_tokens: torch.FloatTensor=None):
        # encode tokens
        encoded_input = self._encode(tokens=tokens)

        # concatenate generated labels and continuous document vecs as input representation
        input_repr = torch.cat([encoded_input['cont_repr']], 1)

        # use parameterized distribution to compute latent code and KL divergence
        _, kld, theta = self._dist.generate_latent_code(input_repr, n_sample=1)

        # decode using the latent code.
        decoded_output, flattened_decoded_output = self._decode(encoded_docs=encoded_input['embedded_text'],
                                                                mask=encoded_input['mask'],
                                                                theta=theta)

        # compute a reconstruction loss
        reconstruction_loss = self._reconstruction_loss(tokens, flattened_decoded_output, encoded_input['mask'])

        # compute marginal likelihood
        nll = reconstruction_loss

        # add in the KLD to compute the ELBO
        elbo = nll + kld.to(nll.device) * self.weight_scheduler(epoch_num)
        output = {'elbo': elbo,
                  'nll': nll,
                  'kld': kld * self.weight_scheduler(epoch_num),
                  'encoder_output': encoded_input['encoder_output'],
                  'reconstruction': reconstruction_loss,
                  'theta': theta,
                  'flattened_decoded_output': flattened_decoded_output} 
        return output

    @overrides
    def forward(self,
                epoch_num: int,
                tokens: Dict[str, torch.Tensor],
                label: torch.IntTensor=None,  
                metadata: torch.IntTensor=None,
                embedded_tokens: torch.FloatTensor=None) -> Dict[str, torch.Tensor]:
        """
        Run one step of VAE with RNN decoder
        """
        # generate labels
        # _, unlabeled_instances = split_instances(tokens, self._unlabel_index, label, metadata, embedded_tokens)
        
        unlabeled_output =  self._run(tokens=tokens, epoch_num=epoch_num, metadata=metadata, embedded_tokens=embedded_tokens)

        loss = unlabeled_output['elbo'].mean()


        # set output_dict
        output_dict = {}
        output_dict['elbo'] = loss
        output_dict['flattened_decoded_output'] = unlabeled_output['flattened_decoded_output']
        output_dict['u_encoder_output'] = unlabeled_output['encoder_output']
        output_dict['u_kld'] = unlabeled_output['kld'].data.cpu().numpy()
        output_dict['u_nll'] = unlabeled_output['nll'].data.cpu().numpy()
        output_dict['u_recon'] = unlabeled_output['reconstruction'].data.cpu().numpy()
        return output_dict
