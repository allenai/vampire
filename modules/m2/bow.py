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
from common.util import compute_bow, split_instances, log_standard_categorical


@VAE.register("bag_of_words_vae")
class BOW_VAE(VAE):
    """
    Implementation of a VAE with an BOW-based decoder.

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
    classifier: ``FeedForward``
        Feedforward classifier for label generation
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
                 encoder: FeedForward,
                 decoder: FeedForward,
                 classifier: FeedForward,
                 num_labels: int,
                 distribution: str = "normal",
                 hidden_dim: int = 128,
                 latent_dim: int = 50,
                 kl_weight: float = 1.0,
                 dropout: float = 0.2,
                 pretrained_file: str = None,
                 freeze_pretrained_weights: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(BOW_VAE, self).__init__()
        self.vocab = vocab
        self._num_labels = num_labels
        self._unlabel_index = self.vocab.get_token_to_index_vocabulary("labels").get("-1")
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
            # VAE with VMF prior. kappa is fixed at 100
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
        h_dim = self.hidden_dim
        self._theta_projection_h = torch.nn.Linear(self.latent_dim, h_dim)
        self._theta_projection_c = torch.nn.Linear(self.latent_dim, h_dim)

        # projection to reconstruct input
        self._decoder_out = torch.nn.Linear(self._decoder.get_output_dim(),
                                            self.vocab.get_vocab_size("full") - 2)
        self._reconstruction_criterion = torch.nn.CrossEntropyLoss()

        self.stopword_indicator = torch.zeros(self.vocab.get_vocab_size("full"))
        indices = [self.vocab.get_token_to_index_vocabulary('full')[x]
                   for x in self.vocab.get_token_to_index_vocabulary('full').keys()
                   if x in ('@@PADDING@@', '@@UNKNOWN@@')]
        self.stopword_indicator[indices] = 1

        if pretrained_file is not None:
            if os.path.isfile(pretrained_file):
                archive = load_archive(pretrained_file)
                self._initialize_weights_from_archive(archive, freeze_pretrained_weights)
            else:
                logger.error("model file for initializing weights is passed, but does not exist.")
        else:
            initializer(self)

    @overrides
    def _initialize_weights_from_archive(self, archive: Archive, freeze_weights: bool = False) -> None:
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
        Encode the tokens into embeddings.

        Params
        ______

        tokens: ``Dict[str, torch.Tensor]`` - tokens to embed

        Returns
        _______

        input_repr: ``Dict[str, torch.Tensor]``
            Dictionary containing:
                - onehot document vectors
        """
        onehot_repr = compute_bow(tokens,
                                  self.vocab.get_index_to_token_vocabulary("full"),
                                  self.stopword_indicator)
        onehot_proj = self._encoder(onehot_repr)
        encoded_input = {'onehot_repr': onehot_repr,
                         'onehot_proj': onehot_proj}
        return encoded_input

    @overrides
    def _decode(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode theta into reconstruction of input using a feedforward network.

        Params
        ______

        theta: ``torch.Tensor``
            latent code

        Returns
        _______

        decoded_output: ``torch.Tensor``
            output of decoder
        """
        # reconstruct input
        logits = self._decoder(theta)
        decoded_output = self._decoder_out(logits)
        # x_recon = self._batch_norm_xrecon(x_recon)
        decoded_output = torch.nn.functional.softmax(decoded_output, dim=1)
        return decoded_output

        
    @overrides
    def _reconstruction_loss(self,
                             onehot_repr: torch.FloatTensor,
                             decoder_output: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculate reconstruction loss between input tokens and output of decoder

        Params
        ______

        onehot_repr: ``torch.FloatTensor``
            onehot representation of documents

        decoder_output: ``torch.FloatTensor``
            output of decoder, a projection to the vocabulary
        """
        return -torch.sum(onehot_repr * (decoder_output + 1e-10).log(), dim=-1)

    @overrides
    def _discriminate(self, encoded_input, label) -> Tuple[torch.FloatTensor, torch.IntTensor]:
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
        clf_out = self._classifier(encoded_input['onehot_proj'])
        logits = self._classifier_logits(clf_out)
        gen_label = logits.max(1)[1]
        label_onehot = clf_out.new_zeros(encoded_input['onehot_proj'].size(0), self._num_labels).float()
        label_onehot = label_onehot.scatter_(1, gen_label.reshape(-1, 1), 1)
        is_labeled = (label != -1).nonzero().squeeze()
        generative_clf_loss = is_labeled.float() * self._classifier_loss(logits, label)
        return generative_clf_loss, logits, label_onehot

        
    def _run(self,  
             tokens: Dict[str, torch.Tensor],
             label: torch.IntTensor,
             metadata: torch.IntTensor=None,
             embedded_tokens: torch.FloatTensor=None):
        # encode tokens
        encoded_input = self._encode(tokens=tokens)

        # generate labels
        generative_clf_loss, logits, label_onehot = self._discriminate(encoded_input, label)
        encoded_input['label_repr'] = label_onehot

        # concatenate generated labels and onehot document vecs as input representation
        input_repr = torch.cat([encoded_input['onehot_proj'], encoded_input['label_repr']], 1)

        # use parameterized distribution to compute latent code and KL divergence
        _, kld, theta = self._dist.generate_latent_code(input_repr, n_sample=1)

        # decode using the latent code.
        decoded_output = self._decode(theta=theta)

        # compute a reconstruction loss
        reconstruction_loss = self._reconstruction_loss(encoded_input['onehot_repr'],
                                                        decoded_output)

        y_prior = log_standard_categorical(label)
        # compute marginal likelihood
        nll = reconstruction_loss - y_prior 

        # add in the KLD to compute the ELBO
        elbo = nll + kld.to(nll.device) + generative_clf_loss

        output = {'logits': logits,
                  'elbo': elbo.mean(),
                  'nll': nll,
                  'kld': kld,
                  'reconstruction': reconstruction_loss,
                  'generative_clf_loss': generative_clf_loss,
                  'decoded_output': decoded_output,
                  'theta': theta} 
        return output

    @overrides
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.IntTensor,
                metadata: torch.IntTensor=None,
                embedded_tokens: torch.FloatTensor=None) -> Dict[str, torch.Tensor]:
        """
        Run one step of VAE with RNN decoder
        """
        # generate labels
        labeled_instances, unlabeled_instances = split_instances(tokens, self._unlabel_index, label, metadata, embedded_tokens)

        if labeled_instances:
            labeled_output = self._run(**labeled_instances)
            loss = labeled_output['elbo']
        else:
            loss = 0

        if unlabeled_instances:
            batch_size = unlabeled_instances['tokens']['tokens'].shape[0]
            device = unlabeled_instances['tokens']['tokens'].device
            elbos = torch.zeros(self._num_labels, batch_size).to(device)
            
            for i in range(1, self._num_labels + 1):
                artificial_label = (torch.ones(batch_size).long() * i).to(device=device)
                unlabeled_output = self._run(**unlabeled_instances, label=artificial_label)
                elbos[i-1] = unlabeled_output['elbo']

            unlabeled_output['logits'] = torch.nn.functional.softmax(unlabeled_output['logits'], dim=-1)
            L_weighted = torch.sum(unlabeled_output['logits'] * elbos.t(), dim=-1)
            H = -torch.sum(unlabeled_output['logits'] * torch.log(unlabeled_output['logits'] + 1e-8), dim=-1)
            loss += torch.sum(L_weighted + H)


        # set output_dict
        output_dict = {}

        output_dict['elbo'] = loss
        output_dict['flattened_decoded_output'] = labeled_output['decoded_output']
        if labeled_instances:
            output_dict['generative_clf_loss'] = labeled_output['generative_clf_loss']
            output_dict['l_kld'] = labeled_output['kld'].data.cpu().numpy()
            output_dict['l_nll'] = labeled_output['nll'].data.cpu().numpy()
            output_dict['l_logits'] = labeled_output['logits']
            output_dict['l_recon'] = labeled_output['reconstruction'].data.cpu().numpy()
            
        if unlabeled_instances:
            output_dict['generative_clf_loss'] = unlabeled_output['generative_clf_loss']
            output_dict['u_kld'] = unlabeled_output['kld'].data.cpu().numpy()
            output_dict['u_nll'] = unlabeled_output['nll'].data.cpu().numpy()
            output_dict['u_recon'] = unlabeled_output['reconstruction'].data.cpu().numpy()
        return output_dict
