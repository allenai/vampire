from typing import Dict, Optional, List, Any
import torch
from modules.encoder import Encoder
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.models.archival import load_archive, Archive
from allennlp.nn.util import get_text_field_mask, masked_mean, get_final_encoder_states

@Model.register("seq2seq_classifier")
class Seq2SeqClassifier(Model):
    """
    This ``Model`` implements a basic logistic regression classifier on onehot embeddings of text.
    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training
    """
    def __init__(self,
                 continuous_embedder: TextFieldEmbedder,
                 encoder: Encoder,
                 vocab: Vocabulary,
                 pretrained_vae_file: str=None,
                 share_encoder: bool=False,
                 share_theta: bool=False,
                 freeze_weights: bool=False) -> None:
        super().__init__(vocab)
        self._vocab = vocab
        self._continuous_embedder = continuous_embedder
        self._vocab_size = vocab.get_vocab_size(namespace="full")
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._share_theta = share_theta
        self._share_encoder = share_encoder
        self._freeze_weights = freeze_weights
        logit_input_dim = encoder._architecture.get_output_dim()
        if pretrained_vae_file is not None:
            archive = load_archive(pretrained_vae_file)
            self._vae = archive.model
            import ipdb; ipdb.set_trace()
            self._vae.eval()
            self._vae.vocab_namespace = "vae"
            self._vae._unlabel_index = None
            if share_encoder:
                logit_input_dim += self._vae._encoder._architecture.get_output_dim()
            if share_theta:
                logit_input_dim += self._vae.latent_dim
            if freeze_weights:
                self._vae._freeze_weights()
        else:
            self._vae = None
        self._encoder = encoder
        self._dropout = torch.nn.Dropout(0.2)
        self._output_feedforward = torch.nn.Linear(logit_input_dim,
                                                   self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                targets: Dict[str, torch.LongTensor],
                vae_tokens: Dict[str, torch.LongTensor]=None,
                label: torch.IntTensor = None,
                is_labeled: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata on tokens to persist
        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # generate onehot bag of words embeddings
        
        embedded_text = self._continuous_embedder(tokens)

        mask = get_text_field_mask(tokens).float()
        
        encoder_output = self._encoder(embedded_text, mask)

        input_reprs = [encoder_output['encoder_output']]
        if self._vae is not None:
            mask = get_text_field_mask(vae_tokens)
            embedded_text = self._vae._continuous_embedder(vae_tokens)
            encoder_output = self._vae._encoder(embedded_text=embedded_text, mask=mask)
            _, _, theta = self._vae._dist.generate_latent_code(encoder_output['encoder_output'], 1)
            if self._share_theta:
                theta = self._dropout(theta)
                input_reprs.append(theta)
            if self._share_encoder:
                encoder_output['encoder_output'] = self._dropout(encoder_output['encoder_output'])
                input_reprs.append(encoder_output['encoder_output'])
        
        repr = torch.cat(input_reprs, 1)
        linear_output = self._output_feedforward(repr)

        label_probs = torch.nn.functional.log_softmax(linear_output, dim=-1)

        output_dict = {"label_logits": linear_output, "label_probs": label_probs}

        loss = self._loss(linear_output, label.long().view(-1))
        self._accuracy(linear_output, label)
        output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
