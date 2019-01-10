from typing import Dict, Optional, List, Any
import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.models.archival import load_archive, Archive
from allennlp.nn.util import get_text_field_mask

@Model.register("logistic_regression")
class LogisticRegression(Model):
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
    def __init__(self, onehot_embedder: TextFieldEmbedder, vocab: Vocabulary, pretrained_vae_file: str=None) -> None:
        super().__init__(vocab)
        self._onehot_embedder = onehot_embedder
        self._vocab_size = vocab.get_vocab_size(namespace="full")
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        logit_input_dim = self._vocab_size
        if pretrained_vae_file is not None:
            archive = load_archive(pretrained_vae_file)
            self._vae = archive.model
            self._vae.vocab = vocab
            self._vae._unlabel_index = None
            
            logit_input_dim += self._vae.latent_dim
           
        else:
            self._vae = None

        self._output_feedforward = torch.nn.Linear(logit_input_dim,
                                                   self._num_labels)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                targets: Dict[str, torch.LongTensor],
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
        
        onehot_repr = self._onehot_embedder(tokens)

        if self._vae is not None:
            mask = get_text_field_mask(tokens)
            encoder_output = self._vae._encoder(embedded_text=onehot_repr, mask=mask)
            _, _, theta = self._vae._dist.generate_latent_code(encoder_output['encoder_output'], 1)
            repr = torch.cat([onehot_repr, theta], 1)
        else:
            repr = onehot_repr

        linear_output = self._output_feedforward(repr)

        label_probs = torch.nn.functional.log_softmax(linear_output, dim=-1)

        output_dict = {"label_logits": linear_output, "label_probs": label_probs}

        loss = self._loss(linear_output, label.long().view(-1))
        self._accuracy(linear_output, label)
        output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
