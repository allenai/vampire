from typing import Any, Dict, List, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import (get_final_encoder_states, get_text_field_mask,
                              masked_max, masked_mean)
from allennlp.training.metrics import CategoricalAccuracy

from vae.models.classifier import Classifier


@Classifier.register("seq2seq_classifier")
@Model.register("seq2seq_classifier")
class Seq2SeqClassifier(Classifier):
    """
    This ``Model`` implements a classifier with a seq2seq encoder of text.

    See allennlp.modules.seq2seq_encoders for available encoders.

    Parameters
    ----------
    vocab : ``Vocabulary``
    input_embedder : ``TextFieldEmbedder``
        Used to embed the ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        Used to encode the text
    classification_layer : ``FeedForward``
        This feedforward network computes the output logits.
    dropout : ``float``, optional (default=0.5)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training
    """
    def __init__(self,
                 vocab: Vocabulary,
                 input_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 aggregations: List[str],
                 classification_layer: FeedForward,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._input_embedder = input_embedder
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._encoder = encoder
        self._aggregations = aggregations
        self._classification_layer = classification_layer
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
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
            Metadata to persist

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self._input_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        encoder_output = self._encoder(embedded_text, mask)

        encoded_repr = []
        for aggregation in self._aggregations:
            if aggregation == "meanpool":
                broadcast_mask = mask.unsqueeze(-1).float()
                context_vectors = encoder_output * broadcast_mask
                encoded_text = masked_mean(context_vectors,
                                           broadcast_mask,
                                           dim=1,
                                           keepdim=False)
            elif aggregation == 'maxpool':
                broadcast_mask = mask.unsqueeze(-1).float()
                context_vectors = encoder_output * broadcast_mask
                encoded_text = masked_max(context_vectors,
                                          broadcast_mask,
                                          dim=1)
            elif aggregation == 'final_state':
                is_bi = self._encoder.is_bidirectional()
                encoded_text = get_final_encoder_states(encoder_output,
                                                        mask,
                                                        is_bi)
            encoded_repr.append(encoded_text)

        encoded_repr = torch.cat(encoded_repr, 1)

        if self.dropout:
            encoded_repr = self.dropout(encoded_repr)

        label_logits = self._classification_layer(encoded_repr)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(label_logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
