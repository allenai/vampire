from typing import Dict, Union

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

from vampire.modules.encoder import Encoder


@Model.register("classifier")
class Classifier(Model):
    """
    Generic classifier model. Differs from allennlp's basic_classifier
    in the fact that it uses a custom Encoder, which wraps all seq2vec
    and seq2seq encoders to easily switch between them during
    experimentation.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 input_embedder: TextFieldEmbedder,
                 encoder: Encoder = None,
                 dropout: Union[str, float] = None,
                 initializer: InitializerApplicator = InitializerApplicator()
                ) -> None:
        """
        Parameters
        ----------
        vocab: `Vocabulary`
            vocab to use
        input_embedder: `TextFieldEmbedder`
            generic embedder of tokens
        encoder: `Encoder`, optional (default = None)
            Seq2Vec or Seq2Seq Encoder wrapper. If no encoder is provided,
            assume that the input is a bag of word counts, for linear classification.
        dropout: `float`, optional (default = None)
            if set, will apply dropout to output of encoder.
        initializer: `InitializerApplicator`
            generic initializer
        """
        super().__init__(vocab)
        self._input_embedder = input_embedder
        if dropout:
            self._dropout = torch.nn.Dropout(float(dropout))
        else:
            self._dropout = None
        self._encoder = encoder
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        if self._encoder:
            self._clf_input_dim = self._encoder.get_output_dim()
        else:
            self._clf_input_dim = self._input_embedder.get_output_dim()
        self._classification_layer = torch.nn.Linear(self._clf_input_dim,
                                                     self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self._input_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        if self._encoder:
            embedded_text = self._encoder(embedded_text=embedded_text,
                                          mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics
