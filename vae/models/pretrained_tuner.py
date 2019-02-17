from typing import Dict, Optional, Union, List, Any

import numpy
import torch
from torch import nn

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, Average

from allennlp.modules import TextFieldEmbedder, TokenEmbedder



@Model.register("text_classification_tune_pretrained")
class TextClassificationTunePretrained(Model):
    """
    Used for fine tuning Pretrained Embeddings for text classification.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.vocab = vocab

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self.text_field_embedder = text_field_embedder
        self.prediction_dim = text_field_embedder.get_output_dim()

        num_classes = self.vocab.get_vocab_size("labels")
        self.output_layer = torch.nn.Linear(self.prediction_dim, num_classes)

        self.metrics = {"accuracy": CategoricalAccuracy()}
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                epoch_num=None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``.
        label : torch.LongTensor, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a
            distribution over the label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        context_vectors = self.text_field_embedder(tokens)

        # context_vectors is shape (batch_size, timesteps, embed_dim)
        # with the top layer LM output
        mask = util.get_text_field_mask(tokens)

        # make the vector for prediction
        # masked max
        # broadcast_mask = mask.unsqueeze(-1).float()
        # one_minus_mask = (1.0 - broadcast_mask).byte()
        # replaced = context_vectors.masked_fill(one_minus_mask, -1e-7)
        # max_value, _ = replaced.max(dim=1, keepdim=False)

        if self.dropout:
            context_vectors = self.dropout(context_vectors)

        logits = self.output_layer(context_vectors)
        class_probabilities = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {'logits': logits, 'class_probabilities': class_probabilities}
        if label is not None:
            loss = self.loss(logits, label)
            # metrics
            self.metrics['accuracy'](logits, label)
            output_dict['loss'] = loss

        return output_dict


    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["class_probabilities"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self.metrics['accuracy'].get_metric(reset)}
        return metrics


if __name__ == '__main__':
    from allennlp.common import Params
    from allennlp.data import DatasetReader, DataIterator

    params = Params.from_file('training_config/sst_tune_elmo.json')
    reader = DatasetReader.from_params(params.pop("dataset_reader"))

    instances = reader.read(params.pop("train_data_path"))
    vocab = Vocabulary.from_instances(instances)

    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)
    for batch in iterator(instances, num_epochs=1, shuffle=False):
        break

    model = Model.from_params(params.pop("model"), vocab=vocab)

    tokens = batch['tokens']
    label = batch['label']


