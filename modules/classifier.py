from common.util import compute_bow
from allennlp.modules import FeedForward
from allennlp.nn.util import get_text_field_mask
from allennlp.modules import Seq2SeqEncoder
from allennlp.common import Registrable
import torch
from typing import Dict


class Classifier(Registrable, torch.nn.Module):

    default_implementation='feedforward'

    def forward(self, **kwargs):
        raise notImplementedError

@Classifier.register("feedforward")
class FeedForward_CLF(Classifier):
    
    def __init__(self, generate_labels: bool):
        super(FeedForward_CLF, self).__init__()
        self._classifier_loss = torch.nn.CrossEntropyLoss()
        self._generate_labels = generate_labels
        
    def _initialize_classifier_hidden(self, input_dim: int):
        self._classifier_hidden = FeedForward(input_dim=input_dim,
                                              num_layers=1,
                                              hidden_dims=input_dim * 2,
                                              activations=torch.nn.ReLU())

    def _initialize_classifier_out(self, num_labels: int):
        self._num_labels = num_labels
        self._classifier_out = torch.nn.Linear(self._classifier_hidden.get_output_dim(),
                                               num_labels)

    def forward(self, input: torch.FloatTensor, label: torch.IntTensor) -> Dict[str, torch.Tensor]:
        projection = self._classifier_hidden(input)
        logits = self._classifier_out(projection)
        output = {"logits": logits}
        if self._generate_labels:
            gen_label = logits.max(1)[1]
            label_onehot = logits.new_zeros(input.size(0), self._num_labels).float()
            label_onehot = label_onehot.scatter_(1, gen_label.reshape(-1, 1), 1)
            output['label_repr'] = label_onehot
        loss = self._classifier_loss(logits, label)
        output['loss'] = loss
        return output