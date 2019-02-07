from typing import Optional

from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator

@Model.register("classifier")  # pylint: disable=abstract-method
class Classifier(Model):

    def __init__(self, vocab,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
