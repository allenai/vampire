from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('vae_classifier')
class VAEClassifierPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.VAEClassifier` model.
    """

    def predict(self, tokens: str) -> JsonDict:
        return self.predict_json({"tokens" : tokens})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"tokens": "..."}``.
        """
        tokens = json_dict["tokens"]
        label = json_dict['category']
        return self._dataset_reader.text_to_instance(tokens=tokens, category=label)