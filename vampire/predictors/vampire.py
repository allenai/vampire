from copy import deepcopy
from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers import Token


@Predictor.register("vampire")
class VampirePredictor(Predictor):
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"label"`` to the output.
        """
        sentence = [Token(x) for x in json_dict["text"].split(" ")]
        return self._dataset_reader.text_to_instance(sentence)
    
    def _array_to_instance(self, arr) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"label"`` to the output.
        """
        return self._dataset_reader.text_to_instance(arr)

