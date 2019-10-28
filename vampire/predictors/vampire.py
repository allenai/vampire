from copy import deepcopy
from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer


@Predictor.register("vampire")
class VampirePredictor(Predictor):
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"label"`` to the output.
        """
        sentence = json_dict["text"]
        if not hasattr(self._dataset_reader, "tokenizer") and not hasattr(
            self._dataset_reader, "_tokenizer"
        ):
            tokenizer = WordTokenizer()
            sentence = tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(sentence)
