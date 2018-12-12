from typing import Dict, List
import logging
import numpy as np
import re
import json
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_filter import StopwordFilter
import itertools
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("textcat")
class TextCatReader(DatasetReader):
    """
    General reader for text classification datasets.

    Reads tokens and their labels in a tsv format.

    ``full`` namespace contains tokenized text with the full vocabulary.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``

    Parameters
    ----------
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    debug : ``bool``, optional, (default = ``False``)
        Whether or not to run in debug mode, 
        where we just subsample the dataset.
    """
    def __init__(self,
                 lazy: bool = False,
                 remove_labels : bool = False,
                 read_filtered_data: bool = False,
                 unlabeled_data: str = None,
                 add_stop_end_tokens: bool = False,
                 max_seq_length: int = None,
                 token_indexers = None,
                 debug: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.debug = debug
        self.unlabeled_filepath = unlabeled_data
        self._max_seq_length = max_seq_length
        self._read_filtered_data = read_filtered_data
        self.remove_labels = remove_labels
        self._add_stop_end_tokens = add_stop_end_tokens
        if add_stop_end_tokens:
            self._full_word_tokenizer = WordTokenizer(start_tokens=["@@START@@"], end_tokens=["@@END@@"])
        else:
            self._full_word_tokenizer = WordTokenizer()
        self._full_token_indexers = {
            "tokens": SingleIdTokenIndexer(namespace="full", lowercase_tokens=True)
        }

    def _get_lines(self, file_path, unlabeled=False):
        if unlabeled:
            logger.info("Reading training data from {}".format(self.unlabeled_filepath))
            with open(cached_path(file_path), "r") as labeled_data_file, open(cached_path(self.unlabeled_filepath), "r") as unlabeled_data_file:
                if self.debug:
                    labeled_lines = np.random.choice(labeled_data_file.readlines(), 100)
                    unlabeled_lines = np.random.choice(unlabeled_data_file.readlines(), 100)
                else:
                    labeled_lines = labeled_data_file.readlines()
                    unlabeled_lines = unlabeled_data_file.readlines() 
        else:
            with open(cached_path(file_path), "r") as labeled_data_file:
                if self.debug:
                    labeled_lines = np.random.choice(labeled_data_file.readlines(), 100)
                else:
                    labeled_lines = labeled_data_file.readlines()
                unlabeled_lines = []
        return unlabeled_lines, labeled_lines 

    @overrides
    def _read(self, file_path):
        if self.unlabeled_filepath is not None:
            unlabeled_lines, labeled_lines = self._get_lines(file_path, unlabeled=True)
        else:
            unlabeled_lines, labeled_lines = self._get_lines(file_path, unlabeled=False)
        labeled = True
        for line in itertools.chain(labeled_lines, ["FLAG"], unlabeled_lines):
            if not line:
                continue
            if line == "FLAG":
                labeled = False
                continue
            items = json.loads(line)
            if self._read_filtered_data:
                tokens = items.get("stopless")
                if tokens is None:
                    raise ConfigurationError("filter stopwords on {} with bin.filter_stopwords script if you'd like to run dataset reader with filter_stopwords flag set.".format(file_path))
            else:
                tokens = items["tokens"]
            if labeled and not self.remove_labels:
                category = str(items["category"])
            else:
                category = 'NA'
            instance = self.text_to_instance(tokens=tokens,
                                             category=category)
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(self, tokens: List[str], category: str = None) -> Instance:  # type: ignore
        """

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        category ``str``, optional, (default = None).
            The category for this sentence.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The category label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        full_tokens = self._full_word_tokenizer.tokenize(tokens)
        if not full_tokens:
            return None
        if self._max_seq_length is not None:
            if self._add_stop_end_tokens:
                inputs = [Token('@@START@@')] + full_tokens
            else:
                inputs = full_tokens
            inputs = inputs[:self._max_seq_length]
            targets = full_tokens[:self._max_seq_length-1]
            if self._add_stop_end_tokens:
                targets = targets + [Token('@@END@@')]
                
        fields['tokens'] = TextField(inputs,
                                     self._full_token_indexers)
        fields['targets'] = TextField(targets,
                                      self._full_token_indexers)
        if category is not None:
            if category in ('NA', 'None'):
                category = str(-1)
            fields['label'] = LabelField(category)
        return Instance(fields)
