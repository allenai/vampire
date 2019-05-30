import itertools
import json
import logging
from io import TextIOWrapper
from typing import Dict
import numpy as np
from overrides import overrides
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.instance import Instance
from allennlp.data.fields import LabelField, TextField, Field, ListField, MetadataField, ArrayField
from vampire.common.util import load_sparse

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("vampire_search")
class VampireSearchReader(TextClassificationJsonReader):
    """
    Reads tokens and (optionally) their labels from a from text classification dataset.

    This dataset reader inherits from TextClassificationJSONReader, but differs from its parent
    in that it is primed for semisupervised learning. This dataset reader allows for:
        1) Ignoring labels in the training data (e.g. for unsupervised pretraining)
        2) Reading additional unlabeled data from another file
        3) Throttling the training data to a random subsample (according to the numpy seed),
           for analysis of the effect of semisupervised models on different amounts of labeled
           data

    Expects a "tokens" field and a "label" field in JSON format.

    The output of ``read`` is a list of ``Instances`` with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``, if not ignoring labels.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    tokenizer : ``Tokenizer``, optional (default = ``{"tokens": WordTokenizer()}``)
        Tokenizer to split the input text into words or other kinds of tokens.
    segment_sentences: ``bool``, optional (default = ``False``)
        If True, we will first segment the text into sentences using SpaCy and then tokenize words.
        Necessary for some models that require pre-segmentation of sentences,
        like the Hierarchical Attention Network.
    sequence_length: ``int``, optional (default = ``None``)
        If specified, will truncate tokens to specified maximum length.
    ignore_labels: ``bool``, optional (default = ``False``)
        If specified, will ignore labels when reading data.
    additional_unlabeled_data_path: ``str``, optional (default = ``None``)
        If specified, will additionally read all unlabeled data from this filepath.
        If ignore_labels is set to False, all data in this file should have a
        consistent dummy-label (e.g. "N/A"), to identify examples that are unlabeled
        in a downstream model that uses this dataset reader.
    sample: ``int``, optional (default = ``None``)
        If specified, will sample data to a specified length.
            **Note**:
                1) This operation will *not* apply to any additional unlabeled data
                   (specified in `additional_unlabeled_data_path`).
                2) To produce a consistent subsample of data, use a consistent seed in your
                   training config.
    skip_label_indexing: ``bool``, optional (default = ``False``)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """
    def __init__(self,
                 skip_label_indexing: bool = False,
                 ignore_labels: bool = False,
                 sample: int = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy,
                         skip_label_indexing=skip_label_indexing)
        self._sample = sample
        self._ignore_labels = ignore_labels
        self._skip_label_indexing = skip_label_indexing


    @overrides
    def _read(self, file_path):
        mat = load_sparse(file_path)
        labels = []
        if not self._ignore_labels:
            with open(cached_path(file_path), "r") as data_file:
                for line in data_file:
                    items = json.loads(line)
                    label = str(items.get('label'))
                    labels.append(label)
        
        mat = mat.tolil()
        if not self._ignore_labels:
            for ix in range(mat.shape[0]):
                instance = self.text_to_instance(vec=mat[ix].toarray().squeeze(), label=labels[ix], covariate=None)
                if instance is not None:
                    yield instance
        else:
            for ix in range(mat.shape[0]):
                instance = self.text_to_instance(vec=mat[ix].toarray().squeeze(), label=None, covariate=None)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self, vec: str=None, label: str = None, covariate: str = None, num_covariates: int = None) -> Instance:  # type: ignore
        """
        Parameters
        ----------
        text : ``str``, required.
            The text to classify
        label ``str``, optional, (default = None).
            The label for this text.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The label label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        fields['vec'] = ArrayField(vec)
        
        if label is not None:
            fields['label'] = LabelField(label,
                                         skip_indexing=self._skip_label_indexing)
        if covariate is not None:
            fields['metadata'] = MetadataField({"num_covariates": num_covariates})
            fields['covariate'] = LabelField(covariate, skip_indexing=True)
        return Instance(fields)


