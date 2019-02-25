import json
import logging
from typing import Dict, List
from itertools import groupby

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (Field, LabelField, ListField, MetadataField,
                                  TextField)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("semisupervised_text_classification_json")
class SemiSupervisedTextClassificationJsonReader(DatasetReader):
    """
    Reads tokens and their labels from a labeled text classification dataset.
    Expects a "tokens" field and a "category" field in JSON format.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    tokenizer : ``Tokenizer``, optional (default = ``{"tokens": WordTokenizer()}``)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    segment_sentences: ``bool``, optional (default = ``False``)
        If True, we will first segment the text into sentences using SpaCy and then tokenize words.
        Necessary for some models that require pre-segmentation of sentences,
        like the Hierarchical Attention Network.
    sequence_length: ``int``, optional (default = ``None``)
        If specified, will truncate tokens to specified maximum length.
    ignore_labels: ``bool``, optional (default = ``False``)
        If specified, will ignore labels when reading data, useful for semi-supervised textcat
    skip_label_indexing: ``bool``, optional (default = ``False``)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 unrestricted_tokenizer: Tokenizer = None,
                 segment_sentences: bool = False,
                 sequence_length: int = None,
                 ignore_labels: bool = False,
                 skip_label_indexing: bool = False,
                 sample: int = None,
                 unlabeled_data_path: str = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._unrestricted_tokenizer = unrestricted_tokenizer
        self._sample = sample
        self._segment_sentences = segment_sentences
        self._sequence_length = sequence_length
        self._ignore_labels = ignore_labels
        self._skip_label_indexing = skip_label_indexing
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._unlabeled_data_path = unlabeled_data_path
        if self._segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    def _reservoir_sampling(self, file_):
        """
        reservoir sampling for reading random lines from file without loading
        entire file into memory

        See here for explanation of algorithm:
        https://stackoverflow.com/questions/35680236/select-100-random-lines-from-a-file-with-a-1-million-which-cant-be-read-into-me

        Parameters
        ----------
        file : `str` - file path
        sample_size : `int` - size of random sample you want

        Returns
        -------
        result : `List[str]` - sample lines of file
        """
        file_iterator = iter(file_)

        try:
            result = [next(file_iterator) for _ in range(self._sample)]

        except StopIteration:
            raise ValueError("Sample larger than population")

        for index, item in enumerate(file_iterator, start=self._sample):
            sample_index = np.random.randint(0, index)
            if sample_index < self._sample:
                result[sample_index] = item

        np.random.shuffle(result)

        return result

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            if self._sample is not None:
                lines = [(item, True) for item in self._reservoir_sampling(data_file)]
            else:
                lines = [(item, True) for item in data_file.readlines()]

        if self._unlabeled_data_path:
            with open(cached_path(self._unlabeled_data_path)) as data_file:
                lines += [(item, False) for item in data_file.readlines()]


        for line, is_labeled in lines:
            items = json.loads(line)
            text = items["text"]
            label = str(items.get('label'))
            if self._ignore_labels:
                instance = self.text_to_instance(text=text, label=label, is_labeled=False)
            else:
                instance = self.text_to_instance(text=text, label=label, is_labeled=is_labeled)
            if instance is not None and len(instance.fields['tokens'].tokens) > 0:
                yield instance

    def _truncate(self, tokens):
        """
        truncate a set of tokens using the provided sequence length
        """
        if len(tokens) > self._sequence_length:
            tokens = tokens[:self._sequence_length]
        return tokens

    @overrides
    def text_to_instance(self, text: str, label: str = None, is_labeled: bool = False) -> Instance:  # type: ignore
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
        if self._segment_sentences:
            sentences: List[Field] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                word_tokens = self._tokenizer.tokenize(sentence)
                if self._sequence_length is not None:
                    word_tokens = self._truncate(word_tokens)
                sentences.append(TextField(word_tokens, self._token_indexers))
            fields['tokens'] = ListField(sentences)
        else:

            tokens = self._tokenizer.tokenize(text)
            if self._sequence_length is not None:
                tokens = self._truncate(tokens)

            fields['tokens'] = TextField(tokens, self._token_indexers)

            if self._unrestricted_tokenizer:
                unrestricted_tokens = self._unrestricted_tokenizer.tokenize(text)
                if self._sequence_length is not None:
                    unrestricted_tokens = self._truncate(unrestricted_tokens)
                fields['classifier_tokens'] = TextField(unrestricted_tokens, self._token_indexers)
        if is_labeled or self._unlabeled_data_path is not None:
            # TODO: Document 'default' unsupervised label as pre-condition.
            fields['label'] = LabelField(label, skip_indexing=self._skip_label_indexing)
        fields['metadata'] = MetadataField({"is_labeled": is_labeled})

        return Instance(fields)


@DatasetReader.register("joint_semisupervised_text_classification_json")
class JointSemiSupervisedTextClassificationJsonReader(SemiSupervisedTextClassificationJsonReader):
    """
    Reads tokens and their labels from a labeled text classification dataset.
    Expects a "tokens" field and a "category" field in JSON format.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``

    Being specialized for joint training, it will resample unlabeled data
    as including all unlabeled data that's available causes imbalance between
    labeled and unlabeled instances per batch.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    tokenizer : ``Tokenizer``, optional (default = ``{"tokens": WordTokenizer()}``)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    segment_sentences: ``bool``, optional (default = ``False``)
        If True, we will first segment the text into sentences using SpaCy and then tokenize words.
        Necessary for some models that require pre-segmentation of sentences,
        like the Hierarchical Attention Network.
    sequence_length: ``int``, optional (default = ``None``)
        If specified, will truncate tokens to specified maximum length.
    ignore_labels: ``bool``, optional (default = ``False``)
        If specified, will ignore labels when reading data, useful for semi-supervised textcat
    skip_label_indexing: ``bool``, optional (default = ``False``)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 unrestricted_tokenizer: Tokenizer = None,
                 segment_sentences: bool = False,
                 sequence_length: int = None,
                 ignore_labels: bool = False,
                 skip_label_indexing: bool = False,
                 sample: int = None,
                 unlabeled_data_path: str = None,
                 unlabeled_data_factor: int = 2,
                 lazy: bool = True) -> None:
        super().__init__(
                token_indexers=token_indexers, tokenizer=tokenizer, unrestricted_tokenizer=unrestricted_tokenizer,
                segment_sentences=segment_sentences, sequence_length=sequence_length, ignore_labels=ignore_labels,
                skip_label_indexing=skip_label_indexing, sample=sample, unlabeled_data_path=unlabeled_data_path,
                lazy=lazy)

        self._unlabeled_data_factor = unlabeled_data_factor

        # Save throttled labeled data to prevent resampling.
        self._labeled_lines = None

    def interleave(self, labeled_data, unlabeled_data):
        # Courtesy of https://stackoverflow.com/a/19293966
        num_examples = len(labeled_data) + len(unlabeled_data)

        # For a given list x = [1, 2, 3, 4] and y = [a, b],
        # produces [(1, a) (2, a) (3, b) (4 b)].
        groups = [(labeled_data[(len(labeled_data) * i) // num_examples],
                   unlabeled_data[len(unlabeled_data)*i//num_examples])
                  for i in range(num_examples)]
        groups = groupby(groups, key=lambda x: x[0])

        return [j[i] for k, g in groups for i, j in enumerate(g)]

    @overrides
    def _read(self, file_path):
        # Cache sampled labeled instances to prevent resampling.
        # If no throttle is set, all of the labeled data is used.
        if not self._labeled_lines:
            with open(cached_path(file_path), "r") as data_file:
                if self._sample is not None:
                    labeled_lines = [(item, True) for item in self._reservoir_sampling(data_file)]

                    # From now on, only unlabeled data is sampled.
                    self._sample *= self._unlabeled_data_factor
                else:
                    labeled_lines = [(item, True) for item in data_file.readlines()]

                self._labeled_lines = labeled_lines

        lines = self._labeled_lines.copy()

        # Resample the unlabeled data.
        # If no throttle is set, all of the unlabeled data is used.
        if self._unlabeled_data_path and self._sample is not None and self._sample > 0:
            with open(cached_path(self._unlabeled_data_path)) as data_file:
                if self._sample is not None:
                    unlabeled_lines = [(item, False) for item in self._reservoir_sampling(data_file)]
                else:
                    unlabeled_lines = [(item, False) for item in data_file.readlines()]

            lines = self.interleave(unlabeled_lines, lines)

        for line, is_labeled in lines:
            items = json.loads(line)
            text = items["text"]
            label = str(items['label'])
            if self._ignore_labels:
                instance = self.text_to_instance(text=text, label=label, is_labeled=False)
            else:
                instance = self.text_to_instance(text=text, label=label, is_labeled=is_labeled)
            if instance is not None:
                yield instance
