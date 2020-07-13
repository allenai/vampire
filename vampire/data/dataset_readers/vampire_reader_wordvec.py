import logging
from typing import Dict, Union

import numpy as np
from scipy import sparse
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.instance import Instance
from overrides import overrides
from vampire.common.util import load_sparse
from tqdm import tqdm
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
from io import TextIOWrapper



# from vampire.common.util import load_sparse

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("vampire_wordvec_reader")
class VampireWordVecReader(DatasetReader):
    """
    Reads bag of word vectors from a sparse matrices representing training and validation data.

    Expects a sparse matrix of size N documents x vocab size, which can be created via
    the scripts/preprocess_data.py file.

    The output of ``read`` is a list of ``Instances`` with the field:
        vec: ``ArrayField``

    Parameters
    ----------
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    sample : ``int``, optional, (default = ``None``)
        If specified, we will randomly sample the provided
        number of lines from the dataset. Useful for debugging.
    min_sequence_length : ``int`` (default = ``3``)
        Only consider examples from data that are greater than
        the supplied minimum sequence length.
    """
    def __init__(self,
                 lazy: bool = False,
                 sample: int = None,
                 max_sequence_length: int = 400,
                 min_sequence_length: int = 0) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = WhitespaceTokenizer()
        self._token_indexers = {'tokens': SingleIdTokenIndexer()}
        self._max_sequence_length = max_sequence_length
        self._sample = 200
        self._min_sequence_length = min_sequence_length

    @staticmethod
    def _reservoir_sampling(file_: TextIOWrapper, sample: int):
        """
        A function for reading random lines from file without loading the
        entire file into memory.

        For more information, see here: https://en.wikipedia.org/wiki/Reservoir_sampling

        To create a k-length sample of a file, without knowing the length of the file in advance,
        we first create a reservoir array containing the first k elements of the file. Then, we further
        iterate through the file, replacing elements in the reservoir with decreasing probability.

        By induction, one can prove that if there are n items in the file, each item is sampled with probability
        k / n.

        Parameters
        ----------
        file : `_io.TextIOWrapper` - file path
        sample_size : `int` - size of random sample you want

        Returns
        -------
        result : `List[str]` - sample lines of file
        """
        # instantiate file iterator
        file_iterator = iter(file_)

        try:
            # fill the reservoir array
            result = [next(file_iterator) for _ in range(sample)]
        except StopIteration:
            raise ConfigurationError(f"sample size {sample} larger than number of lines in file.")

        # replace elements in reservoir array with decreasing probability
        for index, item in enumerate(file_iterator, start=sample):
            sample_index = np.random.randint(0, index)
            if sample_index < sample:
                result[sample_index] = item

        for line in result:
            yield line

    def _truncate(self, tokens):
        """
        truncate a set of tokens using the provided sequence length
        """
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[: self._max_sequence_length]
        return tokens

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            if self._sample is not None:
                data_file = self._reservoir_sampling(data_file, self._sample)
            for line in data_file:
                instance = self.text_to_instance(text=line)
                if instance is not None and instance.fields['tokens'].tokens:
                    yield instance

    @overrides
    def text_to_instance(self, text: str) -> Instance:  # type: ignore
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
        tokens = self._tokenizer.tokenize(text)
        if self._max_sequence_length is not None:
            tokens = self._truncate(tokens)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        return Instance(fields)

