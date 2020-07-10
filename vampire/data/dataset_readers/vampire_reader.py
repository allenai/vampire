import logging
from typing import Dict

import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field
from allennlp.data.instance import Instance
from overrides import overrides

from vampire.common.util import load_sparse

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("vampire_reader")
class VampireReader(DatasetReader):
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
                 min_sequence_length: int = 0) -> None:
        super().__init__(lazy=lazy)
        self._sample = sample
        self._min_sequence_length = min_sequence_length

    @overrides
    def _read(self, file_path):
        # load sparse matrix
        mat = load_sparse(file_path)
        # convert to lil format for row-wise iteration
        mat = mat.tolil()
        # optionally sample the matrix
        if self._sample:
            indices = np.random.choice(range(mat.shape[0]), self._sample)
        else:
            indices = range(mat.shape[0])

        for index in indices:
            instance = self.text_to_instance(vec=mat[index].toarray().squeeze())
            if instance is not None and mat[index].toarray().sum() > self._min_sequence_length:
                yield instance

    @overrides
    def text_to_instance(self, vec: str = None) -> Instance:  # type: ignore
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
        fields['tokens'] = ArrayField(vec)
        return Instance(fields)
