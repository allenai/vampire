from typing import List
import regex
from overrides import overrides
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_filter import WordFilter, StopwordFilter
from allennlp.common.file_utils import cached_path

@WordFilter.register('regex_')
class RegexFilterEnhanced(WordFilter):
    """
    A ``RegexFilter`` removes words according to supplied regex patterns.
    Parameters
    ----------
    patterns : ``List[str]``
        Words matching these regex patterns will be removed as stopwords.
    """
    def __init__(self,
                 patterns: List[str]) -> None:
        self._patterns = patterns
        self._joined_pattern = regex.compile("|".join(self._patterns))

    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        stopwords = [word for word in words
                     if not self._joined_pattern.match(word.text)]
        return stopwords


@WordFilter.register('regex_and_stopwords')
class RegexAndStopwordFilter(WordFilter):
    """
    A ``RegexandStopWordFilter`` removes words according to supplied regex patterns.
    Parameters
    ----------
    patterns : ``List[str]``
        Words matching these regex patterns will be removed as stopwords.
    """
    def __init__(self,
                 patterns: List[str],
                 stopword_file: str = None,
                 tokens_to_add: List[str] = None) -> None:
        self._regex_filter = RegexFilterEnhanced(patterns=patterns)
        if stopword_file is not None or tokens_to_add is not None:
            self._stopword_filter = StopwordFilter(stopword_file=cached_path(stopword_file),
                                                   tokens_to_add=tokens_to_add)
        else:
            self._stopword_filter = None

    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        words = self._regex_filter.filter_words(words)
        if self._stopword_filter is not None:
            words = self._stopword_filter.filter_words(words)
        return words
