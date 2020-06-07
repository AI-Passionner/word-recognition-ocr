from symspellpy import SymSpell, Verbosity
from src.config import unigrams_frequency_file, bigrams_frequency_file


class SymSpellCheck:
    """
    Leverage SymSpell to perform the spell check in OCR post-processing.
    However, adding a frequency dictionary of your specialized technical domain will have a better performance.
    Besides, it need to consider the punctuation transferring in spell check, too.

    Please refer the original GitHub repositories.
    https://github.com/mammothb/symspellpy  (python -m pip install -U symspellpy )
    https://github.com/wolfgarbe/symspell
    http://storage.googleapis.com/books/ngrams/books/datasetsv2.html (google ngram)
    """

    # term_index is the column of the term and count_index is the column of the term frequency
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary(unigrams_frequency_file, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigrams_frequency_file, term_index=0, count_index=2)

    @classmethod
    def unigrams_check(cls, word):
        candidates = cls.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True,
                                          transfer_casing=True)
        candidate = candidates[0]  # choose the first option (term, term frequency, and edit distance)
        return candidate.term, candidate.count, candidate.distance

    @classmethod
    def ngrams_check(cls, multi_word_string):
        candidates = cls.sym_spell.lookup_compound(multi_word_string, max_edit_distance=2, transfer_casing=True)
        candidate = candidates[0]  # choose the first option (term, term frequency, and edit distance)
        return candidate.term, candidate.count, candidate.distance

    @classmethod
    def word_segmentation(cls, input_string):
        result = cls.sym_spell.word_segmentation(input_string)
        return result.corrected_string, result.distance_sum, result.log_prob_sum
