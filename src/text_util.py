import Levenshtein
import re

from src.spell_check import SymSpellCheck


class TextUtils(object):
    @staticmethod
    def word_freq(word, ngram_dict):
        """
        Find a word's probability
        :param word: alphabet string
        :param ngram_dict: a dictionary of {word: probability}, where probability is just a ranking value
        """
        word = word.lower()
        return ngram_dict[word] if word in ngram_dict else 0

    @staticmethod
    def create_ngrams(word_list, n):
        """
        Computes n-grams from the list of words.
        :param word_list: list of words on which n_grams have to be computed.
        :param n: number of grams.
        :return: zipped collection of n-grams.
        """
        return zip(*[word_list[i:] for i in range(n)])

    @staticmethod
    def get_best_candidate(word, ngram_dict, threshold=0.8):
        """
        Find the best candidate, give a word.
        :param word: alphabet string
        :param ngram_dict: a dictionary of {word: frequency}, where the frequency is just a ranking parameter from a text corporus
        :param threshold: a float, a adaption of Levenshtein edit distance in spell check.
        :return: a tuple of (string, float), the best candidate and its probability of a given word
        """
        candidates = []
        w_l = len(word)
        freq = ngram_dict[word] if word in ngram_dict else 0.0
        if w_l >= 5:
            for uniq_word in ngram_dict:
                if word != uniq_word:
                    edit_dist = Levenshtein.distance(word, uniq_word)
                    levenshtein_ratio = 1.0 - edit_dist / w_l
                    if levenshtein_ratio >= threshold:
                        candidates.append([uniq_word, ngram_dict[uniq_word], edit_dist])
                else:
                    candidates.append([word, ngram_dict[word], edit_dist])

            if len(candidates) == 0:
                return word, freq, 0

            candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
            return candidates[0]
        else:
            return word, freq, 0


def bigrams_spellcheck(word_dict):
    """
    Correct OCR errors with bi-grams spell check. However, it should be mentioned that the spell check need to be
    used carefully. It seems to be easily over used. Therefore, the spell check should be adapted for different
    purposes. For example, we can only apply the bi-grams spell check to correct particular key words for the AI
    application.

    :param word_dict: python dictionary, storing word-level OCR
    :return: updated word_dict after spell check
    """
    all_word_ids = list(word_dict.keys())
    for bi in TextUtils.create_ngrams(all_word_ids, 2):
        bi_words = word_dict[bi[0]]['text'], word_dict[bi[1]]['text']
        bi_words = [SymSpellCheck.unigrams_check(w)[0] for w in bi_words]  # uni-grams check for better accuracy
        bi_text = ' '.join(w.strip() for w in bi_words)

        # add filters to reduce the computation
        torf1 = len(re.sub('[a-z]', '', bi_text)) <= 2       # consider a maximum of 2 non-letters only
        torf2 = len(bi_text) >= 5                            # check word of minimum 5 characters
        if torf1 and torf2:
            new_text = SymSpellCheck.unigrams_check(bi_text)[0]
            new_words = new_text.split()
            if len(new_words) == 2:
                bi_words = new_words

        word_dict[bi[0]]['text'], word_dict[bi[1]]['text'] = bi_words

    return word_dict
