import pickle
import gzip
import json
from sklearn.utils import shuffle


class ResourceUtils:

    @staticmethod
    def load_data(word_img_file_path):     
        f = gzip.open(word_img_file_path, 'rb')
        data = pickle.load(f, encoding='latin1')
        x, y = data['X'], data['Y']
        f.close()   
        x, y = shuffle(x, y)
        return x, y

    @staticmethod
    def load_ngram_prior(ngram_dict_path):
        with open(ngram_dict_path, 'r') as f:
            ngram_dict = json.load(f)

        return ngram_dict

