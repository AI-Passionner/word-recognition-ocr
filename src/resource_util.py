import pickle
import gzip
from sklearn.utils import shuffle


class ResourceUtils:

    @staticmethod
    def load_data(word_img_file_path):     
        f = gzip.open(word_img_file_path , 'rb')
        data = pickle.load(f, encoding='latin1')
        x, y = data['X'], data['Y']
        f.close()   
        x, y = shuffle(x, y)
        return x, y

