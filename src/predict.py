import numpy as np
import itertools
import logging
import traceback

from tensorflow.keras.models import model_from_json
from src.config import characters
from src.train_util import TrainingUtils
from src.config import model_json_file, model_weight_file, input_shape

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RecognitionUtils:
    MODEL = None

    @classmethod
    def load_model(cls):
        try:
            json_file = open(model_json_file)
            model_json = json_file.read()
            json_file.close()
            cls.MODEL = model_from_json(model_json)
            cls.MODEL.load_weights(model_weight_file)
            print("A pre-trained model was loaded")

        except Exception as exception:
            trc = traceback.format_exc()
            raise Exception('Exception found while loading the model, {}'.format(trc))

        return
    
    @classmethod
    def decode_batch(cls, crnn_out):
        """
        Run the pre-trained model to recognize each word image. It assumes word_images are clean and not empty
        Parameters
        ----------
        crnn_out:

        Returns
        -------
        return

        """
        ret = []
        for j in range(crnn_out.shape[0]):
            probs_array = crnn_out[j, 2:]
            pred_cls = list(np.argmax(probs_array, 1))
            pred_probs = [probs_array[i][pred_cls[i]] for i in range(len(pred_cls))]

            output = []
            index = 0
            for k, g in itertools.groupby(pred_cls):  
                n = len(list(g))
                if k != len(characters)-1:           # k!= 94 or a character is not a whitespace  
                    output.append([pred_cls[index], np.max(pred_probs[index:index+n])])
    
                index += n
    
            outstr = ''
            char_probs = [] 
            for i, p in output:
                if i < len(characters):
                    outstr += characters[i]
                    char_probs.append(p + np.finfo(float).eps)
            
            ret.append([outstr, np.mean(char_probs)])

        return ret

    @classmethod
    def convert_cnn_input(cls, word_imgs):
        """
        Normlize original word images to be fed into CRNN model. This normalization must be same as in preparing training data

        Parameters
        ----------
        word_imgs : a dictionary, {word_id: word_image, ...}

        Returns
        -------
        word_images : a dictionary, {word_id: trans_word_image, ...}

        """
        try:
            width, height, channel = input_shape
            if height != 32 or width != 400 or channel != 1:
                raise Exception('Error found in the image input shape. It must be (32, 400, 1) in prediction')

        except Exception as exception:
            trc = traceback.format_exc()
            raise Exception('Error found in the image shape, {} \n {}'.format(str(exception), trc))

        else:
            imgs_to_crnn = {}
            for k in word_imgs:
                img = TrainingUtils.norm_img(word_imgs[k], (height, width), 2)  # transform original images to fit the CNN
                img = img.astype(np.float32)
                if np.max(img) > 0:
                    img /= np.max(img)
                else:
                    img = np.ones((height, width))  # pad with 1s

                img = np.expand_dims(img.T, -1)
                imgs_to_crnn[k] = img

        return imgs_to_crnn

    @classmethod
    def recognize(cls, word_imgs):
        """
        Run the pre-trained model to recognize each word image. It assumes word_images are clean and not empty
        Parameters
        ----------
        input_model : pre-trained model
        input_shape : (height, width, channel), image shape fed into the CNN
        word_images : a dictionary, {word_id: word_image, ...}          

        Returns
        -------
        recognized_words : a dictionary, {word_id: {'Text': word, 'condifence': conf}, ...}    

        """

        imgs_to_crnn = RecognitionUtils.convert_cnn_input(word_imgs)
        word_ids = list(imgs_to_crnn.keys())
        x = np.array([imgs_to_crnn[k] for k in imgs_to_crnn])
        y_pred = RecognitionUtils.MODEL.predict(x, len(x))
        result = RecognitionUtils.decode_batch(y_pred)
        recog_words = {i: {'Text': word, 'Confidence': conf} for i, (word, conf) in zip(word_ids, result)}
    
        return recog_words 
    

# load model
try:
    RecognitionUtils.load_model()

except Exception as error:
    logger.error("Could't load the pre-trained CNN model, {}".format(str(error)))