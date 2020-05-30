import cv2 
import json
import numpy as np
import itertools
from scipy.stats.mstats import gmean

from tensorflow.keras.models import model_from_json
from src.config import characters
from src.train_util import TrainingUtils
from src.image_util import ImageUtils
from src.textract_util import TextractUtils


class Predict:
    
    @staticmethod
    def load_model(model_json_file, model_weight_file):   
        # load model json file
        json_file = open(model_json_file)
        model_json = json_file.read()
        json_file.close()
        
        loaded_model = model_from_json(model_json)    
        loaded_model.load_weights(model_weight_file)
        print("pre-trained model loaded")
        
        return loaded_model
    
    @staticmethod
    def decode_batch(out):
        ret = []
        for j in range(out.shape[0]):        
            probs_array = out[j, 2:]
            pred_cls = list(np.argmax(probs_array , 1))
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
            for i, p in  output:
                if i < len(characters):
                    outstr += characters[i]
                    char_probs.append(p + np.finfo(float).eps)
            
            ret.append([outstr, gmean(char_probs)])

        return ret
    
    @staticmethod
    def imgs_to_words(input_model, input_shape, word_images):    
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
         
        width, height, channel = input_shape
        word_ids = [k for k, v in word_images.items()]
        imgs = [v for k, v in word_images.items()]
        x = np.zeros((len(imgs), width, height, 1), dtype=np.float32)          
        for ii in range(len(imgs)):
            # transform original images to fit the CNN
            img = TrainingUtils.norm_img(imgs[ii], (height, width), 2) 
            img = img.astype(np.float32)
            if np.max(img) > 0:
                img /= np.max(img)
            else: 
                img = np.ones((height, width))  # pad with 1s

            img = np.expand_dims(img.T, -1)
            x[ii] = img                
            
        y_pred = input_model.predict(x, len(x)) 
        result = Predict.decode_batch(y_pred) 
        recog_words = {i: {'Text': word, 'Confidence': conf} for i, (word, conf) in zip(word_ids, result)}
    
        return recog_words 
    
    @staticmethod
    def image_for_extraction(raw_image):
        """
        Very critical step in the image processing, since it is also used in the preparation of training data 
        """
        gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)        
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)   
        return thresh

    @staticmethod
    def get_word_images_from_textract(response, img_file):
        """      
        Parameters
        ----------
        response : string, path to the JSON representation of the Textract OCR            
        img_file : string, path to the image

        Returns
        -------
        a dictionary,  having {word_id: word_image, ...}

        """
                   
        img = cv2.imread(img_file)       
        img_thresh = Predict.image_for_extraction(img)
        # remove lines not good for highlighted words
        # img_thresh = ImageUtils.remove_lines(img_thresh, horizontalsize=120, verticalsize=100 )  #remove lines not good for highlighted words
        img_h, img_w = img_thresh.shape[:2]
                     
        with open(response, 'r') as r:
           textract_ocr = json.load(r)  
           
        document = TextractUtils(textract_ocr)
        word_map, line_map = document.TextractParser()
        Text = document.GetText()   
        h_w_ratio = [v['height']/v['width'] for k, v in word_map.items()]    
        if np.mean(h_w_ratio) > 1:     # rotated images
           raise Exception('Rotated images are not supported')
           
        else:
            word_imgs = {}
            for k, v in line_map.items() :
                bbox = [v['left'],  v['top'],  v['right'],  v['bottom']] 
                l, t, r, b = ImageUtils.reverseXY(img_h, img_w, bbox)
                line_img = img_thresh[t:b, l:r ] 
                line_img = ImageUtils.crop_image(line_img, axis=2)
                height_in_line = line_img.shape[0]                    
                ids = v['ids']   
                # ids = sorted(ids, key = lambda x: word_map[x]['left'])
                for i in ids:         
                    word_obj = word_map[i]    
                    bbox0 = [word_obj['left'],  word_obj['top'],  word_obj['right'],  word_obj['bottom']] 
                    l0, t0, r0, b0 = ImageUtils.reverseXY(img_h, img_w, bbox0)
                    word_img = img_thresh[t0:b0, l0:r0]
                    word_img = ImageUtils.crop_image(word_img, 2)  
                    size = word_img.shape[:2]                            
                    half_h = max(0, int((height_in_line - size[0])/2))
                    word_img = np.pad(word_img, ((half_h, half_h), (0, 0)), 'constant', constant_values=0)
                    word_imgs[i] = word_img 
                                    
            return Text, word_imgs