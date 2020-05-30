import cv2
import json
import numpy as np
import os
import gzip
import pickle
from sklearn.utils import shuffle
# import matplotlib.gridspec as gridspec

from tensorflow.keras import backend as bknd

from src.config import characters, max_string_len, input_shape
# from src.evaluate import EvaluateCallback


class TrainingUtils:
    
    @staticmethod     
    def load_word_imgs(train_data_path):
        all_X = []
        all_Y  = []
        files = os.listdir(train_data_path)
        for f in files:
            f = gzip.open(train_data_path + '/' + f, 'rb')
            data = pickle.load(f, encoding='latin1')
            all_X += [e['bw_img'] for e in data ] 
            all_Y += [e['Text'] for e in data ] 
            
        all_X, all_Y = shuffle (all_X, all_Y)    
        return all_X, all_Y

    @staticmethod 
    def read_cnn_layers(model_json):
        with open(model_json) as f:
            f = json.load(f)
    
        layers = f['config']['layers']
        for layer in layers:
            layer_name = layer['class_name']
            if layer_name == 'InputLayer':
                print(layer_name, layer['config']['batch_input_shape'])
            elif layer_name == 'Conv2D':
                print(layer_name, layer['config']['filters'], layer['config']['kernel_size'], layer['config']['strides'])
            elif layer_name == 'MaxPooling2D':
                print(layer_name, layer['config']['pool_size'], layer['config']['strides'] )
            elif layer_name == 'Dense':
                print(layer_name, layer['config']['units'], layer['config']['activation'])
            elif layer_name == 'LSTM':
                print(layer_name, layer['config']['units'])
            elif layer_name == 'Dropout':
                print(layer_name, layer['config']['rate'])
            else:
                print(layer_name)
        
        return

    @staticmethod 
    def norm_img(img, down_size, padding):    
        new_img = np.zeros(down_size) 
        img_h, img_w = img.shape[:2]
        img_H, img_W = down_size[0] - 2*padding, down_size[1] - 2*padding
        scale_h = min(img_W/img_w, img_H/img_h )         
        img = cv2.resize(img, (0, 0), fx=scale_h, fy=scale_h)
        img_h, img_w = img.shape[:2]    
        dif_h = (down_size[0] - img_h)//2         
        new_img[dif_h:dif_h + img_h, padding:padding + img_w] = img           
        return new_img
  
    @staticmethod
    def img_gen_train(train_x, train_y, batch_size, bn_shape):
        width, height, channel = input_shape
        x = np.zeros((batch_size, width, height, channel), dtype=np.float32)
        yy= np.zeros((batch_size, max_string_len), dtype=np.uint8)
       
        while True:
            for ii in range(batch_size):           
                while True:  # abandon the lexicon which is longer than 16 characters
                    pick_index = np.random.randint(0, len(train_y) - 1)
                    img = train_x[pick_index]
                    lexicon = train_y[pick_index]
                    if (img is not None) and len(lexicon) <= max_string_len  and np.max(img) > 0:
                        img_size = img.shape  
                        if img_size[1] > 2 and img_size[0] > 2:
                            break                         
                                   
                img = TrainingUtils.norm_img(img, (height, width), padding=2)
                img = img.astype(np.float32)
                img /= np.max(img)
                while len(lexicon) < max_string_len:
                    lexicon += " "
                img = np.expand_dims(img.T, -1)
                x[ii] = img
                yy[ii] = [characters.find(c) for c in lexicon]                
                
            # inputs = {'the_input': x,
            #   'the_labels': yy,
            #   'input_length': np.ones(batch_size) * int(input_shape[1] - 2),
            #   'label_length': np.ones(batch_size) * max_string_len,
            #   'source_str': source_str  # used for visualization only
            #   }
                
            # outputs = {'ctc': np.zeros([batch_size])}  # dummy data for dummy loss function
            
            yield [x, yy, np.ones(batch_size) * int(bn_shape[1] - 2), np.ones(batch_size) * max_string_len ], yy
            
    
    @staticmethod
    def img_gen_val(val_X, val_Y, input_shape, batch_size):
        width, height, channel = input_shape
        x = np.zeros((batch_size, width, height, 1), dtype=np.float32)      
        yy = []
        
        while True:
            for ii in range(batch_size):           
                while True:  # abandon the lexicon which is longer than 16 characters
                    pick_index = np.random.randint(0, len(val_Y) - 1)
                    img = val_X[pick_index]
                    lexicon = val_Y[pick_index]
                    if (img is not None) and len(lexicon) <= max_string_len:
                        img_size = img.shape  
                        if img_size[1] > 2 and img_size[0] > 2:
                            break                         
                                  
                img = TrainingUtils.norm_img(img, (height, width), padding=2)
                img = img.astype(np.float32)
                img /= np.max(img)
                while len(lexicon) < max_string_len:
                    lexicon += " "           
                
                img = np.expand_dims(img.T, -1)
                x[ii] = img
                yy.append(lexicon) 
                
            yield x, yy     
        
    @staticmethod
    def ctc_lambda_func(args): 
        # the actual loss calc occurs here despite it not being an internal Keras loss function
        # the 2 is critical here since the first couple outputs of the RNN tend to be garbage
        iy_pred, ilabels, iinput_length, ilabel_length = args    
        iy_pred = iy_pred[:, 2:, :]  
        return bknd.ctc_batch_cost(ilabels, iy_pred, iinput_length, ilabel_length)
    
    @staticmethod
    def evaluate(pred_model,val_x, val_y, input_shape, val_batch_size):
        correct_prediction = 0
        generator = TrainingUtils.img_gen_val(val_x, val_y, input_shape, val_batch_size)
        x_test, y_test = next(generator)       
        y_pred = pred_model.predict(x_test) 
        shape = y_pred[:, 2:, :].shape 
        ctc_decode = bknd.ctc_decode(y_pred[:, 2:, :], input_length = np.ones(shape[0])*shape[1])[0][0]
        out = bknd.get_value(ctc_decode)[:, :max_string_len]
    
        for m in range(val_batch_size):
            result_str = ''.join([characters[k] for k in out[m]])
            result_str = result_str.replace(' ', '')
            orignal_str = y_test[m].replace(' ', '')
            if result_str == orignal_str:
                correct_prediction += 1                
            else:
                print(orignal_str + '\t' + '\t' + result_str)
    
        return correct_prediction*100.0/val_batch_size     
    
        
    # @staticmethod    
    # def ctc_visualization(batch_word_imgs, batch_words_out, ground_truth_words):
    #     """ 
    #     For a real OCR application, this should be beam search with a dictionary and language model. 
    #     For this example, best path is sufficient.
    #     """     
    #     pred_texts = PredictionUtils.decode_batch(batch_words_out)
    #     bs = batch_word_imgs.shape[0] 
    #     for i in range(bs):
    #         img = batch_word_imgs[i][:, :, 0].T   
    #         ImageUtils.show_image ( img, caption = '')
    #         fig = plt.figure(figsize=(20, 20))
    #         outer = gridspec.GridSpec(1, 1, wspace=10, hspace=0.1)  
        
    #     #     ax1 = plt.Subplot(fig, outer[0])
    #     #     fig.add_subplot(ax1)
    #     #     ax1.set_title('Input img')
    #     #     ax1.imshow(img, cmap='gray')
    #     #     ax1.set_xticks([])
    #     #     ax1.set_yticks([])
            
            
    #         ax2 = plt.Subplot(fig, outer[0])
    #         fig.add_subplot(ax2)
    #         print('Predicted: %s\nTrue: %s' % (pred_texts[i], ground_truth_words[i]))
        
    #         ax2.set_title('Activations')
    #         ax2.imshow(out[i].T, cmap='binary', interpolation='nearest')
    #         ax2.set_yticks(list(range(len(characters) + 1)))
    #         ax2.set_yticklabels(characters + ['blank'])
    #         ax2.grid(False)
        
    #         for h in np.arange(-0.5, len(characters) + 1 + 0.5, 1):
    #             ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)
        
    #     #     for x in np.arange(-0.5, out[0].shape[0] + 1 + 0.5, 1):
    #     #         ax2.axvline(x, linestyle='--', color='k')
                
    #         plt.show()
            