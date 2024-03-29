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


class TrainingUtils:

    @staticmethod
    def load_word_imgs(train_data_path):
        all_x = []
        all_y = []
        files = os.listdir(train_data_path)
        for f in files:
            f = gzip.open(train_data_path + '/' + f, 'rb')
            data = pickle.load(f, encoding='latin1')
            all_x += [e['bw_img'] for e in data]
            all_y += [e['Text'] for e in data]

        all_x, all_y = shuffle(all_x, all_y)
        return all_x, all_y

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
                print(layer_name, layer['config']['filters'], layer['config']['kernel_size'],
                      layer['config']['strides'])
            elif layer_name == 'MaxPooling2D':
                print(layer_name, layer['config']['pool_size'], layer['config']['strides'])
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
        """might be good to pad 2 pickles in both horizontal ends. need to try"""
        new_img = np.zeros(down_size)
        img_h, img_w = img.shape[:2]
        img_d_h, img_d_w = down_size[0] - 2 * padding, down_size[1] - 2 * padding
        scale_h = min(img_d_w / img_w, img_d_h / img_h)
        img = cv2.resize(img, (0, 0), fx=scale_h, fy=scale_h)
        img_h, img_w = img.shape[:2]
        dif_h = (down_size[0] - img_h) // 2
        new_img[dif_h:dif_h + img_h, padding:padding + img_w] = img
        return new_img

    @staticmethod
    def img_gen_train(train_x, train_y, batch_size, bn_shape):
        width, height, channel = input_shape
        x = np.zeros((batch_size, width, height, channel), dtype=np.float32)
        yy = np.zeros((batch_size, max_string_len), dtype=np.uint8)

        while True:
            for ii in range(batch_size):
                while True:  # abandon the lexicon which is longer than 20 characters
                    pick_index = np.random.randint(0, len(train_y) - 1)
                    img = train_x[pick_index]
                    lexicon = train_y[pick_index]
                    if (img is not None) and len(lexicon) <= max_string_len and np.max(img) > 0:
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

            yield [x, yy, np.ones(batch_size) * int(bn_shape[1] - 2), np.ones(batch_size) * max_string_len], yy

    @staticmethod
    def img_gen_val(val_x, val_y, cnn_input_shape, batch_size):
        width, height, channel = cnn_input_shape
        x = np.zeros((batch_size, width, height, 1), dtype=np.float32)
        yy = []

        while True:
            for ii in range(batch_size):
                while True:  # abandon the lexicon which is longer than 16 characters
                    pick_index = np.random.randint(0, len(val_y) - 1)
                    img = val_x[pick_index]
                    lexicon = val_y[pick_index]
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
    def evaluate(pred_model, val_x, val_y, cnn_input_shape, val_batch_size):
        correct_prediction = 0
        generator = TrainingUtils.img_gen_val(val_x, val_y, cnn_input_shape, val_batch_size)
        x_test, y_test = next(generator)
        y_pred = pred_model.predict(x_test)
        shape = y_pred[:, 2:, :].shape
        ctc_decode = bknd.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
        out = bknd.get_value(ctc_decode)[:, :max_string_len]

        for m in range(val_batch_size):
            result_str = ''.join([characters[k] for k in out[m]])
            result_str = result_str.replace(' ', '')
            original_str = y_test[m].replace(' ', '')
            if result_str == original_str:
                correct_prediction += 1
            else:
                print(original_str + '\t' + '\t' + result_str)

        return correct_prediction * 100.0 / val_batch_size
