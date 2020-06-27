from sklearn.utils import shuffle
from unidecode import unidecode  
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, \
                                    Reshape, LSTM, Lambda, add, concatenate

from src.config import *
from src.train_util import TrainingUtils
from src.evaluate import EvaluateCallback


def train():
    
    width, height, channel = input_shape
    num_classes = len(characters)    
    
    all_X, all_Y = TrainingUtils.load_word_imgs(train_data_path)
    all_X, all_Y = shuffle(all_X, all_Y)
    all_Y = [unidecode(y) for y in all_Y]   
    train_x, val_x, train_y,  val_y = train_test_split(all_X, all_Y, test_size=0.2, random_state=2020)

    # build model     
    # Network parameters
    cnn_filters = [64, 128, 256, 512, 512]
    kernel = (3, 3)
    pool_size_1 = (2, 2)
    pool_size_2 = (1, 2)
    pool_size_3 = (1, 3)
    lstm_size = 512
    time_dense_size = 512
    dropout_rate = 0.25
   
    # since the original dimension is (width, height), the pool_size must be (horizonal, vertical)
    inputShape = Input((width, height, channel))  # base on Tensorflow backend
    
    conv_1 = Conv2D(cnn_filters[0],  kernel, activation='relu', padding='same')(inputShape)  # was (3,3)
    conv_1 = Conv2D(cnn_filters[0],  kernel, activation='relu', padding='same')(conv_1)      # was (3,3)
    batchnorm_1 = BatchNormalization()(conv_1)
    pool_1 = MaxPooling2D(pool_size=pool_size_1)(batchnorm_1)
           
    conv_2 = Conv2D(cnn_filters[1], kernel, activation='relu', padding='same')(pool_1)     # was (3,3)
    conv_2 = Conv2D(cnn_filters[1], kernel, activation='relu', padding='same')(conv_2)     # was (3,3)
    batchnorm_2 = BatchNormalization()(conv_2)
    pool_2 = MaxPooling2D(pool_size=pool_size_2)(batchnorm_2)
    
    conv_3 = Conv2D(cnn_filters[2], kernel, activation='relu', padding='same')(pool_2)
    conv_3 = Conv2D(cnn_filters[2], kernel, activation='relu', padding='same')(conv_3)
    batchnorm_3 = BatchNormalization()(conv_3)
    pool_3 = MaxPooling2D(pool_size=pool_size_2)(batchnorm_3)
    
    conv_4 = Conv2D(cnn_filters[3], kernel, activation='relu', padding='same')(pool_3)
    conv_4 = Conv2D(cnn_filters[3], kernel, activation='relu', padding='same')(conv_4)
    batchnorm_4 = BatchNormalization()(conv_4)
    pool_4 = MaxPooling2D(pool_size=pool_size_3)(batchnorm_4)
    
    conv_5 = Conv2D(cnn_filters[4], kernel , activation='relu', padding='same')(pool_4)  
    batchnorm_5 = BatchNormalization()(conv_5)
     
    # Vectorize
    bn_shape = batchnorm_5.get_shape()
    x_reshape = Reshape(target_shape=(int(bn_shape[1]), int(bn_shape[2] * bn_shape[3])))(batchnorm_5)
    fc_1 = Dense(time_dense_size, activation='relu')(x_reshape) 
    
    # assert 
    print(x_reshape.get_shape())  
    print(fc_1.get_shape()) 
    
    # Two layers of bidirectional LSTM
    rnn_1 = LSTM(lstm_size, kernel_initializer="he_normal", return_sequences=True)(fc_1)
    rnn_1b = LSTM(lstm_size, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(fc_1)
    rnn1_merged = add([rnn_1, rnn_1b])
    
    rnn_2 = LSTM(lstm_size, kernel_initializer="he_normal", return_sequences=True)(rnn1_merged)
    rnn_2b = LSTM(lstm_size, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(rnn1_merged)
    rnn2_merged = concatenate([rnn_2, rnn_2b])
    
    drop_1 = Dropout(dropout_rate)(rnn2_merged)
    
    fc_2 = Dense(num_classes, kernel_initializer='he_normal', activation='softmax')(drop_1)
    
    # model setting
    base_model = Model(inputs=inputShape, outputs=fc_2)  # the model for predicting
    labels = Input(name='the_labels', shape=[max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(TrainingUtils.ctc_lambda_func, output_shape=(1,), name='ctc')([fc_2, labels, input_length, label_length])    
    model = Model(inputs=[inputShape, labels, input_length, label_length], outputs=[loss_out])  # the model for training
    
    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.0003, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)   # adam = optimizers.Adam()
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.summary()  
            
    checkpoint = ModelCheckpoint(cp_save_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    evaluator = EvaluateCallback(base_model, val_x, val_y, input_shape, val_batch_size)   # Evaluate()
    
    if pre_trained_model:        
        model.load_weights(pre_trained_model)     
       
    model.fit(TrainingUtils.img_gen_train(train_x, train_y, train_batch_size, bn_shape), steps_per_epoch=1000,
              epochs=1, verbose=1, workers=1, callbacks=[evaluator, checkpoint])
    
    base_model.save(model_save_path + '/' + 'tf2_word_recognition.hdf5')
    model_json = base_model.to_json()
    with open(model_save_path + '/' + 'tf2_word_recognition.json', "w") as json_file:
        json_file.write(model_json)  

    return 


if __name__ == '__main__':
    train()
