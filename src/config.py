import json
import string

# training configures
characters = string.digits + string.ascii_letters + string.punctuation + ' '
cp_save_path = './models/tf2_word_recognition_ccp_{epoch:03d}-{loss:.4f}.hdf5'  # save checkpoint path
tb_log_dir = './models/log'                                                     # TensorBoard save path, Optional
model_save_path = './training'
pre_trained_model = './best_model/tf2_word_recognition_loss_0.013.hdf5'
train_data_path = './data/train'
val_data_path = './data/dev'

train_batch_size = 100
val_batch_size = 1000
input_shape = (400, 32, 1)
max_string_len = 20 
steps_per_epoch = 1000
epochs = 20
verbose = 1
workers = 1
split_frac = 0.2

# spell check files
unigrams_frequency_file = './support_data/frequency_dictionary_en_82_765.txt'
bigrams_frequency_file = './support_data/frequency_bigramdictionary_en_243_342.txt'
ngram_dict_path = './support_data/inv_ngram_freq_dictionary.json'

# model files
model_json_file = './best_model/tf2_word_recognition_loss_0.013.json'
model_weight_file = './best_model/tf2_word_recognition_loss_0.013.hdf5'