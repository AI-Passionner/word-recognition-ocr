from tensorflow.keras.callbacks import Callback

from src.train_util import TrainingUtils


class EvaluateCallback(Callback):  
    
    def __init__(self, pred_model, val_x, val_y, input_shape, val_batch_size):
        self.pred_model = pred_model
        self.input_shape = input_shape
        self.val_batch_size = val_batch_size
        self.val_x = val_x
        self.val_y = val_y           
    
    def on_epoch_end(self, epoch, logs=None):
        acc = TrainingUtils.evaluate(self.pred_model,self.val_x, self.val_y, self.input_shape, self.val_batch_size)
        print('')
        print('acc:'+str(acc)+"%")
