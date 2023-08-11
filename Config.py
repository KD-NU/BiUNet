# -*- coding: utf-8 -*-
import os
import torch
import time
#import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 667
os.environ['PYTHONHASHSEED'] = str(seed)
#torch.cuda.set_device(1)

weight_decay = 1e-4
n_channels = 3
n_labels = 1  # MoNuSeg & Covid19
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 100

pretrain = False
task_name = 'Covid19' # MoNuSeg
learning_rate = 3e-4  # MoNuSeg: 1e-3, Covid19: 3e-4
batch_size = 24  # MoNuSeg: 2, Covid19: 24

model_name = 'BiUNet'

train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset = './datasets/' + task_name + '/Val_Folder/'
test_dataset = './datasets/' + task_name + '/Test_Folder/'
task_dataset = './datasets/' + task_name + '/Train_Folder/'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'

test_session = "Test_session_07.28_00h31"