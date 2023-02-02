import argparse, torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='CoMo')

# Hardware specifications
parser.add_argument("--cuda", default=True, action='store_true', help='Use cuda?')
parser.add_argument('--n_GPUs', type=int, default=1, help='parallel training with multiple GPUs')
parser.add_argument('--GPU_ID', type=int, default=0, help='parallel training with multiple GPUs')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loading, ==Num. CPU Cores')
parser.add_argument('--seed', type=int, default=1024, help='random seed')

# data specifications
parser.add_argument('--dir_data', type=str, default='../../Datasets/', help='dataset directory')
parser.add_argument('--ext', type=str, default='img', help='dataset file extension')
parser.add_argument('--data_train_hr', type=str, default=['Train/DIV2K/HR'], help='train dataset name')
parser.add_argument('--data_train_lr', type=str, default=['Train/DIV2K/LR_bicubic/X3'], help='train dataset name')
#parser.add_argument('--data_valid', type=str, default=['Set5', 'BSD100'], help='validation/test dataset')
parser.add_argument('--data_valid', type=str, default=['Set14'], help='validation/test dataset')
parser.add_argument('--n_train', type=int, default=[800], help='number of training set')
parser.add_argument('--shuffle', type=bool, default=False, help='number of training set')
parser.add_argument('--store_in_ram', type=bool, default=True, help='number of training set')
parser.add_argument('--model_path', type=str, default='', help='path to save model')
parser.add_argument('--scale', type=int, default=2, help='scale factor')
parser.add_argument('--patch_size', type=int, default=48, help='input patch size')
parser.add_argument('--value_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')

# Model specifications:
parser.add_argument('--net_type', type=str, default='UHDN', help='type of backbone: ')

# parser.add_argument('--n_blocks', type=int, default=4, help='load the model from the specified epoch')
# parser.add_argument('--n_units', type=int, default=6, help='load the model from the specified epoch')
# # parser.add_argument('--n_layers', type=list, default=[8, 6, 4, 4, 6, 8], help='load the model from the specified epoch')
# parser.add_argument('--n_layers', type=list, default=[6, 5, 4, 4, 5, 6], help='load the model from the specified epoch')
# parser.add_argument('--use_CoMo', default=True, help='load the model from the specified epoch')
# parser.add_argument('--n_channels', type=int, default=48, help='number of feature maps')
# parser.add_argument('--growth_rate', type=int, default=8, help='number of feature maps')

parser.add_argument('--n_blocks', type=int, default=8, help='load the model from the specified epoch')
parser.add_argument('--n_units', type=int, default=6, help='load the model from the specified epoch')
# parser.add_argument('--n_layers', type=list, default=[8, 6, 4, 4, 6, 8], help='load the model from the specified epoch')
parser.add_argument('--n_layers', type=list, default=[6, 5, 4, 4, 5, 6], help='load the model from the specified epoch')
parser.add_argument('--use_CoMo', default=True, help='load the model from the specified epoch')
parser.add_argument('--n_channels', type=int, default=128, help='number of feature maps')
parser.add_argument('--growth_rate', type=int, default=32, help='number of feature maps')

parser.add_argument('--act', default=nn.LeakyReLU(0.1, True), help='residual scaling')

# Training specifications
parser.add_argument('--train', default='train', action='store_true', help='True for training, False for testing')
parser.add_argument('--iter_epoch', type=int, default=2000, help='iteration in each epoch')
parser.add_argument('--start_epoch', default=-1, type=int, help='start epoch for training AGSR')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--best_epoch', type=int, default=150, help='best epoch from validation PSNR')
parser.add_argument('--resume', type=str, default='', help='load the model from the specified epoch')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')

# Optimization specifications
parser.add_argument('--optimizer', default={'SR': torch.optim.Adam}, help='optimizers')
parser.add_argument('--lr', type=float, default={'SR': 2e-4}, help='learning rate')
parser.add_argument('--lr_gamma_1', type=float, default={'SR': 50}, help='learning rate decay factor for step decay')
parser.add_argument('--lr_gamma_2', type=float, default={'SR': 1e-5}, help='learning rate decay per N epochs')

args = parser.parse_args()

