import argparse
import torch.nn as nn

parser = argparse.ArgumentParser(description='IKM')

# Hardware specifications
parser.add_argument("--cuda", default=False, action="store_true", help="Use cuda?")
parser.add_argument('--n_GPUs', type=int, default=1, help='parallel training with multiple GPUs')
parser.add_argument('--GPU_ID', type=int, default=0, help='parallel training with multiple GPUs')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loading, ==Num. CPU Cores')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# data specifications
parser.add_argument('--dir_data', type=str, default='../../Datasets/', help='dataset directory')
parser.add_argument('--ext', type=str, default='img', help='dataset file extension')
parser.add_argument('--data_train', type=str, default='DF2K', help='train dataset name')
#parser.add_argument('--data_valid', type=str, default=['Set5', 'BSD100'], help='validation/test dataset')
parser.add_argument('--data_valid', type=str, default=['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109'], help='validation/test dataset')
parser.add_argument('--n_train', type=int, default=3450, help='number of training set')
parser.add_argument('--model_path', type=str, default='', help='path to save model')
parser.add_argument('--scale', type=int, default=2, help='scale factor')
parser.add_argument('--patch_size', type=int, default=48, help='input patch size')
parser.add_argument('--value_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')

# Model specifications:
parser.add_argument('--net_type', type=str, default='UHDN', help='type of backbone: ')
parser.add_argument('--n_blocks', type=int, default=4, help='load the model from the specified epoch')
parser.add_argument('--n_units', type=int, default=6, help='load the model from the specified epoch')
parser.add_argument('--use_Att', default='IKM', help='load the model from the specified epoch')
parser.add_argument('--n_channels', type=int, default=64, help='number of feature maps')
parser.add_argument('--growth_rate', type=int, default=12, help='number of feature maps')
parser.add_argument('--groups', type=int, default=64//4, help='number of feature maps')
parser.add_argument('--act', default=nn.ReLU(inplace=True), help='residual scaling')

# Training specifications
parser.add_argument("--train", default='Test', action="store_true", help="True for training, False for testing")
parser.add_argument('--iter_epoch', type=int, default=2000, help='iteration in each epoch')
parser.add_argument("--start_epoch", default=0, type=int, help="start epoch for training AGSR")
parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs to train')
parser.add_argument('--best_epoch', type=int, default=150, help='best epoch from validation PSNR')
parser.add_argument('--resume', type=str, default='', help='load the model from the specified epoch')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--lr_gamma', type=int, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--lr_step_size', type=int, default=50, help='learning rate decay per N epochs')

args = parser.parse_args()
