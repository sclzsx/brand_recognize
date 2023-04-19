import argparse
import torch
from pathlib import Path
import os


if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'

pos_dict = {
    'adidas_regular': 1,
    'apple_regular': 2,
    'BWM_regular': 3,
    'nike_regular': 4,
    'Pepsi_regular':5,
    'wechat_regular':6,
}

num_classes = len(pos_dict) + 1

print(device)
print(pos_dict)
print(num_classes)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument('--backbone', type=str, default='vgg16')
parser.add_argument("--end_epoch", type=int, default=100)
parser.add_argument('--resume_path', type=str, default='')
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--pretrained", type=bool, default=False)
parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='results/v0')
opt = parser.parse_args()


