import argparse
import torch
from pathlib import Path
import os
from dataset import CLS_Dataset
import cv2
from tqdm import tqdm
import numpy as np
import json

dataset_root = './data/brand_dataset'

if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'
print('device:\n', device)

with open(dataset_root + '/trainset_info.json', 'r') as f:
    trainset_info = json.load(f)

label_dict = trainset_info[0]
print('label_dict:\n', label_dict)

count_dist = trainset_info[1]
print('count_dist:\n', count_dist)

ms_dict = trainset_info[2]
mean = ms_dict['mean_bgr']
std = ms_dict['std_bgr']
print('ms_dict:\n', ms_dict)

num_classes = len(label_dict) + 1
print('num_classes:\n', num_classes)

label_counts = [count_dist[str(i)] for i in range(num_classes)]
max_count = max(label_counts)
label_weights = [max_count / count_dist[str(i)] for i in range(num_classes)]
print('label_weights:\n', label_weights)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument('--backbone', type=str, default='vgg16')
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--end_epoch", type=int, default=10)
parser.add_argument('--resume_path', type=str, default='')
parser.add_argument("--pretrained", type=bool, default=True)
parser.add_argument('--save_dir', type=str, default='results/v1')
opt = parser.parse_args()

milestones = [i * opt.end_epoch // 10 for i in range(4, 10, 2)]
print('milestones:\n', milestones)

train_dataset = CLS_Dataset(dataset_root + '/train', label_dict, opt.img_size, mean, std)
val_dataset = CLS_Dataset(dataset_root + '/val', label_dict, opt.img_size, mean, std)
test_dataset = CLS_Dataset(dataset_root + '/test', label_dict, opt.img_size, mean, std)


