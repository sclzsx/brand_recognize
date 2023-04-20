from pathlib import Path
import cv2
import os
import random
import shutil
from tqdm import tqdm
import numpy as np
import json


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def save_video_frames(video_root, dataset_root, capture_step, img_size):
    for video_path in Path(video_root).rglob('*.*'):
        print('capturing:', video_path.name)

        name = video_path.name.split('.')[0]
        save_dir = dataset_root + '/' + name
        mkdir(save_dir)

        videoCapture = cv2.VideoCapture(str(video_path))
        fNUMS = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(0, fNUMS, capture_step):
            videoCapture.set(cv2.CAP_PROP_POS_FRAMES, i) 
            success, frame = videoCapture.read()
            if not success:
                break

            h, w, _ = frame.shape
            if h < w:
                frame = frame[:, w//2-h//2:w//2+h//2, :]
            else:
                frame = frame[h//2-w//2:h//2+w//2, :, :]
            frame = cv2.resize(frame, (img_size, img_size))

            cv2.imwrite(save_dir + '/' + str(i) + '.jpg', frame)

        videoCapture.release()

def split_train_test(dataset_root, val_rate, test_rate):
    image_dirs = [i for i in Path(dataset_root).iterdir() if i.is_dir()]
    for image_dir in image_dirs:
        print('splitting:', image_dir.name)

        image_paths = [i for i in Path(image_dir).glob('*.jpg')]
        random.shuffle(image_paths)
        val_num = int(len(image_paths) * val_rate)
        test_num = int(len(image_paths) * test_rate)

        for _ in range(test_num):
            image_path = image_paths.pop()
            save_dir = dataset_root + '/test/' + image_path.parent.name
            mkdir(save_dir)
            shutil.move(str(image_path), save_dir)

        for _ in range(val_num):
            image_path = image_paths.pop()
            save_dir = dataset_root + '/val/' + image_path.parent.name
            mkdir(save_dir)
            shutil.move(str(image_path), save_dir)

        for _ in range(len(image_paths)):
            image_path = image_paths.pop()
            save_dir = dataset_root + '/train/' + image_path.parent.name
            mkdir(save_dir)
            shutil.move(str(image_path), save_dir)

    for dir_name in os.listdir(dataset_root):
        if dir_name not in ['train', 'val', 'test']:
            shutil.rmtree(dataset_root + '/' + dir_name)

def cal_trainset_info(dataset_root):
    train_dir = dataset_root + '/train'
    
    print('cal label_dict:')
    label_dict = {}
    label_val = 1
    for label_name in os.listdir(train_dir):
        if '_fake' not in label_name:
            label_dict.setdefault(label_name, label_val)
            label_val += 1
    print(label_dict)

    print('cal count_dict:')
    count_dict = {}
    for label_name, label_val in label_dict.items():
        label_cnt = len(os.listdir(train_dir + '/' + label_name))
        count_dict.setdefault(label_val, label_cnt)
    image_paths = [i for i in Path(train_dir).rglob('*.jpg')]
    negative_paths = [i for i in image_paths if i.parent.name not in label_dict]
    count_dict.setdefault(0, len(negative_paths))
    print(count_dict)

    print('cal ms:')
    ms_dict = {}
    mean_r, mean_g, mean_b = 0, 0, 0
    std_r, std_g, std_b = 0, 0, 0
    N = len(image_paths)
    for image_path in tqdm(image_paths):
        image = cv2.imread(str(image_path))
        image = image / 255
        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]
        mean_b = mean_b + np.mean(b)
        mean_g = mean_g + np.mean(g)
        mean_r = mean_r + np.mean(r)
        std_b = std_b + np.std(b)
        std_g = std_g + np.std(g)
        std_r = std_r + np.std(r)
    mean_b = mean_b / N
    mean_g = mean_g / N
    mean_r = mean_r / N
    std_b = std_b / N
    std_g = std_g / N
    std_r = std_r / N
    ms_dict.setdefault('mean_bgr', [mean_b, mean_g, mean_r])
    ms_dict.setdefault('std_bgr', [std_b, std_g, std_r])
    print(ms_dict)

    trainset_info = [label_dict, count_dict, ms_dict]
    with open(dataset_root + '/trainset_info.json', 'w') as f:
        json.dump(trainset_info, f, indent=2)


if __name__ == '__main__':
    video_root = 'data/Camera Roll' # 视频文件夹。按要求命名。
    dataset_root = 'data/brand_dataset' # 生成数据集路径
    capture_step = 10 # 采帧间隔
    image_size = 224 # 图像尺度
    val_rate = 0.2 # 验证集比例
    test_rate = 0.2 # 测试集比例

    if os.path.exists(dataset_root):
        shutil.rmtree(dataset_root)

    save_video_frames(video_root, dataset_root, capture_step, image_size) # 保存视频帧
    split_train_test(dataset_root, val_rate, test_rate) # 划分数据集
    cal_trainset_info(dataset_root) # 统计训练集的均值、方差、数目信息