from torch.utils.data import Dataset
from pathlib import Path
import torch
import cv2


def preprocess_np(img, img_size, mean, std): # 图像预处理
    h, w, _  = img.shape
    if h != img_size or w != img_size:
        img = cv2.resize(img, (img_size, img_size))

    img = img.astype('float32')
    img = img / 255

    img = (img - mean) / std

    return img

class CLS_Dataset(Dataset):
    def __init__(self, data_root, label_dict, img_size, mean, std):
        self.img_paths = [i for i in Path(data_root).rglob('*.jpg')]
        self.img_size = img_size
        self.label_dict = label_dict
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_name = self.img_paths[idx].parent.name
        if img_name in self.label_dict:
            class_id = self.label_dict[img_name]
            label = torch.tensor(class_id, dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)

        img_path = str(self.img_paths[idx])
        img = cv2.imread(img_path)

        img = preprocess_np(img, self.img_size, self.mean, self.std)

        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img, label