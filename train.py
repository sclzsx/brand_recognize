from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from cfg import device, pos_dict, opt, num_classes
from model import CLS_MODEL


def preprocess(img, img_size):
    h, w, _ = img.shape
    crop_size = h if h < w else w
    img = img[:crop_size, :crop_size, :]
    img = cv2.resize(img, (img_size, img_size))
    minVal, maxVal = np.min(img), np.max(img)
    img = (img - minVal) / (maxVal - minVal)
    return img.astype('float32')

class dataset(Dataset):
    def __init__(self, data_root, img_size):
        self.img_paths = [i for i in Path(data_root).rglob('*.jpg')]
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_name = self.img_paths[idx].parent.name
        if img_name in pos_dict:
            class_id = pos_dict[img_name]
            label = torch.tensor(class_id, dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)

        img_path = str(self.img_paths[idx])
        img = cv2.imread(img_path)
        img = preprocess(img, self.img_size)

        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, label


def train(opt):
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
        
    opt_dict = vars(opt)
    for k, v in opt_dict.items():
        print(k, ':', v)
    with open(opt.save_dir + '/train_opts.json', 'w') as f:
        json.dump(opt_dict, f, indent=2)

    writer = SummaryWriter(opt.save_dir)

    net = CLS_MODEL(opt.backbone, num_classes, opt.pretrained).to(device)

    print(net)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

    milestones = [i * opt.end_epoch // 10 for i in range(7, 10)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    if os.path.exists(opt.resume_path):
        checkpoint = torch.load(opt.resume_path)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        print('resume from epoch', start_epoch)
    else:
        start_epoch = 0

    train_dataset = dataset(opt.data_root + '/train', opt.img_size)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    val_dataset = dataset(opt.data_root + '/val', opt.img_size)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    iter_n = len(train_dataloader)
    max_val_f1 = -1000
    min_val_loss = 1000
    for epoch in range(start_epoch + 1, opt.end_epoch + 1):
        # train
        net.train()
        for i, batchData in enumerate(train_dataloader):
            img = batchData[0].to(device)
            label = batchData[1].to(device)
            optimizer.zero_grad()
            pred = net(img)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            if i % int(iter_n * 0.1) == 0:
                print('Epoch:{}, Iter:[{}/{}], loss:{}'.format(epoch, i + 1, iter_n, loss.item()))
            writer.add_scalar('train_loss', loss.item(), global_step=epoch)
            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)

        # val
        net.eval()
        labels, preds = [], []
        avg_loss = []
        for batchData in val_dataloader:
            with torch.no_grad():
                img = batchData[0].to(device)
                label = batchData[1].to(device)
                pred = net(img)
            loss = criterion(pred_label, label)
            avg_loss.append(loss.item())
            label = label.cpu().tolist()
            softmax = F.softmax(pred_label.cpu(), dim=1)
            pred_label = torch.max(softmax, 1)[1].cpu().tolist()
            # print(label, pred_label)
            labels.extend(label)
            preds.extend(pred_label)
        avg_loss = sum(avg_loss) / len(avg_loss)
        writer.add_scalar('val_loss', avg_loss, global_step=epoch)

        if num_classes > 2:
            val_f1 = f1_score(labels, preds, average='weighted')
        else:
            val_f1 = f1_score(labels, preds, average='binary')
        writer.add_scalar('val_f1', val_f1, global_step=epoch)

        print('[eval] val_loss:{}, val_f1:{}'.format(avg_loss, val_f1))

        if avg_loss < min_val_loss:
            min_val_loss = avg_loss
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, opt.save_dir + '/min_val_loss.pt')
            print('-' * 50)
            print('Saved checkpoint as min_val_loss:', min_val_loss)
            print('-' * 50)

        if val_f1 > max_val_f1:
            max_val_f1 = val_f1
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, opt.save_dir + '/max_val_f1.pt')
            print('-' * 50)
            print('Saved checkpoint as max_val_f1:', max_val_f1)
            print('-' * 50)


if __name__ == '__main__':
    train(opt)
