from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import shutil
import json
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from cfg import device, opt, num_classes, train_dataset, val_dataset, label_weights, milestones
from model import CLS_MODEL

class MultiClassFocalLossWithAlpha(torch.nn.Module):
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
    

def train(opt):
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
        
    opt_dict = vars(opt)
    for k, v in opt_dict.items():
        print(k, ':', v)
    with open(opt.save_dir + '/train_opts.json', 'w') as f:
        json.dump(opt_dict, f, indent=2)

    log_dir = opt.save_dir + '/log'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    net = CLS_MODEL(opt.backbone, num_classes, opt.pretrained).to(device)
    # print(net)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = MultiClassFocalLossWithAlpha(alpha=label_weights, gamma=2, reduction='mean')

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

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

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    iter_n_per_epoch = len(train_dataloader)
    max_val_f1 = -1000
    min_val_loss = 1000
    for epoch in range(start_epoch + 1, opt.end_epoch + 1):
        # train
        net.train()
        avg_loss = []
        for i, batchData in enumerate(train_dataloader):
            img = batchData[0].to(device)
            label = batchData[1].to(device)
            optimizer.zero_grad()
            pred = net(img)
            loss = criterion(pred, label)
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            # if i % int(iter_n_per_epoch * 0.1) == 0:
            print('Epoch:{}, Iter:[{}/{}], loss:{}'.format(epoch, i + 1, iter_n_per_epoch, loss.item()))
        avg_loss = sum(avg_loss) / len(avg_loss)
        writer.add_scalar('train_loss', avg_loss, global_step=epoch)
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
            loss = criterion(pred, label)
            avg_loss.append(loss.item())
            label = label.cpu().tolist()
            softmax = F.softmax(pred.cpu(), dim=1)
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
