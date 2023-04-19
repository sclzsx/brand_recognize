from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from pathlib import Path
import json
from cfg import device, pos_dict, model, opt, num_classes
import torch
from model import CLS_MODEL
from train import dataset

def evaluate(opt, test_pt_name):
    net = CLS_MODEL(opt.backbone, num_classes, opt.pretrained).to(device)

    checkpoint = torch.load(opt.save_dir + '/' + test_pt_name + '.pt')
    net.load_state_dict(checkpoint['net'])

    test_dataset = dataset(opt.data_root + '/test', opt.img_size)

    net.eval()

    labels, preds, pred_scores = [], [], []
    cnt = 0
    for Data in tqdm(test_dataset):
        img = Data[0].unsqueeze(0).to(device)
        label = Data[1].item()
        with torch.no_grad():
            pred = net(img)
        softmax = F.softmax(pred, dim=1)
        score = torch.max(softmax, 1)[0].squeeze().item()
        pred_label = torch.max(softmax, 1)[1].squeeze().item()
        labels.append(label)
        preds.append(pred_label)
        pred_scores.append(score)
        # print(label, pred_label, score)

        # img = (np.array(img.cpu().squeeze().permute(1, 2, 0)) * 255).astype('uint8')

    if num_classes > 2:
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
        metrics = {'precision': precision, 'recall': recall, 'f1_score': f1}
    else:
        acc = accuracy_score(labels, preds)
        p = precision_score(labels, preds, average='binary')
        r = recall_score(labels, preds, average='binary')
        f1 = f1_score(labels, preds, average='binary')
        conf = confusion_matrix(labels, preds).tolist()
        metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1, 'confusion_matric': conf}

        fpr, tpr, thresholds = roc_curve(labels, pred_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(opt.save_dir + '/test_roc_auc.jpg')

    for item in metrics.items():
        print(item)

    with open(opt.save_dir + '/test_metrics_' + test_pt_name + '.json', 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    evaluate(opt, test_pt_name='max_val_f1')