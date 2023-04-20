import numpy as np
import cv2
import torch.nn.functional as F
import torch
from model import CLS_MODEL

backbone = 'vgg16'
img_size = 224
pt_path = 'results/v1/max_val_f1.pt'
device = 'cuda'
label_dict = {'wechat_regular': 1, 'apple_regular': 2, 'BMW_regular': 3, 'nike_regular': 4, 'adidas_regular': 5, 'Pepsi_regular': 6}
mean = [0.3492684340755209, 0.39777500688040696, 0.4017369643003754]
std = [0.2167788890170244, 0.19677341627743578, 0.19102047277476858]


def find_label_name(label, label_dict):
    neg_name = 'fake_or_unknown'
    for label_name, label_val in label_dict.items():
        if label == label_val:
            return label_name
    return neg_name

def demo():
    # 加载网络
    num_classes = len(label_dict) + 1
    net = CLS_MODEL(backbone, num_classes, False).to(device)
    checkpoint = torch.load(pt_path)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        
        if not ret:
            print('capture error')
            break
        
        # 预处理
        h, w, _  = img.shape
        if h != img_size or w != img_size:
            img = cv2.resize(img, (img_size, img_size))
        img = img.astype('float32')
        img = img / 255
        img = (img - mean) / std

        # 识别
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = net(img)
        softmax = F.softmax(pred, dim=1)
        score = torch.max(softmax, 1)[0].squeeze().item()
        pred_label = torch.max(softmax, 1)[1].squeeze().item()

        # 显示
        img = np.array(img.cpu().squeeze().permute(1, 2, 0))
        img = img * std + mean
        img = np.clip(img * 255, 0, 255).astype('uint8').copy()
        pred_name = find_label_name(pred_label, label_dict)
        cv2.putText(img, 'pred : %s' % pred_name, (0, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img, 'score : %.3f' % score, (0, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("capture", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    demo()
