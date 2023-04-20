'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torchvision

class CLS_MODEL(nn.Module):
    def __init__(self, backbone, num_classes, pretrained):
        super().__init__()

        if backbone == 'alexnet':
            model = torchvision.models.alexnet(pretrained=pretrained)
            self.features = model.features[:-1]
            fea_c_num = 256
        elif backbone == 'vgg16':
            model = torchvision.models.vgg16(pretrained=pretrained)
            self.features = model.features[:-1]
            fea_c_num = 512
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
            fea_c_num = 256

        self.line = nn.Sequential(
            nn.Conv2d(fea_c_num, fea_c_num * 2, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(fea_c_num * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=fea_c_num * 2, out_features=fea_c_num, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=fea_c_num, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=num_classes, bias=True),
        )

    def forward(self, x):
        f = self.features(x)
        x = self.line(f)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = CLS_MODEL('vgg16', 4, False)
    x = torch.randn(8,3,224,224)
    y = net(x)
    print(y.size())