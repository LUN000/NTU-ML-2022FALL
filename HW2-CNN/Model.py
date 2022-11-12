import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
import numpy

class P1Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__= "P1Cnn"
        self.conv1 = nn.Conv2d(1, 9, (1,1), dilation=(1, 1), stride=(1, 1)) # --> h = w = [(64-1)/1]+1 = 64. c = 9
        self.b1 = nn.BatchNorm2d(9)
        self.conv2 = nn.Conv2d(9, 16, (2,2), stride=(2, 2)) # --> h = w = [(64-1-1)/2]+1 = 32, c = 16
        self.pool1 = nn.MaxPool2d(2,2) # --> h = w = [(32-1-1)/2]+1 = 16, c = 16
        self.conv3 = nn.Conv2d(16, 16, (2,2)) # --> h = w = [(16-1-1)/1]+1 = 15, c = 16
        self.pool2 = nn.MaxPool2d(3,1) # --> h = w = [(15-2-1)/1]+1 = 13, c = 16
        self.conv4 = nn.Conv2d(16, 16, (2,2)) # --> h = w = [(13-1-1)/1]+1 = 12, c = 16
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 7)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        p1 = self.pool1(c2)
        c3 = F.relu(self.conv3(p1))
        p2 = self.pool2(c3)
        c4 = self.conv4(p2)
        flat = torch.flatten(c4, 1)
        f1 = F.relu(self.fc1(flat))
        f2 = F.relu(self.fc2(f1))
        f3 = F.relu(self.fc3(f2))
        out = F.relu(self.fc4(f3))
        # out = F.relu(self.fc4(f3))
        return out

class P1Cnn_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__= "P1Cnn_b"
        self.conv1 = nn.Conv2d(1, 8, (1,1), dilation=(1, 1), stride=(1, 1)) # --> h = w = [(64-1)/1]+1 = 32. c = 8
        self.conv2 = nn.Conv2d(8, 16, (2,2), stride=(1, 1)) # --> h = w = [(64-1-1)/1]+1 = 63, c = 16
        self.b1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2,1) # --> h = w = [(63-1-1)/1]+1 = 62, c = 16

        self.conv3 = nn.Conv2d(16, 32, (2,2)) # --> h = w = [(62-1-1)/1]+1 = 61, c = 32
        self.b2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2,1) # --> h = w = [(61-1-1)/1]+1 = 60, c = 32

        self.conv4 = nn.Conv2d(32, 16, (2,2), stride=(2, 2)) # --> h = w = [(60-1-1)/2]+1 = 30, c = 16
        self.b3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2,1) # --> h = w = [(30-1-1)/1]+1 = 29, c = 16

        self.conv5 = nn.Conv2d(16, 8, (2,2)) # --> h = w = [(29-1-1)/1]+1 = 28, c = 8
        self.b4 = nn.BatchNorm2d(8)
        self.pool4 = nn.MaxPool2d(2,2) # --> h = w = [(28-1-1)/2]+1 = 14, c = 8

        self.conv6 = nn.Conv2d(8, 6, (2,2), stride=(2, 2)) # --> h = w = [(14-1-1)/2]+1 = 7, c = 6

        self.fc = nn.Linear(294, 7)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.b1(self.conv2(c1)))
        p1 = self.pool1(c2)
        c3 = F.relu(self.b2(self.conv3(p1)))
        p2 = self.pool2(c3)
        c4 = F.relu(self.b3(self.conv4(p2)))
        p3 = self.pool3(c4)
        c5 = F.relu(self.b4(self.conv5(p3)))
        p4 = self.pool4(c5)
        c6 = self.conv6(p4)

        flat = torch.flatten(c6, 1)
        out = F.relu(self.fc(flat))
        return out

class P1Cnn_b_gap(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__= "P1Cnn_b_gap"
        self.conv1 = nn.Conv2d(1, 8, (1,1), dilation=(1, 1), stride=(1, 1)) # --> h = w = [(64-1)/1]+1 = 32. c = 8
        self.conv2 = nn.Conv2d(8, 16, (2,2), stride=(1, 1)) # --> h = w = [(64-1-1)/1]+1 = 63, c = 16
        self.b1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2,1) # --> h = w = [(63-1-1)/1]+1 = 62, c = 16

        self.conv3 = nn.Conv2d(16, 32, (2,2)) # --> h = w = [(62-1-1)/1]+1 = 61, c = 32
        self.b2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2,1) # --> h = w = [(61-1-1)/1]+1 = 60, c = 32

        self.conv4 = nn.Conv2d(32, 16, (2,2), stride=(2, 2)) # --> h = w = [(60-1-1)/2]+1 = 30, c = 16
        self.b3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2,1) # --> h = w = [(30-1-1)/1]+1 = 29, c = 16

        self.conv5 = nn.Conv2d(16, 8, (2,2)) # --> h = w = [(29-1-1)/1]+1 = 28, c = 8
        self.b4 = nn.BatchNorm2d(8)
        self.pool4 = nn.MaxPool2d(2,2) # --> h = w = [(28-1-1)/2]+1 = 14, c = 8

        self.conv6 = nn.Conv2d(8, 7, (2,2), stride=(2, 2)) # --> h = w = [(14-1-1)/2]+1 = 7, c = 7

        self.gap = nn.AvgPool2d(7, stride=1)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.b1(self.conv2(c1)))
        p1 = self.pool1(c2)
        c3 = F.relu(self.b2(self.conv3(p1)))
        p2 = self.pool2(c3)
        c4 = F.relu(self.b3(self.conv4(p2)))
        p3 = self.pool3(c4)
        c5 = F.relu(self.b4(self.conv5(p3)))
        p4 = self.pool4(c5)
        c6 = self.conv6(p4)
        out = torch.flatten(self.gap(c6),1)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = "resnet18"
        self.conv1 = nn.Conv2d(1, 3, (1,1), dilation=(1, 1), stride=(1, 1))
        self.res18 = models.resnet18(pretrained=False)
        self.res18.fc = nn.Linear(512, 7)

    def forward(self, x):
        x = self.conv1(x)
        out = self.res18(x)
        return out

class ResNet152(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = "resnet152"
        self.conv1 = nn.Conv2d(1, 3, (1,1), dilation=(1, 1), stride=(1, 1))
        self.res152 = models.resnet152(pretrained=False)
        self.res152.fc = nn.Linear(2048, 7)
    def forward(self, x):
        x = self.conv1(x)
        out = self.res152(x)
        return out
