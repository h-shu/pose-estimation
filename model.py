import torch.nn.functional as F
import torch.nn as nn

from config import cfg

class HourglassModel(nn.Module):
    def __init__(self):
        super(HourglassModel, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 3, 3, padding="same")
        self.conv2 = nn.Conv2d(3, 64, 3, padding="same")
        self.conv3 = nn.Conv2d(64, 128, 3, padding="same")
        self.convT1 = nn.ConvTranspose2d(128, 64, 2, stride=2, dilation=1)
        self.convT2 = nn.ConvTranspose2d(64, 3, 2, stride=2, dilation=1)

        self.b1 = nn.BatchNorm2d(3)
        self.b2 = nn.BatchNorm2d(64)
        self.b3 = nn.BatchNorm2d(128)
        self.b4 = nn.BatchNorm2d(64)
        self.b5 = nn.BatchNorm2d(3)
        self.b6 = nn.BatchNorm2d(3)

        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(3*cfg["image_height"]*cfg["image_width"], 4096)
        self.l2 = nn.Linear(4096, 51)

    def forward(self, x):
        x = F.relu(self.b1(self.conv1(x))) # (3, 640, 640)
        s1 = x
        x = self.maxpool(x)
        x = F.relu(self.b2(self.conv2(x))) # (64, 320, 320)
        s2 = x
        x = self.maxpool(x)
 
        x = F.relu(self.b3(self.conv3(x))) # (128, 160, 160)

        x = F.relu(self.b4(self.convT1(x))) # (64, 320, 320)
        x = x + s2
        x = F.relu(self.b5(self.convT2(x))) # (3, 640, 640)
        x = x + s1

        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        return x