import torch.nn.functional as F
import torch.nn as nn
import torch

from config import cfg

class HourglassModel(nn.Module):
    def __init__(self):
        super(HourglassModel, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv0_0 = nn.Conv2d(3, 16, 3, padding="same")
        self.conv0_1 = nn.Conv2d(16, 16, 3, padding="same")
        self.conv1_0 = nn.Conv2d(16, 32, 3, padding="same")
        self.conv1_1 = nn.Conv2d(32, 32, 3, padding="same")
        self.conv2_0 = nn.Conv2d(32, 64, 3, padding="same")
        self.conv2_1 = nn.Conv2d(64, 64, 3, padding="same")
        self.conv3_0 = nn.Conv2d(64, 128, 3, padding="same")
        self.conv3_1 = nn.Conv2d(128, 128, 3, padding="same")
        
        self.b0 = nn.BatchNorm2d(16)
        self.b1 = nn.BatchNorm2d(32)
        self.b2 = nn.BatchNorm2d(64)
        self.b3 = nn.BatchNorm2d(128)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv4 = nn.Conv2d(128, 128, 3, padding="same")
        self.conv5 = nn.Conv2d(128, 64, 3, padding="same")
        self.conv6 = nn.Conv2d(64, 32, 3, padding="same")
        self.conv7 = nn.Conv2d(32, 17, 3, padding="same")

        self.b4 = nn.BatchNorm2d(128)
        self.b5 = nn.BatchNorm2d(64)
        self.b6 = nn.BatchNorm2d(32)
        self.b7 = nn.BatchNorm2d(17)

    def batch_soft_argmax(self, tensor):
        # Create grids of x, y coordinates each feature map.
        N, C, H, W = tensor.shape
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W))

        # Apply softmax to the input tensor along the height and width dimensions.
        softmaxed_tensor = torch.nn.functional.softmax(tensor.view(N,C,1,-1), dim=3).view(tensor.size())

        # Calculate the x, y coordinates using a weighted sum for each feature map.
        x = torch.sum(x_coords.float() * softmaxed_tensor, dim=(2,3))
        y = torch.sum(y_coords.float() * softmaxed_tensor, dim=(2,3))

        # Zip the x, y coordinates together.
        zipped = torch.hstack((x, y))
        return zipped

    def forward(self, x):
        # Encode.
        x = F.relu(self.b0(self.conv0_0(x)))
        x = F.relu(self.b0(self.conv0_1(x)))
        x = F.relu(self.b1(self.conv1_0(x)))
        x = F.relu(self.b1(self.conv1_1(x)))
        x = F.relu(self.b2(self.conv2_0(x)))
        x = F.relu(self.b2(self.conv2_1(x)))
        x = F.relu(self.b3(self.conv3_0(x)))
        x = F.relu(self.b3(self.conv3_1(x)))

        # Decode.
        x = F.relu(self.b4(self.conv4(self.upsample(x))))
        x = F.relu(self.b5(self.conv5(self.upsample(x))))
        x = F.relu(self.b6(self.conv6(self.upsample(x))))
        x = F.relu(self.b7(self.conv7(self.upsample(x))))

        return self.batch_soft_argmax(x)