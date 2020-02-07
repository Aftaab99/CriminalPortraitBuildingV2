import torch.nn as nn
import torch


class FeatureExtractionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, (3, 3), stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d((3, 3), stride=3)
        self.conv2 = nn.Conv2d(8, 8, (3, 3), stride=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d((3, 3), stride=3)

        self.linear1 = nn.Linear(3 * 3 * 8, 40)
        self.sigm1 = nn.Sigmoid()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = x.view(-1, 3 * 3 * 8)
        x = self.linear1(x)
        return self.sigm1(x)
