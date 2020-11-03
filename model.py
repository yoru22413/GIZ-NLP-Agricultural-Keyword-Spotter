import torch.nn.functional as F
from torch import nn


class ModelForward(nn.Module):
    def __init__(self, num_classes=193):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, 64, 3)
        self.cnn2 = nn.Conv2d(64, 64, 3)
        self.cnn3 = nn.Conv2d(64, 64, 3)
        self.cnn4 = nn.Conv2d(64, 128, 3)
        self.cnn5 = nn.Conv2d(128, 256, 3)
        #self.cnn6 = nn.Conv2d(256, 512, 3)

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.batch_norm5 = nn.BatchNorm2d(256)
        #self.batch_norm6 = nn.BatchNorm2d(512)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.cnn1(x)))
        x = F.relu(self.batch_norm2(self.cnn2(x)))
        x = F.relu(self.batch_norm3(self.cnn3(x)))
        x = F.relu(self.batch_norm4(self.cnn4(x)))
        x = F.relu(self.batch_norm5(self.cnn5(x)))
        #x = F.relu(self.batch_norm6(self.cnn6(x)))
        x = self.gap(x).view(x.shape[0], -1)
        x = self.linear(x)
        return x
