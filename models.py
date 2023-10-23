import numpy as np
import torch.nn as nn

class PitchEstimationModel(nn.Module):
    def __init__(self):
        super(PitchEstimationModel, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1024, kernel_size=512, stride=4)
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=64, stride=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=64, stride=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=64, stride=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=64, stride=1)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=64, stride=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(2048, 360)

    def forward(self, x):
        # First layer
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        # Second layer
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        # Third layer
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        # Fourth layer
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        # Fifth layer,
        x = self.conv5(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        # Sixth layer
        x = self.conv6(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        # Not sure if 100% needed yet
        x = x.view(x.size(0), -1)
        # Fully connected layer
        x = self.fc(x)
        return x


model = PitchEstimationModel()
print(model)
