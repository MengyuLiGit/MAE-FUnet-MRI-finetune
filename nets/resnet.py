import torch
import torch.nn as nn


class ResnetModel(nn.Module):
    def __init__(self, resnet, num_classes1):
        super(ResnetModel, self).__init__()
        self.model_resnet = resnet
        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs, num_classes1)

    def forward(self, x):
        x = self.model_resnet(x)
        out1 = self.fc1(x)
        return out1
