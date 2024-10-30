import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet152, ResNet152_Weights


class pharynet152(nn.Module):
    def __init__(self, num_classes=2):
        super(pharynet152, self).__init__()
        # load pre-trained ResNet-152 model
        self.model = resnet152(weights=ResNet152_Weights.DEFAULT)
        # replace final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def get_model():
    model = pharynet152()
    return model
