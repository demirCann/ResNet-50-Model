import torch.nn as nn
from torchvision import models


def resnet50(pretrained=True, num_classes=10):
    model = models.resnet50(pretrained=pretrained)
    # Replace the last fully connected layer
    # CIFAR-10 has 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
