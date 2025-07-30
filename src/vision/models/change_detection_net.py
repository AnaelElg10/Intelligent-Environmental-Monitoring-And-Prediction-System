"""Change Detection Network for satellite image analysis."""

import torch
import torch.nn as nn


class ChangeDetectionNetwork(nn.Module):
    """Network for detecting changes between satellite images."""
    
    def __init__(self, backbone='resnet50', num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.conv1 = nn.Conv2d(6, 64, 3, padding=1)  # 6 channels for before/after
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)