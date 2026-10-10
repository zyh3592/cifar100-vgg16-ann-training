"""CIFAR-sized VGG-16: 13 convolutions + 3 fully connected layers."""
import runtime  # configure caches before torch import
import torch
from torch import nn


class VGG16CIFAR(nn.Module):
    def __init__(self, num_classes=100, batch_norm=True, pool='max',
                 hidden_dim=4096, dropout=0.5):
        super().__init__()
        if pool not in {'avg', 'max'}:
            raise ValueError('pool must be avg or max')
        self.model_config = dict(num_classes=num_classes, batch_norm=batch_norm,
                                 pool=pool, hidden_dim=hidden_dim, dropout=dropout)
        layers = []
        in_channels = 3
        for channels, count in [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]:
            for _ in range(count):
                layers.append(nn.Conv2d(in_channels, channels, 3, padding=1, bias=True))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(channels))
                layers.append(nn.ReLU(inplace=False))
                in_channels = channels
            layers.append(nn.AvgPool2d(2) if pool == 'avg' else nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.ReLU(inplace=False), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=False), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 32, 32):
            raise ValueError('Expected N x 3 x 32 x 32 CIFAR input')
        return self.classifier(self.features(x).flatten(1))


def build_model(config):
    return VGG16CIFAR(**config)
