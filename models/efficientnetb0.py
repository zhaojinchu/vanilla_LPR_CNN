import torch.nn as nn
import torchvision.models as models


class LicensePlateDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        base_model.classifier = nn.Identity()
        self.feature_extractor = base_model.features
        in_channels = base_model.classifier.in_features if hasattr(base_model.classifier, 'in_features') else 1280
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.reg_head(x)


def build_model(pretrained=True):
    return LicensePlateDetector(pretrained)
