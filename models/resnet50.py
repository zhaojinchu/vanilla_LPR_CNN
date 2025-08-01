import torch.nn as nn
import torchvision.models as models


class LicensePlateDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        self.feature_extractor[0] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_channels = 2048
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.reg_head(x)


def build_model(pretrained=True):
    return LicensePlateDetector(pretrained)
