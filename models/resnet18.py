import torch.nn as nn
import torchvision.models as models


class LicensePlateDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        # Adjust first convolution layer to better handle 640x480 images
        self.feature_extractor[0] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_channels = 512
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
