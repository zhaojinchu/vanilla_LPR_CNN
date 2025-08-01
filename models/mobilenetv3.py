import torch.nn as nn
import torchvision.models as models


class LicensePlateDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        base_model.classifier = nn.Identity()
        first_block = base_model.features[0]
        conv_layer = first_block[0]
        conv_layer.stride = (1, 1)
        conv_layer.kernel_size = (7, 7)
        conv_layer.padding = (3, 3)
        self.feature_extractor = base_model.features
        in_channels = 960
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
