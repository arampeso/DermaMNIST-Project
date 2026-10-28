import torch
import torch.nn as nn
import torchvision.models as models


class CustomCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Input: (N, 3, 64, 64)
        self.features = nn.Sequential(
            # Conv2d: (N, 32, 64, 64)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            # ReLU: (N, 32, 64, 64)
            nn.ReLU(inplace=True),
            # Conv2d: (N, 32, 64, 64)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # ReLU: (N, 32, 64, 64)
            nn.ReLU(inplace=True),
            # MaxPool2d: (N, 32, 32, 32)
            nn.MaxPool2d(kernel_size=2),

            # Conv2d: (N, 64, 32, 32)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # ReLU: (N, 64, 32, 32)
            nn.ReLU(inplace=True),
            # Conv2d: (N, 64, 32, 32)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # ReLU: (N, 64, 32, 32)
            nn.ReLU(inplace=True),
            # MaxPool2d: (N, 64, 16, 16)
            nn.MaxPool2d(kernel_size=2),

            # Conv2d: (N, 128, 16, 16)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            # ReLU: (N, 128, 16, 16)
            nn.ReLU(inplace=True),
            # Conv2d: (N, 128, 16, 16)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # ReLU: (N, 128, 16, 16)
            nn.ReLU(inplace=True),
            # AdaptiveAvgPool2d: (N, 128, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            # Flatten: (N, 128)
            nn.Flatten(),
            # Linear: (N, 64)
            nn.Linear(in_features=128, out_features=64),
            # ReLU and Dropout: (N, 64)
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # Linear: (N, num_classes), (N, 7) by default
            nn.Linear(in_features=64, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class MyModel(torch.nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()

        if backbone == "custom":
            self.model = CustomCNN(num_classes=7)
            return

        weights_map = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
        }

        if backbone == "resnet18":
            self.model = models.resnet18(
                weights=weights_map[backbone] if pretrained else None
            )
        elif backbone == "resnet34":
            self.model = models.resnet34(
                weights=weights_map[backbone] if pretrained else None
            )
        elif backbone == "resnet50":
            self.model = models.resnet50(
                weights=weights_map[backbone] if pretrained else None
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 7)

    def forward(self, x):
        return self.model(x)