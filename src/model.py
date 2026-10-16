import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(torch.nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()

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