import torch
import torch.nn as nn
from torchvision.models.convnext import convnext_tiny, convnext_small


class ModelAngleOnly(nn.Module):
    def __init__(self, feature_dim=256, angle_dim=72) -> None:
        super().__init__()
        feature_dim = feature_dim
        angle_dim = angle_dim

        self.encoder = torch.hub.load("pytorch/vision:v0.10.0", "resnet34", pretrained=True)
        self.encoder.fc = nn.Linear(512, feature_dim)

        # MLP output
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, angle_dim),
        )

    def forward(self, target):
        x = self.encoder(target)
        return self.mlp(x)


class ModelMultiCls(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        feature_dim = kwargs["feature_dim"]
        angle_dim = kwargs["angle_dim"]
        type_dim = kwargs["type_dim"]
        backbone = kwargs["backbone"]

        # BACKBONE
        if backbone == "convnext_tiny":
            self.encoder = convnext_tiny(weights="ConvNeXt_Tiny_Weights.DEFAULT")
        elif backbone == "convnext_small":
            self.encoder = convnext_small(weights="ConvNeXt_Small_Weights.DEFAULT")
        elif backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "resnet101":
            self.encoder = torch.hub.load("pytorch/vision:v0.10.0", backbone, pretrained=True)
        else:
            raise NotImplementedError

        # BRIGE
        if backbone == "convnext_tiny" or backbone == "convnext_small":
            self.encoder.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(768, feature_dim))
        elif backbone == "resnet18" or backbone == "resnet34":
            self.encoder.fc = nn.Linear(512, feature_dim)
        elif backbone == "resnet50" or backbone == "resnet101":
            self.encoder.fc = nn.Linear(2048, feature_dim)
        else:
            raise NotImplementedError

        # MLP output
        self.mlp_angle = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, angle_dim),
        )

        self.mlp_type = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, type_dim),
        )

    def forward(self, target):
        x = self.encoder(target)

        x_angle = self.mlp_angle(x)
        x_type = self.mlp_type(x)

        return x_angle, x_type


if __name__ == "__main__":
    model = ModelCls()
    target = torch.randn(1, 3, 512, 512)
    out = model(target)
    print("input shape: ", target.shape)
    print("output shape: ", out.shape)
