import torch
import torch.nn as nn
import torchvision.models as models

class ImageToPointCloud(nn.Module):
    def __init__(self, num_points=1024, backbone_name="resnet18", pretrained=False):
        """
        Simple baseline model: image -> feature -> point cloud.
        Args:
            num_points (int): Number of 3D points to predict.
            backbone_name (str): Backbone CNN architecture (default: resnet18).
            pretrained (bool): Whether to use pretrained ImageNet weights.
        """
        super().__init__()
        # Load backbone
        if backbone_name == "resnet18":
            backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = 512
        elif backbone_name == "resnet34":
            backbone = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = 512
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Remove classification head
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Regression head: map features to 3D point cloud
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_points * 3)
        )

        self.num_points = num_points

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Input image tensor of shape (B, 3, H, W).
        Returns:
            Tensor: Predicted point cloud of shape (B, num_points, 3).
        """
        feat = self.backbone(x)              # (B, feat_dim)
        out = self.head(feat)                # (B, num_points*3)
        return out.view(x.size(0), self.num_points, 3)

