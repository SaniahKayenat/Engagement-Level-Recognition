import torch.nn as nn
import torch
from torchvision.models.video import r3d_18, R3D_18_Weights


class FeatureExtractor(nn.Module):
    def __init__(self, feat_dim=2048, dim =1024):
        super().__init__()
        base = r3d_18(weights=R3D_18_Weights.DEFAULT)
        modules = list(base.children())[:-1]  
        self.rgb_backbone = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        in_feat = base.fc.in_features
        self.fc_rgb = nn.Linear(in_feat, feat_dim)
        self.fc_rgb2 = nn.Linear(feat_dim,dim)

    def forward(self, x_rgb):
        # x_rgb: (B, 3, T, H, W)
        f1 = self.rgb_backbone(x_rgb)        # [B, C, 1, 1, 1]
        f1 = self.pool(f1).flatten(1)       # [B, C]
        f1 = self.fc_rgb(f1)                # [B, feat_dim]
        f1 = self.fc_rgb2(f1)
        return f1


