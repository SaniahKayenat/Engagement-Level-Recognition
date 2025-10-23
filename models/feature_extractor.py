import torch.nn as nn
from torchvision.models.video import r3d_18
import torch 
class TwoStreamExtractor(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        base = r3d_18(pretrained=True)
        modules = list(base.children())[:-2]
        in_feat = base.fc.in_features
        self.rgb_backbone = nn.Sequential(*modules)
        self.flow_backbone = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc_rgb  = nn.Linear(in_feat, feat_dim)
        self.fc_flow = nn.Linear(in_feat, feat_dim)

    def forward(self, x_rgb, x_flow):

        f1 = self.rgb_backbone(x_rgb)
        f1 = self.pool(f1).flatten(1)
        f1 = self.fc_rgb(f1)
        f2 = self.flow_backbone(x_flow)
        f2 = self.pool(f2).flatten(1)
        f2 = self.fc_flow(f2)
        return torch.cat([f1,f2], dim=1)  # (B, 2*feat_dim)