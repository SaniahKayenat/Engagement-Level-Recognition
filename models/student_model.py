import torch
import torch.nn as nn
class StudentGlobalNet(nn.Module):
    def __init__(self,flow_ex,i3d_ex,num_classes):
        super().__init__()
        self.flow_ex      = flow_ex
        self.i3d_ex       = i3d_ex
        self.num_classes = num_classes
 
        flow_dim = getattr(self.flow_ex, "out_dim", 512)
        rgb_dim  = getattr(self.i3d_ex,  "feat_dim", 512)

        self.flow_proj = nn.LazyLinear(512)  
        self.rgb_proj  = nn.LazyLinear(512)   

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )


    def forward(self, student_flows, video_rgb):
      
        # print(student_flows.shape)                        [5, 150, 2, 64, 64]
        embeds = self.flow_ex(student_flows)             #  [5,out_shape]
        pooled_embeds = embeds.mean(dim=0, keepdim=True) #  [1,out_shape] 
        gv = self.i3d_ex(video_rgb)                        # [1,512]
        f = self.flow_proj(pooled_embeds)                  # [1, 512]
        r = self.rgb_proj(gv)                              # [1, 512]

        x = torch.cat([f, r], dim=1) 
        return self.classifier(x).squeeze(0)               # [num_classes]