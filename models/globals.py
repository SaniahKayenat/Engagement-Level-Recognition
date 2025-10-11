import torch
import torch.nn as nn

class StudentGlobalNet(nn.Module):
    def __init__(self,flow_ex,i3d_ex,track_id_max,track_id_min,num_classes):
        super().__init__()
        self.flow_ex      = flow_ex
        self.i3d_ex       = i3d_ex
        n_students = track_id_max - track_id_min + 1
        self.student_fuse = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.classifier   = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, student_flows, video_rgb):
        # both student_flows and video_rgb are already on the correct device

        # 1) get N×256 embeddings
        embeds = self.flow_ex(student_flows.squeeze(0))     # [N,256]

        # 2) pool across students
        sf = embeds.mean(dim=0, keepdim=True)              # [1,256]
        sf = self.student_fuse(sf)                         # [1,512]

        # 3) extract I3D
        gv = self.i3d_ex(video_rgb)                        # [1,512]

        # 4) fuse & classify
        x  = torch.cat([sf, gv], dim=1)                    # [1,1024]
        return self.classifier(x).squeeze(0)               # [num_classes]