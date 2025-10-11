# dataset/flow_video_dataset.py

import torch
from torch.utils.data import Dataset
from collections import defaultdict
from .student_tubes_raft import StudentTubesRaft
from .video_dataset_raft import VideoDatasetRaft

class FlowVideoDataset(Dataset):
    """
    Combined Dataset for Video‐level training:
      - student_flows: RAFT crops per student  → [N_students, tube_length, 2, h_crop, w_crop]
      - video_rgb:     full‐video RGB clip       → [3, tube_length, H, W]
      - label:         engagement class index    → int
    """
    def __init__(
        self,
        csv_path: str,
        flow_root: str,
        videos_root: str,
        tube_length: int,
        flow_size=(128,128),
        resize=(128,128),
        track_id_min=None,
        track_id_max=None
    ):
        # 1) Student‐level flow dataset
        self.flow_ds = StudentTubesRaft(
            csv_path=csv_path,
            flow_root=flow_root,
            tube_length=tube_length,
            flow_size=flow_size,
            track_id_min=track_id_min,
            track_id_max=track_id_max,
            videos_root=videos_root
        )

        # 2) Group flow indices by video
        self.by_video = defaultdict(list)
        for idx, e in enumerate(self.flow_ds.entries):
            self.by_video[e['video']].append(idx)
        self.videos = list(self.by_video.keys())

        # build two mappings:
        self.video_cls   = { vid: self.flow_ds.entries[idxs[0]]['cls']
                            for vid,idxs in self.by_video.items() }
        self.video_label = { vid: self.flow_ds.entries[idxs[0]]['label']
                            for vid,idxs in self.by_video.items() }

        # pass both into VideoDatasetRaft
        self.video_ds = VideoDatasetRaft(
            video_list  = self.videos,
            video_cls   = self.video_cls,
            video_label = self.video_label,
            videos_root = videos_root,
            tube_length = tube_length,
            resize      = resize
        )

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid = self.videos[idx]
        # — student flows —
        flow_idxs = self.by_video[vid]
        student_flows = torch.stack(
            [ self.flow_ds[i][0] for i in flow_idxs ],
            dim=0
        )  # shape: [N_students, T, 2, h_crop, w_crop]

        # — full‐video RGB clip & label —
        video_rgb, label = self.video_ds[idx]
        # video_rgb: [3, T, H, W]

        return student_flows, video_rgb, label
