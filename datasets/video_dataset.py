# dataset/student_tubes.py
import os, csv, cv2, torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms        
from collections import defaultdict


torch.manual_seed(0)
# ─── 2) Video‐level wrapper ───────────────────────────────────────────────────
class VideoTubeDataset(Dataset):
    def __init__(self, clip_dataset):
        self.clip_ds  = clip_dataset
        # group clip indices by video ID
        self.by_video = defaultdict(list)
        for idx, entry in enumerate(clip_dataset.entries):
            self.by_video[entry['video'] ].append(idx)
        self.videos   = list(self.by_video.keys())

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, vidx):
        vid      = self.videos[vidx]
        clip_ids = self.by_video[vid]
        # stack all clips for this video: [N_clips, T, 3, H, W]
        tubes    = torch.stack([ self.clip_ds[i][0] for i in clip_ids ], dim=0)
        # label is same for all clips in this video
        label    = self.clip_ds[ clip_ids[0] ][1]
        return tubes, label