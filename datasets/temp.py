# dataset/student_tubes.py
import os, csv, cv2, numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class StudentTubesRaft(Dataset):
    def __init__(
        self,
        csv_path,
        flow_root,
        tube_length,
        flow_size=(128,128),
        flow_crop_size=(64,64),          # <— NEW: fixed size for each crop
        track_id_min=None,
        track_id_max=None,
        videos_root=None
    ):
        self.tube_length    = tube_length
        self.flow_root      = flow_root
        self.flow_h, self.flow_w   = flow_size
        self.crop_h, self.crop_w   = flow_crop_size  # <— store
        self.track_id_min  = track_id_min
        self.track_id_max  = track_id_max
        self.videos_root   = videos_root

        # Parse CSV
        tubes = defaultdict(list)
        classes = set()
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row['video']
                fid = int(row['frame_id'])
                if fid > tube_length:
                    continue
                tid = int(row['track_id'])
                if (track_id_min is not None and tid < track_id_min) or \
                   (track_id_max is not None and tid > track_id_max):
                    continue
                cls = row['engagement_class']
                classes.add(cls)
                x1, y1, x2, y2 = map(int, (row['x1'], row['y1'], row['x2'], row['y2']))
                tubes[(vid, tid, cls)].append((fid, x1, y1, x2, y2))
        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}

        # Build entries
        self.entries = []
        for (vid, tid, cls), frames in tubes.items():
            frames = sorted(frames, key=lambda x: x[0])
            L = len(frames)
            if L < tube_length:
                needed = tube_length - L
                frames = frames + frames * ((needed // L) + 1)
                frames = frames[:tube_length]
            else:
                frames = frames[:tube_length]
            self.entries.append({
                'video': vid,
                'track_id': tid,
                'cls': cls,
                'label': self.class_to_idx[cls],
                'frames': frames
            })

        # Gather original video sizes if provided
        self.video_sizes = {}
        if videos_root:
            for entry in self.entries:
                vid = entry['video']
                if vid in self.video_sizes:
                    continue
                path = os.path.join(videos_root, entry['cls'], f"{vid}.mp4")
                cap = cv2.VideoCapture(path)
                W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
                self.video_sizes[vid] = (int(W), int(H))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e   = self.entries[idx]
        vid = e['video']
        arr = np.load(os.path.join(self.flow_root,e['cls'], f"{vid[:-4]}.npy"))
        # arr.shape == [T, 2, flow_h, flow_w]

        # determine scaling from original to flow space
        if self.videos_root:
            W0, H0 = self.video_sizes[vid]
            sx = self.flow_w / W0
            sy = self.flow_h / H0
        else:
            sx = sy = 1.0

        crops = []
        for fid, x1, y1, x2, y2 in e['frames']:
            i = max(0, min(fid-1, arr.shape[0]-1))
            # scale box
            x1r, x2r = int(x1*sx), int(x2*sx)
            y1r, y2r = int(y1*sy), int(y2*sy)
            raw = arr[i, :, y1r:y2r, x1r:x2r]            # (2, Hf, Wf)
            t   = torch.from_numpy(raw).unsqueeze(0)      # (1,2,Hf,Wf)
            # resize to fixed crop size:
            t   = F.interpolate(t, size=(self.crop_h, self.crop_w),
                                mode='bilinear', align_corners=False)
            crops.append(t.squeeze(0))                    # (2, crop_h, crop_w)

        # now all crops are (2, crop_h, crop_w), so stack OK
        flows_tensor = torch.stack(crops, dim=0)         # (tube_length, 2, crop_h, crop_w)
        return flows_tensor, e['label']

