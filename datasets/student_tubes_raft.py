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
                path = os.path.join(videos_root, entry['cls'],vid)
                if not os.path.exists(path):
                    raise ValueError(f"Invalid path {path}, entry['video'] : {entry['video']}")
                cap = cv2.VideoCapture(path)
                W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
                self.video_sizes[vid] = (int(W), int(H))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        vid = e['video']
        arr = np.load(os.path.join(self.flow_root,e['cls'], f"{vid[:-4]}.npy"))
        # arr: [T, 2, H_res, W_res]
        # get original size for scaling
        if self.videos_root:
            W_orig, H_orig = self.video_sizes[vid]
            if W_orig == 0 or H_orig == 0:
                raise ValueError(f"Invalid W_orig={W_orig}, H_orig={H_orig} at index {idx} | path: {os.path.join(self.flow_root,e['cls'], f'{vid[:-4]}.npy')}")
            scale_x = self.flow_w / W_orig
            scale_y = self.flow_h / H_orig
        else:
            scale_x = scale_y = 1.0  # assume coordinates already in flow space

        crops = []
        for fid, x1, y1, x2, y2 in e['frames']:
            idxf = max(0, min(fid-1, arr.shape[0]-1))
            # scale coords
            x1r = int(x1 * scale_x)
            y1r = int(y1 * scale_y)
            x2r = int(x2 * scale_x)
            y2r = int(y2 * scale_y)
            # clamp to valid range and enforce at least 1px width/height
            x1r = max(0, min(x1r, self.flow_w - 1))
            x2r = max(x1r + 1, min(x2r, self.flow_w))
            y1r = max(0, min(y1r, self.flow_h - 1))
            y2r = max(y1r + 1, min(y2r, self.flow_h))

            raw = arr[idxf, :, y1r:y2r, x1r:x2r]  # now Hf, Wf ≥ 1
            t   = torch.from_numpy(raw).unsqueeze(0)  # (1,2,Hf,Wf)
            # resize to fixed crop size
            t   = F.interpolate(
                t,
                size=(self.crop_h, self.crop_w),
                mode='bilinear',
                align_corners=False
            )
            crops.append(t.squeeze(0))
        # (tube_length, 2, h_crop, w_crop)
        flows_tensor = torch.stack(crops)
        return flows_tensor, e['label']
