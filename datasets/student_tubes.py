# dataset/student_tubes.py
import os
import csv
import cv2
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms

torch.manual_seed(0)

class StudentTubesDataset(Dataset):
    """
    Modified Dataset:
    - Only considers the first `tube_length` frames per video, discards the rest.
    - Optionally filters track IDs to a specified inclusive range [track_id_min, track_id_max].
    - Returns one clip per unique track_id within the first `tube_length` frames.
    - If total frames for an ID < `tube_length`, repeats frames cyclically to match the tube_length.
    """
    def __init__(self, csv_path, videos_root, tube_length, crop_size=(128,128),
                 track_id_min=0, track_id_max=6):
        self.tube_length = tube_length
        self.videos_root = videos_root
        self.track_id_min = track_id_min
        self.track_id_max = track_id_max
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        tubes = defaultdict(list)
        classes = set()

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_id = int(row['frame_id'])
                if frame_id > tube_length:
                    continue  # discard frames beyond tube_length

                track_id = int(row['track_id'])
                if (self.track_id_min is not None and track_id < self.track_id_min) or \
                   (self.track_id_max is not None and track_id > self.track_id_max):
                    continue  # filter track IDs outside range

                video = row['video']
                cls = row['engagement_class']
                classes.add(cls)

                key = (video, track_id, cls)
                tubes[key].append((frame_id,
                                   int(row['x1']), int(row['y1']),
                                   int(row['x2']), int(row['y2'])))

        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.entries = []

        for (video, track_id, cls), frames in tubes.items():
            frames = sorted(frames, key=lambda x: x[0])
            if len(frames) < tube_length:
                needed = tube_length - len(frames)
                cycles = frames * (needed // len(frames) + 1)
                frames = frames + cycles[:needed]
            else:
                frames = frames[:tube_length]

            self.entries.append({
                'video': video,
                'track_id': track_id,
                'cls': cls,
                'label': self.class_to_idx[cls],
                'clip': frames
            })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        video_path = os.path.join(self.videos_root, e['cls'], f"{e['video'].strip()}.mp4")
        cap = cv2.VideoCapture(video_path)

        frames = []
        last_img = None
        for fid, x1, y1, x2, y2 in e['clip']:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid-1)
            ret, img = cap.read()
            if not ret:
                img = last_img if last_img is not None else np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
            else:
                last_img = img

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            crop = img[y1:y2, x1:x2]
            frames.append(self.transform(crop))

        cap.release()
        tube = torch.stack(frames)  # (tube_length, 3, H, W)
        label = e['label']
        return tube, label
