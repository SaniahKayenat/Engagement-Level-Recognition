# dataset/video_dataset.py
import os, cv2, torch
from torch.utils.data import Dataset
from torchvision import transforms

class VideoDatasetRaft(Dataset):
    """
    - Returns (clip_tensor, label_idx)
    - clip_tensor: [3, T, H, W] from first tube_length frames
    - label_idx:  integer class index
    """
    def __init__(
        self,
        video_list,     # list of video IDs (strings)
        video_cls,      # dict: video_id -> class_name (folder name)
        video_label,    # dict: video_id -> int label index
        videos_root,
        tube_length,
        resize=(128,128)
    ):
        self.videos_root = videos_root
        self.videos      = video_list
        self.video_cls   = video_cls
        self.video_label = video_label
        self.tube_length = tube_length
        self.resize      = resize

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid        = self.videos[idx]
        cls_name   = self.video_cls[vid]      # e.g. "Engaged", "Bored"
        label_idx  = self.video_label[vid]    # e.g. 0,1,2

        # build the correct path
        video_path = os.path.join(self.videos_root, cls_name, f"{vid}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frames = []
        count  = 0
        while count < self.tube_length:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Cannot read frame from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.resize)
            frames.append(self.transform(frame))
            count += 1
        cap.release()

        # stack → [3, T, H, W]
        clip = torch.stack(frames, dim=1)
        return clip, label_idx
