import os, cv2, torch, numpy as np
from torchvision.models.optical_flow import raft_large
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = raft_large(pretrained=True, progress=False).to(device).eval()

video_folder = '/data/Saniah/Video/Datasets/Dataset_30fps_all/Low'
output_dir   = '/data/Saniah/Video/Raft/Raft_features_all/Low'
os.makedirs(output_dir, exist_ok=True)

resize_h, resize_w = 128,128
videos = [f for f in os.listdir(video_folder) if f.lower().endswith('.mp4')]

for video in tqdm(videos):
    video_path = os.path.join(video_folder, video)
    if os.path.exists(os.path.join(output_dir, video.replace('.mp4','.npy'))):
        continue
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame1 = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        frame1 = cv2.resize(frame1, (resize_w, resize_h))
        frame1_tensor = to_tensor(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))\
                            .unsqueeze(0).to(device)

        flows = []
        while True:
            ret, frame2 = cap.read()
            if not ret:
                break
            frame2 = cv2.resize(frame2, (resize_w, resize_h))
            frame2_tensor = to_tensor(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))\
                                .unsqueeze(0).to(device)

            with torch.no_grad():
                flow = model(frame1_tensor, frame2_tensor)[-1]

            flows.append(flow[0].cpu().numpy())
            frame1_tensor = frame2_tensor

        cap.release()

        flows = np.stack(flows, axis=0)  

        out_path = os.path.join(output_dir, video.replace('.mp4','.npy'))
        np.save(out_path, flows)

    except Exception as e:
        print(f"⚠️  Skipped {video}: {e}")
