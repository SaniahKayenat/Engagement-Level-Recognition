import cv2
import torch
import numpy as np
import os
from torchvision.models.optical_flow import raft_large
from torchvision.transforms.functional import to_tensor
from torchvision.utils import flow_to_image

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = raft_large(pretrained=True, progress=False).to(device).eval()

# Parameters
video_path = '/data/Saniah/Video/Dataset_30fps_cleaned/High/high-2636（1-1000）_view134.mp4.mp4'
output_dir = '/data/Saniah/Video/Outputs/high-2636（1-1000）_view134.mp4'
os.makedirs(output_dir, exist_ok=True)

# Resize target resolution
resize_height = 384  
resize_width = 512   

# Video load
cap = cv2.VideoCapture(video_path)
ret, frame1 = cap.read()
if not ret:
    raise ValueError("Could not read first frame.")

frame1 = cv2.resize(frame1, (resize_width, resize_height))
frame1_tensor = to_tensor(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

frame_idx = 0

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    frame2 = cv2.resize(frame2, (resize_width, resize_height))
    frame2_tensor = to_tensor(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    with torch.no_grad():
        flow = model(frame1_tensor, frame2_tensor)[-1]

    flow_img = flow_to_image(flow)[0].permute(1, 2, 0).cpu().numpy()

    out_path = os.path.join(output_dir, f"flow_{frame_idx:04d}.png")
    cv2.imwrite(out_path, cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR))

    # Prepare for next iteration
    frame1_tensor = frame2_tensor
    frame_idx += 1

    # Optional memory cleanup
    del frame2_tensor, flow
    torch.cuda.empty_cache()

cap.release()
cv2.destroyAllWindows()
print(f"Done! Saved {frame_idx} flow frames to {output_dir}")
