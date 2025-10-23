import os
import sys
import torch
import csv
import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(r"F:\Saniah\mmdetection"))
from mmdet.apis import inference_detector, init_detector
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---- CONFIG ----
root_dir         = '/data/Saniah/Video/Datasets/Dataset_30fps_all/Low'
config           = '/data/Saniah/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
checkpoint       = '/data/Saniah/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device           = 'cuda:0'
output_csv       = '/data/Saniah/Video/CSV/track_detect_all_newfound_issue_low.csv'
score_thr        = 0.7    
max_age          = 400    
n_init           = 3   
nms_max_overlap  = 1.0      

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
write_header = not os.path.exists(output_csv)
csv_file = open(output_csv, 'a', newline='')
csv_writer = csv.writer(csv_file)
if write_header:
    csv_writer.writerow(['video', 'frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'engagement_class'])

# Initialize MMDetection model and Deep SORT tracker once
model   = init_detector(config, checkpoint, device=device)
model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

for vid_file in tqdm(sorted(os.listdir(root_dir))):
    tracker = DeepSort(max_age=max_age, n_init=n_init, nms_max_overlap=nms_max_overlap)
    if not vid_file.lower().endswith(('.mp4')):
        continue
    video_path = os.path.join(root_dir, vid_file)
    reader = mmcv.VideoReader(video_path)
    frame_idx = 0
    
    for frame in track_iter_progress((reader, len(reader))):
        frame_idx += 1
        # MMDet inference
        ds   = inference_detector(model, frame, test_pipeline=test_pipeline)
        inst = ds.pred_instances
        bboxes = inst.bboxes.cpu().numpy()
        scores = inst.scores.cpu().numpy()
        labels = inst.labels.cpu().numpy()

        # Filter for persons
        keep = (labels == 0) & (scores >= score_thr)
        bboxes = bboxes[keep]
        scores = scores[keep]

        # Update tracker
        detections = [(bbox, score, 'person') for bbox, score in zip(bboxes, scores)]
        tracks = tracker.update_tracks(detections, frame=frame)

        # Log each confirmed track
        for track in tracks:
            if track.time_since_update > 0 or not (track.is_confirmed()):
                continue
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            tid = track.track_id
            csv_writer.writerow([
                vid_file, frame_idx, tid, x1, y1, x2, y2, "Low"
            ])
        del ds, inst, bboxes, scores, labels, detections, tracks
        torch.cuda.empty_cache()

# Cleanup
csv_file.close()
print(f'Detections appended to {output_csv}')
