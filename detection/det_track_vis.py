# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from deep_sort_realtime.deepsort_tracker import DeepSort

sys.path.insert(0, os.path.abspath(r"/data/Saniah/mmdetection"))
from mmdet.apis import inference_detector, init_detector

# ---- CONFIG ----
input_video      = '/data/Saniah/Video/Dataset_30fps/High/high-2636（1-1000）_view12.mp4.mp4'   # path to your single input video
config           = '/data/Saniah/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
checkpoint       = '/data/Saniah/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device           = 'cuda:0'
score_thr        = 0.7     
max_age          = 400     
n_init           = 3      
nms_max_overlap  = 1.0   
output_video     = '/data/Saniah/Video/Outputs/Detect/High/high-2636（1-1000）_view12_tracked.mp4'

os.makedirs(os.path.dirname(output_video), exist_ok=True)

# Initialize MMDetection model and Deep SORT tracker
model   = init_detector(config, checkpoint, device=device)
model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
tracker = DeepSort(max_age=max_age, n_init=n_init, nms_max_overlap=nms_max_overlap)

# Set to collect unique track IDs
unique_ids = set()

# Open video reader
reader = mmcv.VideoReader(input_video)
if not reader:
    raise RuntimeError(f"Failed to open video: {input_video}")

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    output_video,
    fourcc,
    reader.fps,
    (reader.width, reader.height)
)

# Process each frame
for frame in track_iter_progress((reader, len(reader))):
    # MMDet inference
    ds = inference_detector(model, frame, test_pipeline=test_pipeline)
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

    # Draw boxes and IDs
    for track in tracks:
        if track.time_since_update > 0 or not track.is_confirmed():
            continue
        tid = track.track_id
        unique_ids.add(tid)
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            str(tid),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    writer.write(frame)

writer.release()

print(f"Saved tracked video to {output_video}")
print(f"Total unique track IDs found: {len(unique_ids)}")
