import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datasets.flow_video_dataset import FlowVideoDataset
from models.flow_mlp import TransformerFlowExtractor
from models.i3d import FeatureExtractor
from models.student_model import StudentGlobalNet
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


device        = 'cuda' if torch.cuda.is_available() else 'cpu'
csv_path      = '/data/Saniah/Video/CSV/track_detect_all.csv'
flow_root     = '/data/Saniah/Video/Raft/Raft_features_all'
videos_root   = '/data/Saniah/Video/Datasets/Dataset_30fps_all'
results_path  = '/data/Saniah/Video/Results/Results_cleaned_tr'
model_path    = os.path.join(results_path, 'best_model.pth')
resize        = (128, 128)
tube_length   = 150
lr            = 1e-4
track_id_min  = 1
track_id_max  = 5
num_classes   = 3
metrics_csv   =  '/data/Saniah/Video/Results/Results_cleaned_tr/eval_all.csv'

# ─── Dataset ───────
dataset = FlowVideoDataset(
    csv_path=csv_path,
    flow_root=flow_root,
    videos_root=videos_root,
    tube_length=150,
    flow_size=(128,128),
    resize=(128,128),
    track_id_min=1,
    track_id_max=5
)
indices = np.arange(len(dataset))
df = pd.read_csv(csv_path)
df['engagement_class'] = df['engagement_class'].astype(str).str.strip()

mask_tid = df['track_id'].between(track_id_min, track_id_max)
mask_fid = df['frame_id'] <= tube_length

print("Raw CSV:", df['engagement_class'].value_counts().to_dict())
print("After track_id filter:", df[mask_tid]['engagement_class'].value_counts().to_dict())
print("After frame_id<=T:", df[mask_fid]['engagement_class'].value_counts().to_dict())
print("After both filters:", df[mask_tid & mask_fid]['engagement_class'].value_counts().to_dict())

labels_all = np.array([int(dataset[i][2]) for i in tqdm(indices)]) 

full_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# ─── Load Model ────────
flow_ex = TransformerFlowExtractor().to(device)
i3d_ex  = FeatureExtractor().to(device)
model     = StudentGlobalNet(flow_ex, i3d_ex).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Calculate class weights
train_labels = []
for _, _, labels in tqdm(full_loader):
    train_labels.extend(labels.tolist())

class_counts   = np.bincount(train_labels, minlength=num_classes)  # <- add minlength
total_samples  = class_counts.sum()
eps = 1e-8
class_weights  = total_samples / (num_classes * (class_counts + eps))
class_weights  = torch.tensor(class_weights, dtype=torch.float32, device=device)
criterion      = nn.CrossEntropyLoss(weight=class_weights)

best_f1, best_loss = -1.0, float('inf')
min_delta = 1e-4
wait = 0
# Add scheduler 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',           
    factor=0.5,         
    patience=10,       
    min_lr=1e-7    
)

# ─── Evaluation ───────
model.eval()
test_loss, true_test, pred_test = 0.0, [], []

with torch.no_grad():
    for student_flows_list, video_rgb, label in tqdm(full_loader, desc="Test"):
        video_rgb     = video_rgb.to(device)          
        label         = label.to(device)     

        batch_logits = []
        for i, student_flows in enumerate(student_flows_list):
            student_flows = student_flows.to(device)     
            single_rgb = video_rgb[i:i+1]   
            logits = model(student_flows, single_rgb)  
            batch_logits.append(logits)
        
        batch_logits = torch.stack(batch_logits, dim=0)  
        
        # Compute loss
        loss = criterion(batch_logits, label)

        test_loss += loss.item()
        pred_test.extend(batch_logits.argmax(dim=1).cpu().tolist())
        true_test.extend(label.cpu().tolist())

test_loss /= len(full_loader)

# Overall metrics
acc_test = accuracy_score(true_test, pred_test)
prec_test = precision_score(true_test, pred_test, average='weighted', zero_division=0)
rec_test = recall_score(true_test, pred_test, average='weighted', zero_division=0)
f1_test = f1_score(true_test, pred_test, average='weighted', zero_division=0)

print(f"\nFINAL TEST RESULTS:")
print(f"Test Accuracy:  {acc_test:.4f}")
print(f"Test Precision: {prec_test:.4f}")
print(f"Test Recall:    {rec_test:.4f}")
print(f"Test F1-Score:  {f1_test:.4f}")
print(f"Test Loss:      {test_loss:.4f}")

idx_to_class = {v: k for k, v in dataset.flow_ds.class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

print("\nPer-class Precision / Recall / F1:")
print(classification_report(true_test,
                            pred_test,
                            target_names=class_names,
                            digits=4))

pd.DataFrame([[
    'test', 
    None, 
    acc_test, 
    precision_score(true_test, pred_test, average='weighted'),
    recall_score(true_test, pred_test, average='weighted'),
    f1_score(true_test, pred_test, average='weighted'),
    test_loss,
    None
]], columns=['Split','Epoch','Accuracy','Precision','Recall','F1','Loss','LR']
).to_csv(metrics_csv, mode='a', header=False, index=False)

# ─── Confusion Matrix ───────────────────────────────────────────────────────────
cm = confusion_matrix(true_test, pred_test)
cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names)

plt.figure(figsize=(8, 6))
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_path, 'confusion_matrix_eval_all.png'))

print("Testing complete.")