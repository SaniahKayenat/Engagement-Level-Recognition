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
from models.i3d import SingleStreamExtractor
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ─── Configuration ─────────────────────────────────────────────────────────────
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

# ─── Dataset ───────────────────────────────────────────────────────────────────
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

# Build labels array without triggering heavy __getitem__:
labels_all = np.array([int(dataset[i][2]) for i in tqdm(indices)]) 


full_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
# ─── Model Definition ──────────────────────────────────────────────────────────
flow_ex = TransformerFlowExtractor().to(device)
i3d_ex  = SingleStreamExtractor().to(device)

class StudentGlobalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_ex      = flow_ex
        self.i3d_ex       = i3d_ex
 # infer dims (assumes flow_ex/out_dim & i3d_ex/feat_dim; fallback to 512)
        flow_dim = getattr(self.flow_ex, "out_dim", 512)
        rgb_dim  = getattr(self.i3d_ex,  "feat_dim", 512)

        # Project each stream to a common dim
        self.flow_proj = nn.LazyLinear(512)   # or nn.Linear(flow_dim, 512) if you know it
        self.rgb_proj  = nn.LazyLinear(512)   # or nn.Linear(rgb_dim,  512)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),  # 512(flow) + 512(rgb)
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )


    def forward(self, student_flows, video_rgb):
      
        # print(student_flows.shape)                        [5, 150, 2, 64, 64]
        embeds = self.flow_ex(student_flows)             #  [5,out_shape]
        pooled_embeds = embeds.mean(dim=0, keepdim=True) #  [1,out_shape] not 512
        # 3) extract I3D
        gv = self.i3d_ex(video_rgb)                        # [1,512]

        # 4) fuse & classify
        f = self.flow_proj(pooled_embeds)                  # [1, 512]
        r = self.rgb_proj(gv)                              # [1, 512]

        x = torch.cat([f, r], dim=1) 
        return self.classifier(x).squeeze(0)   

# ─── Load Model ────────────────────────────────────────────────────────────────
model     = StudentGlobalNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Calculate class weights
# Alternative approach - iterate through dataloader (slower)
train_labels = []
for _, _, labels in tqdm(full_loader):
    train_labels.extend(labels.tolist())

class_counts   = np.bincount(train_labels, minlength=num_classes)  # <- add minlength
total_samples  = class_counts.sum()
eps = 1e-8
class_weights  = total_samples / (num_classes * (class_counts + eps))
class_weights  = torch.tensor(class_weights, dtype=torch.float32, device=device)
criterion      = nn.CrossEntropyLoss(weight=class_weights)
#criterion      = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss()
best_f1, best_loss = -1.0, float('inf')
min_delta = 1e-4
wait = 0
# Add scheduler - reduces LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',           # minimize validation loss
    factor=0.5,          # multiply LR by 0.5
    patience=10,         # wait 10 epochs before reducing
    min_lr=1e-7    
)


# ─── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
test_loss, true_test, pred_test = 0.0, [], []

with torch.no_grad():
    for student_flows_list, video_rgb, label in tqdm(full_loader, desc="Test"):
        video_rgb     = video_rgb.to(device)            # [1, 3, T, H, W]
        label         = label.to(device)     
        # Process each sample in the batch
        batch_logits = []
        for i, student_flows in enumerate(student_flows_list):
            student_flows = student_flows.to(device)      # [1, N, T, 2, h, w] # [1, N, T, 2, h, w]
            single_rgb = video_rgb[i:i+1]       # [1, 3, T, H, W]
            logits = model(student_flows, single_rgb)  # [num_classes]
            batch_logits.append(logits)
        
        # Stack logits from all samples in batch
        batch_logits = torch.stack(batch_logits, dim=0)  # [batch_size, num_classes]
        
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
# Per-class metrics
# Build idx->class_name mapping from your dataset
idx_to_class = {v: k for k, v in dataset.flow_ds.class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

print("\nPer-class Precision / Recall / F1:")
print(classification_report(true_test,
                            pred_test,
                            target_names=class_names,
                            digits=4))

# And you can still log the overall row if you want
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

# ─── Plotting Curves ───────────────────────────────────────────────────────────


# ─── Confusion Matrix ───────────────────────────────────────────────────────────
# After test evaluation, calculate confusion matrix
cm = confusion_matrix(true_test, pred_test)
cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names)

plt.figure(figsize=(8, 6))
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_path, 'confusion_matrix_eval_all.png'))

print("Training complete.")