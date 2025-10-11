#Imports

import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets.flow_video_dataset import FlowVideoDataset
from models.flow_mlp import TransformerFlowExtractor
from models.i3d import FeatureExtractor
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm  

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
# Device

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
tube_length   = 150
flow_root     = '/data/Saniah/Video/Raft/Raft_features_all'
videos_root   = '/data/Saniah/Video/Datasets/Dataset_30fps_all'
csv_path      = '/data/Saniah/Video/CSV/track_detect_all.csv'
resize        = (128,128)
num_classes   = 3
track_id_min  = 1
track_id_max  = 5
num_epochs    = 100
num_folds     = 5
patience      = 20
lr            = 1e-4
results_path  = '/data/Saniah/Video/Results/Results_all_tr'
os.makedirs(results_path, exist_ok=True)
metrics_csv = os.path.join(results_path, 'metrics.csv')
if not os.path.exists(metrics_csv):
    pd.DataFrame(
        columns=['Split','Epoch','Accuracy','Precision','Recall','F1','Loss','LR']
    ).to_csv(metrics_csv, index=False)


# Initialize lists to store metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_f1_scores = []
val_f1_scores = []

# … hyperparams …
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

# --- Stratified splits ---
trainval_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=labels_all,         
)

labels_trainval = labels_all[trainval_idx]
train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=0.125,                  # 12.5% of trainval -> 10% of total
    random_state=42,
    shuffle=True,
    stratify=labels_trainval,        
)
def counts(lbls, name):
    c = np.bincount(lbls, minlength=num_classes)
    print(f"{name} class counts:", c.tolist())

counts(labels_all[train_idx], "train")
counts(labels_all[val_idx],   "val")
counts(labels_all[test_idx],  "test")


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized data
    """
    student_flows_list = []
    video_rgb_list = []
    labels_list = []
    
    for student_flows, video_rgb, label in batch:
        student_flows_list.append(student_flows)
        video_rgb_list.append(video_rgb)
        labels_list.append(label)
    
    # Stack videos (should be same size)
    video_rgb_batch = torch.stack(video_rgb_list)  # [batch_size, 3, T, H, W]
    labels_batch = torch.tensor(labels_list)       # [batch_size]
    
    # For student flows, if they have different N (number of students), 
    # we'll process them individually anyway, so just return as list
    return student_flows_list, video_rgb_batch, labels_batch
train_loader = DataLoader(
    Subset(dataset, train_idx), batch_size=1, shuffle=True, num_workers=4, collate_fn = custom_collate_fn
)
val_loader = DataLoader(
    Subset(dataset, val_idx), batch_size=1, shuffle=False, num_workers=4,collate_fn = custom_collate_fn
)
test_loader = DataLoader(
    Subset(dataset, test_idx), batch_size=1, shuffle=False, num_workers=4,collate_fn = custom_collate_fn
)

# Get names from the underlying flow dataset entries
test_video_names = [dataset.flow_ds.entries[int(i)]["video"] for i in tqdm(test_idx)]


out_csv = os.path.join(results_path, "test_videos.csv")
pd.DataFrame({"video": test_video_names}).to_csv(out_csv, index=False)
print(f"Saved test video names to: {out_csv}")

print(f"Dataset splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
# # print(f"Batch size: {train_loader.batch_size}")


class StudentGlobalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_ex      = flow_ex
        self.i3d_ex       = i3d_ex
 
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
        pooled_embeds = embeds.mean(dim=0, keepdim=True) #  [1,out_shape] 
        # 3) extract I3D
        gv = self.i3d_ex(video_rgb)                        # [1,512]

        # 4) fuse & classify
        f = self.flow_proj(pooled_embeds)                  # [1, 512]
        r = self.rgb_proj(gv)                              # [1, 512]

        x = torch.cat([f, r], dim=1) 
        return self.classifier(x).squeeze(0)               # [num_classes]
# ─── Models ───────────────────────────────────────────────────────────────────
flow_ex = TransformerFlowExtractor().to(device)
i3d_ex  = FeatureExtractor().to(device)
              # [num_classes]


# ─── Training Setup ───────────────────────────────────────────────────────────
model     = StudentGlobalNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Calculate class weights

train_labels = []
for _, _, labels in tqdm(train_loader):
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

# ─── Training Loop ────────────────────────────────────────────────────────────
epoch_done = 0
for epoch in range(1, num_epochs+1):
    # ------- Train -------
    model.train()
    train_loss, true_train, pred_train = 0.0, [], []
    for student_flows_list, video_rgb, label in tqdm(train_loader, desc="Train"):
        video_rgb     = video_rgb.to(device)            # [1, 3, T, H, W]
        label         = label.to(device)     
        optimizer.zero_grad()
        # print(student_flows.shape)
        # print(video_rgb.shape)
         # Process each sample in the batch
        batch_logits = []
        for i, student_flows in enumerate(student_flows_list):
            student_flows = student_flows.to(device)
            single_rgb = video_rgb[i:i+1]       # [1, 3, T, H, W]
            logits = model(student_flows, single_rgb)  # [num_classes]
            batch_logits.append(logits)
        
        # Stack logits from all samples in batch
        batch_logits = torch.stack(batch_logits, dim=0)  # [batch_size, num_classes]
        # Compute loss
        loss = criterion(batch_logits, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred_train.extend(batch_logits.argmax(dim=1).cpu().tolist())
        true_train.extend(label.cpu().tolist())


    train_loss /= len(train_loader)
    acc_train = accuracy_score(true_train, pred_train)
    prec_train  = precision_score(true_train, pred_train, average='weighted', zero_division=0)
    rec_train   = recall_score(true_train, pred_train, average='weighted', zero_division=0)
    f1_train    = f1_score(true_train, pred_train, average='weighted', zero_division=0)
    
    train_losses.append(train_loss)
    train_accuracies.append(acc_train)
    train_f1_scores.append(f1_train)
    current_lr = optimizer.param_groups[0]['lr']
    # Log training metrics
    pd.DataFrame([[
        'train', epoch, acc_train, prec_train, rec_train, f1_train, train_loss, current_lr
    ]], columns=['Split','Epoch','Accuracy','Precision','Recall','F1','Loss', 'LR']
    ).to_csv(metrics_csv, mode='a', header=False, index=False)

    # ------- Validate -------
    model.eval()
    val_loss, true_val, pred_val = 0.0, [], []
    with torch.no_grad():
        for student_flows_list, video_rgb, label in tqdm(val_loader, desc="Val"):
            video_rgb     = video_rgb.to(device)            # [1, 3, T, H, W]
            label         = label.to(device)     
            # Process each sample in the batch
            batch_logits = []
            for i, student_flows in enumerate(student_flows_list):
                student_flows = student_flows.to(device)      # [1, N, T, 2, h, w]
                single_rgb = video_rgb[i:i+1]       # [1, 3, T, H, W]
                logits = model(student_flows, single_rgb)  # [num_classes]
                batch_logits.append(logits)
            
            # Stack logits from all samples in batch
            batch_logits = torch.stack(batch_logits, dim=0)  # [batch_size, num_classes]
          
            # print(f"Raw logits sample: {batch_logits[0]}")
            # print(f"Softmax probs: {torch.softmax(batch_logits[0], dim=0)}")
            # Compute loss
            loss = criterion(batch_logits, label)

            val_loss += loss.item()
            pred_val.extend(batch_logits.argmax(dim=1).cpu().tolist())
            true_val.extend(label.cpu().tolist())

    val_loss /= len(val_loader)
    acc_val = accuracy_score(true_val, pred_val)
    prec_val  = precision_score(true_val, pred_val, average='weighted', zero_division=0)

    # pred_counts = np.bincount(pred_val, minlength=num_classes)
    # print(f"Epoch {epoch} predictions: {[f'{idx_to_class[i]}: {count}' for i, count in enumerate(pred_counts)]}")
    rec_val   = recall_score(true_val, pred_val, average='weighted', zero_division=0)
    f1_val    = f1_score(true_val, pred_val, average='weighted', zero_division=0)
    
    val_losses.append(val_loss)
    val_accuracies.append(acc_val)
    val_f1_scores.append(f1_val)
    
    # Log validation metrics
    pd.DataFrame([[
        'val', epoch, acc_val, prec_val, rec_val, f1_val, val_loss, current_lr
    ]], columns=['Split','Epoch','Accuracy','Precision','Recall','F1','Loss','LR']
    ).to_csv(metrics_csv, mode='a', header=False, index=False)
    scheduler.step(val_loss)
    # print metrics
    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f}, Acc: {acc_train:.4f}, "
        f"Prec: {prec_train:.4f}, Rec: {rec_train:.4f}, F1: {f1_train:.4f} | "
        f"Val Loss: {val_loss:.4f}, Acc: {acc_val:.4f}, "
        f"Prec: {prec_val:.4f}, Rec: {rec_val:.4f}, F1: {f1_val:.4f}"
    )
    epoch_done+=1
    improved = (f1_val > best_f1 + min_delta) or (
        abs(f1_val - best_f1) <= min_delta and val_loss < best_loss  # tie-break by lower loss
    )

    if improved:
        best_f1, best_loss = f1_val, val_loss
        wait = 0
        torch.save(model.state_dict(), os.path.join(results_path, 'best_model.pth'))
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch} (no F1 improvement).")
            break

# ─── Final Test ────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(os.path.join(results_path, 'best_model.pth')))
model.eval()
test_loss, true_test, pred_test = 0.0, [], []

with torch.no_grad():
    for student_flows_list, video_rgb, label in tqdm(test_loader, desc="Test"):
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

test_loss /= len(test_loader)

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

# Accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch_done+1), train_accuracies, label='Train Accuracy', color='blue')
plt.plot(range(1, epoch_done+1), val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig(os.path.join(results_path, 'accuracy_curve.png'))

# F1 score curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch_done+1), train_f1_scores, label='Train F1', color='blue')
plt.plot(range(1, epoch_done+1), val_f1_scores, label='Validation F1', color='green')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score Curve')
plt.legend()
plt.savefig(os.path.join(results_path, 'f1_curve.png'))

# Loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch_done+1), train_losses, label='Train Loss', color='blue')
plt.plot(range(1, epoch_done+1), val_losses, label='Validation Loss', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig(os.path.join(results_path, 'loss_curve.png'))

# ─── Confusion Matrix ───────────────────────────────────────────────────────────
# After test evaluation, calculate confusion matrix
cm = confusion_matrix(true_test, pred_test)
cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names)

plt.figure(figsize=(8, 6))
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_path, 'confusion_matrix.png'))

print("Training complete.")