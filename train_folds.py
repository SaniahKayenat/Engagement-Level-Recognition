#Imports
import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets.flow_video_dataset import FlowVideoDataset
from models.flow_mlp import TransformerFlowExtractor
from models.i3d import FeatureExtractor
from models.student_model import StudentGlobalNet

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

# Hyperparameters
tube_length   = 150
flow_root     = 'Optical Flow Feature Folder'
videos_root   = 'Dataset Folder'
csv_path      = 'CSV Path'
resize        = (128,128)
num_classes   = 3
track_id_min  = 1
track_id_max  = 5
num_epochs    = 100
num_folds     = 5
patience      = 20
lr            = 1e-4
results_path  = 'Result Folder'
os.makedirs(results_path, exist_ok=True)
metrics_csv = os.path.join(results_path, 'metrics_five_folds.csv')

# Initialize dataset 
dataset = FlowVideoDataset(
    csv_path=csv_path,
    flow_root=flow_root,
    videos_root=videos_root,
    tube_length=150,
    flow_size=(128, 128),
    resize=(128, 128),
    track_id_min=1,
    track_id_max=5
)

# Create splits for k-fold cross-validation 
kf = KFold(n_splits=2, shuffle=True, random_state=42)
indices = np.arange(len(dataset))

if not os.path.exists(metrics_csv):
    pd.DataFrame(
        columns=['Fold', 'Accuracy', 'Precision', 'Recall', 'F1', 'Loss']
    ).to_csv(metrics_csv, index=False)

# Models
flow_ex = TransformerFlowExtractor().to(device)
i3d_ex = FeatureExtractor().to(device)

# Global metrics for each fold
all_metrics = []

for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
    print(f"Training fold {fold + 1}")

    # Split data 
    trainval_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=42, stratify=[dataset[i][2] for i in train_idx])

    # Create DataLoader 
    train_loader = DataLoader(Subset(dataset, trainval_idx), batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=1, shuffle=False, num_workers=4)

    # fold folder
    fold_result_path = os.path.join(results_path, f'fold_{fold + 1}')
    os.makedirs(fold_result_path, exist_ok=True)

    # Model setup 
    model = StudentGlobalNet(flow_ex, i3d_ex, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_labels = []
    for _, _, labels in tqdm(train_loader):
        train_labels.extend(labels.tolist())

    class_counts   = np.bincount(train_labels, minlength=num_classes) 
    total_samples  = class_counts.sum()
    eps = 1e-8
    class_weights  = total_samples / (num_classes * (class_counts + eps))
    class_weights  = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(class_weights)

    best_f1, best_loss = -1.0, float('inf')
    min_delta = 1e-4
    wait = 0
      
    train_accuracies, val_accuracies = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1_scores, val_f1_scores = [], []
    # Training loop 
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, true_train, pred_train = 0.0, [], []

        for student_flows_list, video_rgb, label in tqdm(train_loader, desc="Train"):
            video_rgb = video_rgb.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            batch_logits = []
            for i, student_flows in enumerate(student_flows_list):
                student_flows = student_flows.to(device)
                single_rgb = video_rgb[i:i + 1]
                logits = model(student_flows, single_rgb)
                batch_logits.append(logits)

            batch_logits = torch.stack(batch_logits, dim=0)
            loss = criterion(batch_logits, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred_train.extend(batch_logits.argmax(dim=1).cpu().tolist())
            true_train.extend(label.cpu().tolist())

        train_loss /= len(train_loader)
        acc_train = accuracy_score(true_train, pred_train)
        prec_train = precision_score(true_train, pred_train, average='weighted', zero_division=0)
        rec_train = recall_score(true_train, pred_train, average='weighted', zero_division=0)
        f1_train = f1_score(true_train, pred_train, average='weighted', zero_division=0)
        train_accuracies.append(acc_train)
        train_precisions.append(prec_train)
        train_recalls.append(rec_train)
        train_f1_scores.append(f1_train)
        # Validation loop 
        model.eval()
        val_loss, true_val, pred_val = 0.0, [], []
        with torch.no_grad():
            for student_flows_list, video_rgb, label in tqdm(val_loader, desc="Val"):
                video_rgb = video_rgb.to(device)
                label = label.to(device)

                batch_logits = []
                for i, student_flows in enumerate(student_flows_list):
                    student_flows = student_flows.to(device)
                    single_rgb = video_rgb[i:i + 1]
                    logits = model(student_flows, single_rgb)
                    batch_logits.append(logits)

                batch_logits = torch.stack(batch_logits, dim=0)
                loss = criterion(batch_logits, label)

                val_loss += loss.item()
                pred_val.extend(batch_logits.argmax(dim=1).cpu().tolist())
                true_val.extend(label.cpu().tolist())

        val_loss /= len(val_loader)
        acc_val = accuracy_score(true_val, pred_val)
        prec_val = precision_score(true_val, pred_val, average='weighted', zero_division=0)
        rec_val = recall_score(true_val, pred_val, average='weighted', zero_division=0)
        f1_val = f1_score(true_val, pred_val, average='weighted', zero_division=0)
        val_accuracies.append(acc_val)
        val_precisions.append(prec_val)
        val_recalls.append(rec_val)
        val_f1_scores.append(f1_val)


        # Early stopping 
        improved = (f1_val > best_f1 + min_delta) or (abs(f1_val - best_f1) <= min_delta and val_loss < best_loss)
        if improved:
            best_f1, best_loss = f1_val, val_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(fold_result_path, 'best_model.pth'))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (no F1 improvement).")
                break
    # Plot and save training/validation curves 
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Val Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curve - Fold {fold + 1}')
    plt.legend()
    plt.savefig(os.path.join(fold_result_path, 'accuracy_curve.png'))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_f1_scores, label='Train F1', color='blue')
    plt.plot(range(1, num_epochs + 1), val_f1_scores, label='Val F1', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score Curve - Fold {fold + 1}')
    plt.legend()
    plt.savefig(os.path.join(fold_result_path, 'f1_curve.png'))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_precisions, label='Train Precision', color='blue')
    plt.plot(range(1, num_epochs + 1), val_precisions, label='Val Precision', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title(f'Precision Curve - Fold {fold + 1}')
    plt.legend()
    plt.savefig(os.path.join(fold_result_path, 'precision_curve.png'))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_recalls, label='Train Recall', color='blue')
    plt.plot(range(1, num_epochs + 1), val_recalls, label='Val Recall', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title(f'Recall Curve - Fold {fold + 1}')
    plt.legend()
    plt.savefig(os.path.join(fold_result_path, 'recall_curve.png'))
    plt.close()  
    # Testing phase 
    model.load_state_dict(torch.load(os.path.join(fold_result_path, 'best_model.pth')))
    model.eval()
    test_loss, true_test, pred_test = 0.0, [], []
    with torch.no_grad():
        for student_flows_list, video_rgb, label in tqdm(test_loader, desc="Test"):
            video_rgb = video_rgb.to(device)
            label = label.to(device)

            batch_logits = []
            for i, student_flows in enumerate(student_flows_list):
                student_flows = student_flows.to(device)
                single_rgb = video_rgb[i:i + 1]
                logits = model(student_flows, single_rgb)
                batch_logits.append(logits)

            batch_logits = torch.stack(batch_logits, dim=0)
            loss = criterion(batch_logits, label)

            test_loss += loss.item()
            pred_test.extend(batch_logits.argmax(dim=1).cpu().tolist())
            true_test.extend(label.cpu().tolist())

    # Calculate test metrics
    acc_test = accuracy_score(true_test, pred_test)
    prec_test = precision_score(true_test, pred_test, average='weighted', zero_division=0)
    rec_test = recall_score(true_test, pred_test, average='weighted', zero_division=0)
    f1_test = f1_score(true_test, pred_test, average='weighted', zero_division=0)

    print(f"\nFINAL TEST RESULTS for Fold {fold + 1}:")
    print(f"Test Accuracy:  {acc_test:.4f}")
    print(f"Test Precision: {prec_test:.4f}")
    print(f"Test Recall:    {rec_test:.4f}")
    print(f"Test F1-Score:  {f1_test:.4f}")
    print(f"Test Loss:      {test_loss:.4f}")

    # Log the test results 
    pd.DataFrame([[fold + 1, acc_test, prec_test, rec_test, f1_test, test_loss]],
                 columns=['Fold', 'Accuracy', 'Precision', 'Recall', 'F1', 'Loss']
                 ).to_csv(metrics_csv, mode='a', header=False, index=False)

    # Confusion matrix 
    cm = confusion_matrix(true_test, pred_test)
    cm_display = ConfusionMatrixDisplay(cm)
    cm_display.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(fold_result_path, 'confusion_matrix.png'))

    # Save test set results
    test_results = pd.DataFrame({
        'True Labels': true_test,
        'Predictions': pred_test
    })
    test_results.to_csv(os.path.join(fold_result_path, 'test_results.csv'), index=False)

    fold_metrics = {
        'fold': fold + 1,
        'accuracy': acc_test,
        'precision': prec_test,
        'recall': rec_test,
        'f1_score': f1_test,
        'loss': test_loss
    }
    all_metrics.append(fold_metrics)

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(metrics_csv, index=False)
