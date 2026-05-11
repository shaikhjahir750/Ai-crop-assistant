import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm
import random
import numpy as np
from collections import Counter

# ---------------------------------------------------------
# 1. Setup & Reproducibility
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# PlantDoc dataset paths
DATASET_DIR   = Path("PlantDoc-Dataset")
PLOTS_DIR     = Path("plantdoc detection plots")
MODELS_DIR    = Path("models")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Dataset Loader
# ---------------------------------------------------------
class PlantDocDataset(Dataset):
    """
    Loads images from PlantDoc-Dataset/<split>/<class_name>/ structure.
    Supports an optional ignore_classes list.
    """
    def __init__(self, root_dir: Path, split: str = 'train',
                 transform=None, class_to_idx: dict = None, ignore_classes: list = None):
        self.root_dir = root_dir / split
        self.transform = transform
        self.ignore_classes = set(ignore_classes or [])
        self.samples = []

        classes_set = set()
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}

        for img_path in self.root_dir.rglob('*'):
            if img_path.suffix in valid_extensions and img_path.is_file():
                class_name = img_path.parent.name
                if class_name in self.ignore_classes:
                    continue
                self.samples.append((str(img_path), class_name))
                classes_set.add(class_name)

        if class_to_idx is not None:
            # Use the mapping provided (for test/val set consistency)
            self.class_to_idx = class_to_idx
            self.classes = sorted(class_to_idx.keys())
        else:
            self.classes = sorted(list(classes_set))
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        label = self.class_to_idx[class_name]
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------------------------------------------------------
# 3. Model Architecture — EfficientNet-B4 with robust head
# ---------------------------------------------------------
def create_model(num_classes: int):
    """
    EfficientNet-B4 backbone (stronger than B3 for small datasets like PlantDoc).
    Head: BN -> Dropout(0.5) -> FC(512) -> ReLU -> BN -> Dropout(0.3) -> FC(num_classes)
    """
    weights = models.EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=weights)

    # Freeze all backbone layers initially
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )

    return model

# ---------------------------------------------------------
# 4. Mixup Augmentation — Blends two training samples together
#    Forces the model to learn smooth decision boundaries
#    especially effective for small datasets like PlantDoc (~2500 imgs)
# ---------------------------------------------------------
def mixup_data(x, y, alpha=0.3):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ---------------------------------------------------------
# 5. Main Training Function
# ---------------------------------------------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Unique run ID — used to name model, class indices, and all plots
    RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {RUN_TIMESTAMP}")

    # ── Hyperparameters ────────────────────────────────────────────────────────
    BATCH_SIZE       = 16    # Small batch: PlantDoc has ~2500 images — smaller batches = more gradient updates
    EPOCHS           = 40    # More epochs: small dataset needs longer training
    HEAD_LR          = 5e-4  # LR for the classifier head (frozen backbone phase)
    FINETUNE_LR      = 3e-5  # LR for fine-tuning the backbone (moderate, not too destructive)
    WEIGHT_DECAY     = 5e-4  # L2 regularization
    PATIENCE         = 10    # Early stopping patience
    UNFREEZE_EPOCH   = 8     # Start fine-tuning backbone after this many head-only epochs
    MIXUP_ALPHA      = 0.3   # Mixup strength (0 = disabled)
    IMAGE_SIZE       = 224

    # ── Data Transforms ───────────────────────────────────────────────────────
    # Heavy augmentation essential for PlantDoc — real-world, low-count dataset
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),      # Simulates aged/B&W photos occasionally
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Load Datasets ─────────────────────────────────────────────────────────
    print("Loading PlantDoc dataset...")
    train_dataset = PlantDocDataset(DATASET_DIR, split='train', transform=train_transform)
    test_dataset  = PlantDocDataset(DATASET_DIR, split='test',  transform=test_transform,
                                    class_to_idx=train_dataset.class_to_idx)

    num_classes = len(train_dataset.classes)
    print(f"Classes ({num_classes}): {train_dataset.classes}")
    print(f"Train: {len(train_dataset)} images | Test: {len(test_dataset)} images")

    # Save class indices matched to this run's model
    idx_path = MODELS_DIR / f'plantdoc_class_indices_{RUN_TIMESTAMP}.json'
    with open(idx_path, 'w') as f:
        json.dump(train_dataset.class_to_idx, f, indent=4)
    print(f"Class indices saved: {idx_path}")

    # ── Weighted Random Sampler (handles class imbalance) ─────────────────────
    train_targets = [train_dataset.class_to_idx[s[1]] for s in train_dataset.samples]
    class_counts  = Counter(train_targets)
    sample_weights = [1.0 / class_counts[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model + Loss + Optimizer ──────────────────────────────────────────────
    model = create_model(num_classes).to(device)

    # Label smoothing (0.1) prevents overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1: Only train the classifier head
    optimizer = optim.AdamW(model.classifier.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)

    # Cosine Annealing: smoothly decays LR over training for better convergence
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # Mixed precision for RTX 3050 speedup
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_f1   = 0.0
    best_val_acc  = 0.0
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'train_acc': [], 'val_acc': []}

    print("\n" + "="*60)
    print("Starting PlantDoc Training")
    print("="*60)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        # ── Phase 2: Gradually unfreeze backbone ──────────────────────────────
        if epoch == UNFREEZE_EPOCH:
            print(f"Unfreezing last 4 backbone blocks for fine-tuning (LR={FINETUNE_LR})...")
            # EfficientNet-B4 features: blocks 0-8, unfreeze 5,6,7,8
            for i, block in enumerate(model.features):
                if i >= 5:
                    for param in block.parameters():
                        param.requires_grad = True

            # Create two param groups: backbone (low LR) and head (high LR)
            optimizer = optim.AdamW([
                {'params': filter(lambda p: p.requires_grad,
                                  [p for n, p in model.features.named_parameters()]),
                 'lr': FINETUNE_LR},
                {'params': model.classifier.parameters(), 'lr': HEAD_LR * 0.5}
            ], weight_decay=WEIGHT_DECAY)
            remaining_epochs = EPOCHS - epoch
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-7)

        # ── Training Phase ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc="Training", ncols=90)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Apply Mixup augmentation
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=MIXUP_ALPHA)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            scaler.scale(loss).backward()
            # Gradient clipping prevents exploding gradients during fine-tuning
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            # Accuracy based on primary label (label_a, the stronger component from lam)
            train_correct += torch.sum(preds == labels_a).item()
            train_total += labels.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # ── Validation Phase ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Validation", ncols=90):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

        # ── Save Best Model (tracked by Macro F1 — better for imbalanced data) ─
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_val_acc = val_acc
            early_stop_counter = 0
            model_path = MODELS_DIR / f'plantdoc_disease_model_{RUN_TIMESTAMP}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"✓ New best model saved: plantdoc_disease_model_{RUN_TIMESTAMP}.pth "
                  f"(F1={val_f1:.4f}, Acc={val_acc:.4f})")
        else:
            early_stop_counter += 1
            print(f"EarlyStopping: {early_stop_counter}/{PATIENCE} (best F1={best_val_f1:.4f})")
            if early_stop_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # ── Final Evaluation using Best Checkpoint ────────────────────────────────
    print(f"\nBest model: Acc={best_val_acc:.4f} | Macro-F1={best_val_f1:.4f}")
    print("Loading best checkpoint for final evaluation report...")

    model.load_state_dict(torch.load(
        MODELS_DIR / f'plantdoc_disease_model_{RUN_TIMESTAMP}.pth', map_location=device))
    model.eval()

    final_preds = []
    final_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Evaluation"):
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())

    generate_plots(history, final_labels, final_preds, train_dataset.classes, RUN_TIMESTAMP)


# ---------------------------------------------------------
# 6. Evaluation Plots & Reports
# ---------------------------------------------------------
def generate_plots(history, labels, preds, class_names, timestamp):
    print("\nGenerating evaluation plots and reports...")

    # ── Loss Curve ────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', color='steelblue')
    plt.plot(history['val_loss'],   label='Val Loss',   color='darkorange')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'loss_curve_{timestamp}.png', dpi=150)
    plt.close()

    # ── Accuracy Curve ────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy', color='steelblue')
    plt.plot(history['val_acc'],   label='Val Accuracy',   color='darkorange')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'accuracy_curve_{timestamp}.png', dpi=150)
    plt.close()

    # ── Macro F1 Curve ────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_f1'], label='Val Macro-F1', color='green')
    plt.title('Validation Macro-F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'f1_curve_{timestamp}.png', dpi=150)
    plt.close()

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    plt.figure(figsize=(22, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'confusion_matrix_{timestamp}.png', dpi=150)
    plt.close()

    # ── Per-Class F1 Bar Chart ────────────────────────────────────────────────
    report_dict = {}
    report = classification_report(labels, preds,
                                   labels=range(len(class_names)),
                                   target_names=class_names,
                                   output_dict=True,
                                   zero_division=0)
    f1_scores = [report[cls]['f1-score'] for cls in class_names]
    colors = ['green' if f >= 0.7 else 'orange' if f >= 0.5 else 'red' for f in f1_scores]

    plt.figure(figsize=(14, 7))
    bars = plt.barh(class_names, f1_scores, color=colors)
    plt.axvline(x=0.7, color='black', linestyle='--', linewidth=1, label='0.70 threshold')
    plt.title('Per-Class F1 Score')
    plt.xlabel('F1 Score')
    plt.xlim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'per_class_f1_{timestamp}.png', dpi=150)
    plt.close()

    # ── Text Classification Report ────────────────────────────────────────────
    report_text = classification_report(labels, preds,
                                        labels=range(len(class_names)),
                                        target_names=class_names,
                                        zero_division=0)
    report_path = PLOTS_DIR / f'evaluation_report_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write(f"PlantDoc Model Evaluation Report ({timestamp})\n")
        f.write("=" * 50 + "\n\n")
        f.write(report_text)

    print(f"\nAll plots and reports saved to '{PLOTS_DIR}/' with timestamp '{timestamp}'.")
    print(f"Report: {report_path}")


if __name__ == '__main__':
    train_model()
