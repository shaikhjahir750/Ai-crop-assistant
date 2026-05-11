import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm
import random
import numpy as np
from collections import Counter
from torch.utils.data import WeightedRandomSampler

# ---------------------------------------------------------
# 1. Setup & Reproducibility
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATASET_DIR = Path("NewPlantDiseaseDataset")
PLOTS_DIR = Path("plant disease detection plots")
MODELS_DIR = Path("models")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Custom Dataset Loader (Handles nested folders)
# ---------------------------------------------------------
class ArrangedPlantDiseaseDataset(Dataset):
    """
    Dynamically loads images and prevents data leakage by merging physical
    'train' and 'valid' folders, then logically splitting the dataset
    based on the base image UUID (80% train / 20% valid split).
    """
    def __init__(self, root_dir: Path, split: str = 'train', transform=None, ignore_classes=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.ignore_classes = set(ignore_classes) if ignore_classes else set()
        self.samples = []
        
        classes_set = set()
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        class_to_uuids = {}
        uuid_to_paths = {}
        
        # 1. Read all images across BOTH train and valid to fix data leakage
        for target_dir in ['train', 'valid']:
            target_path = self.root_dir / target_dir
            if not target_path.exists():
                continue
                
            for img_path in target_path.rglob('*.*'):
                if img_path.suffix.lower() in valid_extensions:
                    class_name = img_path.parent.name
                    if class_name in self.ignore_classes:
                        continue
                    
                    # Extract base image UUID (everything before '___')
                    filename = img_path.name
                    uuid = filename.split('___')[0] if '___' in filename else filename
                    
                    if class_name not in class_to_uuids:
                        class_to_uuids[class_name] = set()
                        
                    class_to_uuids[class_name].add(uuid)
                    
                    if uuid not in uuid_to_paths:
                        uuid_to_paths[uuid] = []
                    uuid_to_paths[uuid].append((str(img_path), class_name))
                    classes_set.add(class_name)
                    
        # 2. Deterministically split UUIDs to ensure 80/20 train/valid separation
        for class_name, uuids in class_to_uuids.items():
            sorted_uuids = sorted(list(uuids))
            
            # Use fixed seed to guarantee consistent splits every time
            rng = random.Random(42)
            rng.shuffle(sorted_uuids)
            
            split_idx = int(0.8 * len(sorted_uuids))
            
            if split == 'train':
                selected_uuids = sorted_uuids[:split_idx]
            elif split == 'valid':
                selected_uuids = sorted_uuids[split_idx:]
            else:
                # Fallback for 'test' if requested
                selected_uuids = sorted_uuids
                
            for uuid in selected_uuids:
                self.samples.extend(uuid_to_paths[uuid])
                
        self.classes = sorted(list(classes_set))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback for corrupted images
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        label = self.class_to_idx[class_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ---------------------------------------------------------
# 3. Model Architecture (Transfer Learning & Dropout)
# ---------------------------------------------------------
def create_model(num_classes: int):
    """
    Uses EfficientNet-B3 as a backbone for transfer learning.
    Upgraded classifier head with BatchNorm + double Dropout to combat the
    severe overfitting (Train 99% vs Val 78%) observed in previous runs.
    """
    weights = models.EfficientNet_B3_Weights.DEFAULT
    model = models.efficientnet_b3(weights=weights)
    
    # Freeze the backbone features initially to prevent destructive updates
    for param in model.features.parameters():
        param.requires_grad = False
        
    in_features = model.classifier[1].in_features
    
    # Upgraded classifier: BN -> Dropout(0.5) -> FC(512) -> BN -> Dropout(0.4) -> FC(num_classes)
    # Two dropout layers + batch normalization acts as a much stronger regularizer
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes)
    )
    
    return model

# ---------------------------------------------------------
# 4. Training Engine with Anti-Overfitting Techniques
# ---------------------------------------------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Single timestamp for this entire run — used to name the model, class indices, and all plots
    RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {RUN_TIMESTAMP}")
    
    # --- Hyperparameters ---
    BATCH_SIZE = 32
    EPOCHS = 10           # Early stopping governs, not this cap
    LEARNING_RATE = 1e-3  # Head-only LR (frozen backbone phase)
    FINETUNE_LR = 5e-5    # Moderate LR for fine-tuning — previous 5e-6 was too low (underfitting)
    WEIGHT_DECAY = 1e-3   # Balanced L2 — 1e-2 was too strong (caused val to plateau at 58%)
    PATIENCE = 7          # Give model time to improve after unfreeze
    
    # --- Data Augmentation ---
    # Stronger data augmentation to artificially expand the dataset and prevent overfitting
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), # Increased crop variance
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5), # Added vertical flip
        transforms.RandomRotation(30), # Increased rotation variance
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # More color variance
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0) # Hides parts of the image to force the model to learn broader features
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # --- Load Datasets ---
    print("Loading datasets...")
    
    # No need to ignore classes anymore! NewPlantDiseaseDataset has a massive, proper validation set.
    train_dataset = ArrangedPlantDiseaseDataset(DATASET_DIR, split='train', transform=train_transform)
    valid_dataset = ArrangedPlantDiseaseDataset(DATASET_DIR, split='valid', transform=valid_transform)
    
    num_classes = len(train_dataset.classes)
    print(f"Found {len(train_dataset)} training images across {num_classes} classes.")
    
    # Ensure validation uses the same class indices as train
    valid_dataset.class_to_idx = train_dataset.class_to_idx
    
    # Save class indices — named with the run timestamp so it matches its model
    with open(MODELS_DIR / f'class_indices_{RUN_TIMESTAMP}.json', 'w') as f:
        json.dump(train_dataset.class_to_idx, f, indent=4)
        
    # --- Handle Class Imbalance (Weighted Sampler) ---
    train_targets = [train_dataset.class_to_idx[s[1]] for s in train_dataset.samples]
    class_counts = Counter(train_targets)
    weights = [1.0 / class_counts[t] for t in train_targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # --- Initialize Model, Loss, Optimizer ---
    model = create_model(num_classes).to(device)
    
    # Note: Removed class weights from CrossEntropyLoss to prevent "double-dipping" 
    # since we are already using WeightedRandomSampler to balance the batches!
    
    # Label Smoothing adds regularization to prevent overconfidence (increased to 0.15 for anti-overfitting)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    
    # Only optimize the classifier head at first
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Scheduler: Reduces learning rate when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)
    
    # --- Training Loop with Early Stopping ---
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # Initialize Mixed Precision Scaler for RTX 3050 speedup
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # Fine-Tuning Phase: Unfreeze last 3 blocks at epoch 4
        # Previous run: 5e-6 LR was too conservative -> val stuck at 58%
        # This run: 5e-5 LR + unfreeze 3 blocks (6,7,8) for more expressive power
        if epoch == 4:
            print("Partially unfreezing backbone (last 3 feature blocks) for fine-tuning...")
            for i, block in enumerate(model.features):
                if i >= 6:
                    for param in block.parameters():
                        param.requires_grad = True
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY
            )
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
            
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()
            train_total += labels.size(0)
            
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(valid_loader, desc="Validation")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(valid_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Learning Rate Scheduler Step
        scheduler.step(avg_val_loss)
        
        # Early Stopping & Model Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            model_path = MODELS_DIR / f'disease_detection_model_{RUN_TIMESTAMP}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model checkpoint: disease_detection_model_{RUN_TIMESTAMP}.pth")
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter} out of {PATIENCE}")
            if early_stop_counter >= PATIENCE:
                print("Early stopping triggered! Training halted to prevent overfitting.")
                break
                
    # --- Load Best Checkpoint & Generate Evaluation Reports ---
    # IMPORTANT: Always evaluate using the best saved checkpoint, not the final epoch weights.
    # This ensures the report reflects peak performance, not a potentially overfit final state.
    print("\nLoading best model checkpoint for final evaluation...")
    model.load_state_dict(torch.load(MODELS_DIR / f'disease_detection_model_{RUN_TIMESTAMP}.pth', map_location=device))
    model.eval()
    all_preds_best = []
    all_labels_best = []
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Final Evaluation"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds_best.extend(preds.cpu().numpy())
            all_labels_best.extend(labels.cpu().numpy())
    
    generate_plots(history, all_labels_best, all_preds_best, train_dataset.classes, RUN_TIMESTAMP)

def generate_plots(history, labels, preds, class_names, timestamp):
    print("Generating evaluation plots...")
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / f'loss_curve_{timestamp}.png')
    plt.close()
    
    # Accuracy Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / f'accuracy_curve_{timestamp}.png')
    plt.close()
    
    # 2. Confusion Matrix
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'confusion_matrix_{timestamp}.png')
    plt.close()
    
    # 3. Classification Report
    report = classification_report(labels, preds, labels=range(len(class_names)), target_names=class_names, zero_division=0)
    with open(PLOTS_DIR / f'evaluation_report_{timestamp}.txt', 'w') as f:
        f.write(f"Model Evaluation Report ({timestamp})\n")
        f.write("=========================================\n\n")
        f.write(report)
        
    print(f"All plots and reports successfully saved to '{PLOTS_DIR}' with timestamp '{timestamp}'.")

if __name__ == '__main__':
    train_model()
