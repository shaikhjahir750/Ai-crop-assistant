import os
import json
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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
PLOTS_DIR = Path("disease detection plots")
MODELS_DIR = Path("models")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Custom Dataset Loader (Handles nested folders)
# ---------------------------------------------------------
class NewPlantDiseaseDataset(Dataset):
    """
    Dynamically loads images from a split folder (e.g., train or valid).
    It handles nested crop subfolders by recursively finding all images.
    """
    def __init__(self, root_dir: Path, split: str = 'train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        
        classes_set = set()
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Search for all images recursively
        for img_path in self.root_dir.rglob('*.*'):
            if img_path.suffix.lower() in valid_extensions:
                # Ensure the image belongs to the specified split
                if split in img_path.parts:
                    class_name = img_path.parent.name
                    self.samples.append((str(img_path), class_name))
                    classes_set.add(class_name)
                    
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
    Includes Dropout layer to prevent overfitting.
    """
    weights = models.EfficientNet_B3_Weights.DEFAULT
    model = models.efficientnet_b3(weights=weights)
    
    # Freeze the backbone features initially to prevent destructive updates
    for param in model.features.parameters():
        param.requires_grad = False
        
    in_features = model.classifier[1].in_features
    
    # Replace the classifier head, adding a Dropout layer (50%)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

# ---------------------------------------------------------
# 4. Training Engine with Anti-Overfitting Techniques
# ---------------------------------------------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Hyperparameters ---
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization
    PATIENCE = 5         # Early Stopping Patience
    
    # --- Data Augmentation ---
    # Helps prevent overfitting by artificially expanding the dataset
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # --- Load Datasets ---
    print("Loading datasets...")
    train_dataset = NewPlantDiseaseDataset(DATASET_DIR, split='train', transform=train_transform)
    valid_dataset = NewPlantDiseaseDataset(DATASET_DIR, split='valid', transform=valid_transform)
    
    num_classes = len(train_dataset.classes)
    print(f"Found {len(train_dataset)} training images across {num_classes} classes.")
    
    # Ensure validation uses the same class indices as train
    valid_dataset.class_to_idx = train_dataset.class_to_idx
    
    # Save class indices
    with open(MODELS_DIR / 'disease_class_indices.json', 'w') as f:
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
    
    # Label Smoothing adds regularization to prevent overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Only optimize the classifier head at first
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Scheduler: Reduces learning rate when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)
    
    # --- Training Loop with Early Stopping ---
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # Fine-Tuning Phase: Unfreeze the whole network after 5 epochs
        if epoch == 5:
            print("Unfreezing backbone for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            # Lower learning rate for fine-tuning
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)
            
        # Training
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
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
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)
        
        # Learning Rate Scheduler Step
        scheduler.step(avg_val_loss)
        
        # Early Stopping & Model Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / 'best_disease_model.pth')
            print("Saved new best model checkpoint.")
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter} out of {PATIENCE}")
            if early_stop_counter >= PATIENCE:
                print("Early stopping triggered! Training halted to prevent overfitting.")
                break
                
    # --- Generate Evaluation Reports & Plots ---
    generate_plots(history, all_labels, all_preds, train_dataset.classes)

def generate_plots(history, labels, preds, class_names):
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
    plt.savefig(PLOTS_DIR / 'loss_curve.png')
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
    plt.savefig(PLOTS_DIR / 'confusion_matrix.png')
    plt.close()
    
    # 3. Classification Report
    report = classification_report(labels, preds, labels=range(len(class_names)), target_names=class_names, zero_division=0)
    with open(PLOTS_DIR / 'evaluation_report.txt', 'w') as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n\n")
        f.write(report)
        
    print(f"All plots and reports successfully saved to '{PLOTS_DIR}'.")

if __name__ == '__main__':
    train_model()
