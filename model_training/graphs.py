# graphs.py (NO ACCURACY/LOSS) — Windows UTF-8 fixed
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "models/disease_model_gpu.h5"
VAL_DIR = "dataset/validation"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# LOAD MODEL
# -------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# -------------------------
# VALIDATION GENERATOR
# -------------------------
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_labels = list(val_generator.class_indices.keys())
n_classes = len(class_labels)
print("Detected classes:", class_labels)

# -------------------------
# PREDICT
# -------------------------
print("Running predictions on validation set...")
steps = int(np.ceil(val_generator.samples / val_generator.batch_size))
pred_probs = model.predict(val_generator, steps=steps, verbose=1)
pred_probs = pred_probs[:val_generator.samples]   # Trim extra rows
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = val_generator.classes

# -------------------------
# CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_path, dpi=300)
plt.close()

print("Saved:", cm_path)

# -------------------------
# CLASSIFICATION REPORT
# -------------------------
report = classification_report(true_classes, pred_classes, target_names=class_labels)
report_path = os.path.join(OUT_DIR, "classification_report.txt")

with open(report_path, "w", encoding="utf-8", errors="ignore") as f:
    f.write("CLASSIFICATION REPORT\n")
    f.write("=====================\n\n")
    f.write(report)

print("Saved:", report_path)

# -------------------------
# ROC CURVE
# -------------------------
y_true_bin = label_binarize(true_classes, classes=list(range(n_classes)))

# Fix shape mismatch
if pred_probs.shape[1] != y_true_bin.shape[1]:
    desired = y_true_bin.shape[1]
    if pred_probs.shape[1] > desired:
        pred_probs = pred_probs[:, :desired]
    else:
        pad = np.zeros((pred_probs.shape[0], desired - pred_probs.shape[1]))
        pred_probs = np.hstack([pred_probs, pad])

plt.figure(figsize=(8,6))

if n_classes == 2:
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), pred_probs[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

else:
    fpr = {}; tpr = {}; roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print("Plotting ROC curves...")
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro (AUC={roc_auc['micro']:.3f})", linestyle=":")

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_labels[i]} (AUC={roc_auc[i]:.3f})")

plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(fontsize=8)
roc_path = os.path.join(OUT_DIR, "roc_curve.png")
plt.tight_layout()
plt.savefig(roc_path, dpi=300)
plt.close()

print("Saved:", roc_path)

# -------------------------
# PRECISION–RECALL CURVE
# -------------------------
plt.figure(figsize=(8,6))

if n_classes == 2:
    precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), pred_probs[:,1])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")

else:
    precision = {}; recall = {}; pr_auc = {}

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], pred_probs[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
        plt.plot(recall[i], precision[i], label=f"{class_labels[i]} (AUC={pr_auc[i]:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend(fontsize=8)
pr_path = os.path.join(OUT_DIR, "pr_curve.png")
plt.tight_layout()
plt.savefig(pr_path, dpi=300)
plt.close()

print("Saved:", pr_path)

# -------------------------
# MODEL ARCHITECTURE (UTF-8 safe)
# -------------------------
try:
    from tensorflow.keras.utils import plot_model
    arch_path = os.path.join(OUT_DIR, "model_architecture.png")
    plot_model(model, to_file=arch_path, show_shapes=True, show_layer_names=True)
    print("Saved:", arch_path)

except Exception:
    # fallback to text summary
    summary_path = os.path.join(OUT_DIR, "model_summary.txt")
    with open(summary_path, "w", encoding="utf-8", errors="ignore") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))
    print("Saved summary text:", summary_path)

# -------------------------
# COMBINED FIGURE (No accuracy/loss)
# -------------------------
plt.figure(figsize=(12,5))

# Confusion Matrix
plt.subplot(1,2,1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels, cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# ROC
plt.subplot(1,2,2)
if n_classes == 2:
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
else:
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro AUC={roc_auc['micro']:.3f}")
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=class_labels[i])

plt.plot([0,1],[0,1],"k--")
plt.title("ROC Curve")
plt.legend(fontsize=8)

combined_path = os.path.join(OUT_DIR, "combined_figure.png")
plt.tight_layout()
plt.savefig(combined_path, dpi=300)
plt.close()

print("Saved:", combined_path)

print("\n🎉 ALL DONE — Your graphs are in the 'plots/' folder!")
