import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

def main():
    # Set seaborn theme for beautiful plots
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'research report/run_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    # Load data
    print("Loading data...")
    data_path = 'data/Train_Dataset.xlsx'
    df = pd.read_excel(data_path)

    # Separate features and target
    X = df.drop('Crop', axis=1)
    y = df['Crop']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': GaussianNB()
    }

    # 1. Train and Evaluate
    results = []
    cms = {}
    reports = {}

    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        cms[name] = confusion_matrix(y_test, y_pred)
        reports[name] = classification_report(y_test, y_pred, zero_division=0)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print("Saved model_comparison.csv")

    with open(os.path.join(output_dir, 'classification_reports.txt'), 'w') as f:
        for name, report in reports.items():
            f.write(f"=== {name} ===\n")
            f.write(report)
            f.write("\n\n")
    print("Saved classification_reports.txt")

    # 2. Plot Confusion Matrices
    print("Plotting confusion matrices...")
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, 7 * rows))
    axes = axes.flatten()

    unique_classes = sorted(y_test.unique())

    for i, (name, cm) in enumerate(cms.items()):
        # Draw heatmap
        sns.heatmap(cm, annot=False, cmap='Blues', ax=axes[i], cbar=True, 
                    xticklabels=unique_classes, yticklabels=unique_classes)
        axes[i].set_title(f'{name} Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
        axes[i].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('True Label', fontsize=12, fontweight='bold')
        axes[i].tick_params(axis='x', rotation=90, labelsize=9)
        axes[i].tick_params(axis='y', rotation=0, labelsize=9)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()
    print("Saved confusion_matrices.png")

    # 3. Plot Evaluation Metrics
    print("Plotting evaluation metrics...")
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(x='Model', y=metric, data=results_df, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Model Comparison: {metric}', fontsize=15, fontweight='bold')
        axes[i].set_ylim(0, 1.15) # Leave room for the labels
        axes[i].tick_params(axis='x', rotation=45, labelsize=11)
        axes[i].tick_params(axis='y', labelsize=11)
        axes[i].set_ylabel(metric, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('', fontsize=12)
        
        # Add values on top of bars
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt='%.3f', padding=3, size=10, fontweight='bold')

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'), dpi=300)
    plt.close()
    print("Saved evaluation_metrics.png")

    # 4. Plot Train-Test Curves (Learning Curves)
    print("Plotting train-test curves...")
    fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        print(f"Calculating learning curve for {name}...")
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring='accuracy'
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        axes[i].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score", linewidth=2, markersize=8)
        axes[i].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score", linewidth=2, markersize=8)
        axes[i].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.15, color="r")
        axes[i].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.15, color="g")
        
        axes[i].set_title(f'{name} Learning Curve', fontsize=16, fontweight='bold', pad=10)
        axes[i].set_xlabel('Training Examples', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
        axes[i].tick_params(axis='both', labelsize=10)
        axes[i].legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
        axes[i].grid(True, linestyle='--', alpha=0.7)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'train_test_curves.png'), dpi=300)
    plt.close()
    print("Saved train_test_curves.png")

    print(f"Evaluation complete. All results saved in the '{output_dir}' directory.")

if __name__ == '__main__':
    main()
