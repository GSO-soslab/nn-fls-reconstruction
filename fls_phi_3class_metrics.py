import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd
import os
from plot_test import extract_pcl_points_from_row

import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from full_three_stage_model import FullThreeStageModelCNN, BathymetryDataset

# Set global font sizes for labels and titles (but keep tick labels moderate)
plt.rcParams.update({
    'axes.labelsize': 20,      # X and Y labels (bigger)
    'axes.titlesize': 22,      # Plot titles (bigger)
    'legend.fontsize': 16,     # Legend text (bigger)
    'xtick.labelsize': 14,     # X-axis tick numbers (moderate)
    'ytick.labelsize': 14,     # Y-axis tick numbers (moderate)
})


def calculate_metrics_from_inference(model, test_loader, device):
    """Calculate 3-class metrics directly from model inference"""
    model.eval()
    all_preds = []
    all_ground_truth = []

    print("Running inference on dataset...")
    with torch.no_grad():
        for batch_idx, (intensities, ground_truth) in enumerate(test_loader):
            intensities = intensities.to(device)
            ground_truth = ground_truth.to(device)

            final_preds, _, _, _ = model(intensities)

            all_preds.append(final_preds.cpu().numpy())
            all_ground_truth.append(ground_truth.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    # Flatten and concatenate all batches
    preds = np.concatenate([p.flatten() for p in all_preds])
    gt = np.concatenate([g.flatten() for g in all_ground_truth])

    print(f"\nTotal predictions: {len(preds)}")
    print(f"Total ground truth: {len(gt)}")

    # Convert to 3-class labels
    gt_classes = classify_phi_values(gt)
    pred_classes = classify_phi_values(preds)

    return gt, preds, gt_classes, pred_classes


def classify_phi_values(phi_array):
    """
    Classify phi values into 3 categories:
    0: Valid measurement (not -10 or -20)
    1: No return (-10)
    2: Out of range/invalid (-20)
    """
    phi_array = np.array(phi_array)
    classes = np.zeros(len(phi_array), dtype=int)
    classes[phi_array == -10.0] = 1
    classes[phi_array == -20.0] = 2
    return classes


def calculate_3class_metrics(y_true, y_pred, class_names=['Valid', '(-10)', '(-20)']):
    """Calculate TP, FP, FN, TN for each of the 3 classes"""
    n_classes = len(class_names)

    print(f"\n{'='*90}")
    print(f"3-Class Classification Metrics (One-vs-Rest)")
    print(f"{'='*90}")
    print(f"{'Class':<20} {'TP':>10} {'FP':>10} {'FN':>10} {'TN':>10} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print(f"{'-'*110}")

    metrics_dict = {}

    for class_idx in range(n_classes):
        # One-vs-Rest binary classification for this class
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)

        TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_dict[class_names[class_idx]] = {
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'Precision': precision, 'Recall': recall, 'F1': f1
        }

        print(f"{class_names[class_idx]:<20} {TP:>10} {FP:>10} {FN:>10} {TN:>10} "
              f"{precision:>12.3f} {recall:>12.3f} {f1:>12.3f}")

    print(f"{'='*110}")
    print(f"\nOverall Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Total samples: {len(y_true)}")

    return metrics_dict


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate MSE and MAE only for valid phi values (excluding -10 and -20)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Create mask for valid measurements (not -10 or -20)
    valid_mask = (y_true != -10.0) & (y_true != -20.0) & (y_pred != -10.0) & (y_pred != -20.0)

    # Filter to only valid points
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    if len(y_true_valid) == 0:
        print("Warning: No valid data points found!")
        return None

    # Calculate metrics
    mse = np.mean((y_true_valid - y_pred_valid) ** 2)
    mae = np.mean(np.abs(y_true_valid - y_pred_valid))
    rmse = np.sqrt(mse)

    print(f"\n{'='*70}")
    print(f"Regression Metrics (Valid Data Points Only)")
    print(f"{'='*70}")
    print(f"Total valid points: {len(y_true_valid)} out of {len(y_true)} ({len(y_true_valid)/len(y_true)*100:.1f}%)")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"{'='*70}")

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'n_valid': len(y_true_valid),
        'n_total': len(y_true)
    }


def plot_3class_confusion_matrix(original_phis, predicted_phis, output_dir="./", title_suffix=""):
    """
    Create confusion matrix for 3-class phi classification
    """
    # Remove NaN values (but keep -10 and -20)
    orig = np.array(original_phis).flatten()
    pred = np.array(predicted_phis).flatten()

    valid_mask = ~np.isnan(orig) & ~np.isnan(pred)
    orig_valid = orig[valid_mask]
    pred_valid = pred[valid_mask]

    print(f"Total points for confusion matrix: {len(orig_valid)}")

    # Classify into 3 categories
    y_true = classify_phi_values(orig_valid)
    y_pred = classify_phi_values(pred_valid)

    # Count distribution
    print(f"\nGround Truth Distribution:")
    print(f"  Valid measurements: {np.sum(y_true == 0)} ({np.sum(y_true == 0)/len(y_true)*100:.1f}%)")
    print(f"  (-10):    {np.sum(y_true == 1)} ({np.sum(y_true == 1)/len(y_true)*100:.1f}%)")
    print(f"  (-20):      {np.sum(y_true == 2)} ({np.sum(y_true == 2)/len(y_true)*100:.1f}%)")

    print(f"\nPrediction Distribution:")
    print(f"  Valid measurements: {np.sum(y_pred == 0)} ({np.sum(y_pred == 0)/len(y_pred)*100:.1f}%)")
    print(f"  (-10):    {np.sum(y_pred == 1)} ({np.sum(y_pred == 1)/len(y_pred)*100:.1f}%)")
    print(f"  (-20):      {np.sum(y_pred == 2)} ({np.sum(y_pred == 2)/len(y_pred)*100:.1f}%)")

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    class_names = ['Valid', 'Partial Return', 'No-return']

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    ax1.set_xlabel('Predicted Class', fontsize=12)
    ax1.set_ylabel('Ground Truth Class', fontsize=12)
    ax1.set_title(f'Confusion Matrix - Counts {title_suffix}', fontsize=14)

    # Normalized (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Proportion'}, annot_kws={'size': 14})
    ax2.set_xlabel('Predicted Class', fontsize=12)
    ax2.set_ylabel('Ground Truth Class', fontsize=12)
    ax2.set_title(f'Confusion Matrix - Normalized {title_suffix}', fontsize=14)

    plt.tight_layout()
    filename = f"{output_dir}/confusion_matrix_3class{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Print classification report
    print("\n" + "="*70)
    print("Scikit-learn Classification Report:")
    print("="*70)
    print(classification_report(y_true, y_pred,
                                target_names=class_names,
                                zero_division=0))

    # Calculate detailed metrics with TP/FP/FN/TN
    metrics = calculate_3class_metrics(y_true, y_pred, class_names)

    return cm, cm_normalized, metrics


def plot_metrics_breakdown(metrics, output_dir="./", title_suffix=""):
    """
    Visualize TP/FP/FN/TN for each class
    """
    class_names = list(metrics.keys())
    n_classes = len(class_names)

    # Extract metrics
    tp_vals = [metrics[c]['TP'] for c in class_names]
    fp_vals = [metrics[c]['FP'] for c in class_names]
    fn_vals = [metrics[c]['FN'] for c in class_names]
    tn_vals = [metrics[c]['TN'] for c in class_names]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Stacked bar chart for TP/FP/FN
    x = np.arange(n_classes)
    width = 0.5

    ax1.bar(x, tp_vals, width, label='True Positives', color='green', alpha=0.8)
    ax1.bar(x, fp_vals, width, bottom=tp_vals, label='False Positives', color='red', alpha=0.8)
    bottom_vals = np.array(tp_vals) + np.array(fp_vals)
    ax1.bar(x, fn_vals, width, bottom=bottom_vals, label='False Negatives', color='orange', alpha=0.8)

    ax1.set_xlabel('Class')  # Uses global fontsize
    ax1.set_ylabel('Count')  # Uses global fontsize
    ax1.set_title(f'TP/FP/FN per Class {title_suffix}')  # Uses global fontsize
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Grouped bar chart for Precision/Recall/F1
    precision_vals = [metrics[c]['Precision'] for c in class_names]
    recall_vals = [metrics[c]['Recall'] for c in class_names]
    f1_vals = [metrics[c]['F1'] for c in class_names]

    x_pos = np.arange(n_classes)
    width = 0.25

    ax2.bar(x_pos - width, precision_vals, width, label='Precision', color='blue', alpha=0.8)
    ax2.bar(x_pos, recall_vals, width, label='Recall', color='green', alpha=0.8)
    ax2.bar(x_pos + width, f1_vals, width, label='F1 Score', color='purple', alpha=0.8)

    ax2.set_xlabel('Class')  # Uses global fontsize
    ax2.set_ylabel('Score')  # Uses global fontsize
    ax2.set_title(f'Precision/Recall/F1 per Class {title_suffix}')  # Uses global fontsize
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_names)
    ax2.set_ylim([0, 1.1])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filename = f"{output_dir}/metrics_breakdown_3class{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def extract_and_print_classification_metrics(metrics, gt_classes, pred_classes):
    """Extract recall values from metrics dict and print combined report"""

    class_keys = list(metrics.keys())

    # Extract recall values from the metrics dictionary
    recall_valid = metrics[class_keys[0]]['Recall']
    recall_minus10 = metrics[class_keys[1]]['Recall']
    recall_minus20 = metrics[class_keys[2]]['Recall']

    # Extract precision values
    precision_valid = metrics[class_keys[0]]['Precision']
    precision_minus10 = metrics[class_keys[1]]['Precision']
    precision_minus20 = metrics[class_keys[2]]['Precision']

    # Extract F1 scores
    f1_valid = metrics[class_keys[0]]['F1']
    f1_minus10 = metrics[class_keys[1]]['F1']
    f1_minus20 = metrics[class_keys[2]]['F1']

    # Calculate overall accuracy
    overall_accuracy = np.sum(gt_classes == pred_classes) / len(gt_classes)

    print(f"\n{'='*80}")
    print(f"COMBINED MEASUREMENT QUALITY REPORT")
    print(f"{'='*80}")

    print(f"\nReturn Classification Performance:")
    print(f"{'='*80}")
    print(f"{'Class':<20} {'Recall':>12} {'Precision':>12} {'F1-Score':>12}")
    print(f"{'-'*80}")
    print(f"{'Valid':<20} {recall_valid*100:>11.1f}% {precision_valid*100:>11.1f}% {f1_valid*100:>11.1f}%")
    print(f"{'Partial Return':<20} {recall_minus10*100:>11.1f}% {precision_minus10*100:>11.1f}% {f1_minus10*100:>11.1f}%")
    print(f"{'No-return':<20} {recall_minus20*100:>11.1f}% {precision_minus20*100:>11.1f}% {f1_minus20*100:>11.1f}%")
    print(f"{'-'*80}")
    print(f"{'Overall Accuracy':<20} {overall_accuracy*100:>11.1f}%")
    print(f"{'='*80}")

    return {
        'recall_valid': recall_valid,
        'recall_minus10': recall_minus10,
        'recall_minus20': recall_minus20,
        'precision_valid': precision_valid,
        'precision_minus10': precision_minus10,
        'precision_minus20': precision_minus20,
        'f1_valid': f1_valid,
        'f1_minus10': f1_minus10,
        'f1_minus20': f1_minus20,
        'overall_accuracy': overall_accuracy
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')

    output_dir = "./confusion_matrix_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load all splits
    print("Loading datasets...")
    train_csv = "./data_splits/train_data.csv"
    val_csv = "./data_splits/val_data.csv"
    test_csv = "./data_splits/test_data.csv"

    train_dataset = BathymetryDataset(train_csv, prediction_type='phi')
    val_dataset = BathymetryDataset(val_csv, prediction_type='phi')
    test_dataset = BathymetryDataset(test_csv, prediction_type='phi')

    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Create combined dataset
    combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=8, shuffle=False)

    print(f"Combined set size: {len(combined_dataset)}")

    # Load model
    print("\nLoading model...")
    model = FullThreeStageModelCNN(prediction_type='phi', dropout_rate=0.1)
    model_path = 'best_full_three_stage_model.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print(f"Loaded model from {model_path}")
    else:
        print(f"ERROR: Model not found at {model_path}")
        return

    # Calculate metrics from inference on combined dataset
    print("\n" + "="*60)
    print("CALCULATING METRICS FROM DIRECT INFERENCE (COMBINED)")
    print("="*60)

    gt_values, pred_values, gt_classes, pred_classes = calculate_metrics_from_inference(
        model, combined_loader, device
    )

    # Calculate regression metrics (MSE, MAE, RMSE) for valid data only
    print("\n" + "="*60)
    print("REGRESSION METRICS")
    print("="*60)
    regression_metrics = calculate_regression_metrics(gt_values, pred_values)

    # Save regression metrics to file
    if regression_metrics:
        with open(f"{output_dir}/regression_metrics_combined.txt", 'w') as f:
            f.write(f"Regression Metrics (Train+Val+Test)\n")
            f.write(f"Valid Data Points Only\n")
            f.write(f"{'='*70}\n")
            f.write(f"Total valid points: {regression_metrics['n_valid']} out of {regression_metrics['n_total']} ")
            f.write(f"({regression_metrics['n_valid']/regression_metrics['n_total']*100:.1f}%)\n")
            f.write(f"MSE:  {regression_metrics['mse']:.6f}\n")
            f.write(f"RMSE: {regression_metrics['rmse']:.6f}\n")
            f.write(f"MAE:  {regression_metrics['mae']:.6f}\n")
        print(f"Regression metrics saved to {output_dir}/regression_metrics_combined.txt")

    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm, cm_norm, metrics = plot_3class_confusion_matrix(
        gt_values,
        pred_values,
        output_dir=output_dir,
        title_suffix="(Train+Val+Test)"
    )

    # Plot metrics breakdown
    print("Generating metrics breakdown...")
    plot_metrics_breakdown(metrics, output_dir=output_dir, title_suffix="(Train+Val+Test)")

    # Extract and print classification metrics
    class_metrics = extract_and_print_classification_metrics(metrics, gt_classes, pred_classes)

    # Save to file
    with open(f"{output_dir}/classification_metrics_combined.txt", 'w') as f:
        f.write(f"Return Classification Performance (Combined: Train+Val+Test)\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'Class':<20} {'Recall':>12} {'Precision':>12} {'F1-Score':>12}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'Valid':<20} {class_metrics['recall_valid']*100:>11.1f}% {class_metrics['precision_valid']*100:>11.1f}% {class_metrics['f1_valid']*100:>11.1f}%\n")
        f.write(f"{'Partial Return':<20} {class_metrics['recall_minus10']*100:>11.1f}% {class_metrics['precision_minus10']*100:>11.1f}% {class_metrics['f1_minus10']*100:>11.1f}%\n")
        f.write(f"{'No-return':<20} {class_metrics['recall_minus20']*100:>11.1f}% {class_metrics['precision_minus20']*100:>11.1f}% {class_metrics['f1_minus20']*100:>11.1f}%\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'Overall Accuracy':<20} {class_metrics['overall_accuracy']*100:>11.1f}%\n")

    print(f"\nResults saved to: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()