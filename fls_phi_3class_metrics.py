import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd
import os
from plot_test import extract_pcl_points_from_row

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
    # Everything else stays 0 (valid measurements)

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

def plot_3class_confusion_matrix(original_phis, predicted_phis, output_dir="./"):
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
    print(f"(-10):    {np.sum(y_true == 1)} ({np.sum(y_true == 1)/len(y_true)*100:.1f}%)")
    print(f" (-20):      {np.sum(y_true == 2)} ({np.sum(y_true == 2)/len(y_true)*100:.1f}%)")

    print(f"\nPrediction Distribution:")
    print(f"  Valid measurements: {np.sum(y_pred == 0)} ({np.sum(y_pred == 0)/len(y_pred)*100:.1f}%)")
    print(f" (-10):    {np.sum(y_pred == 1)} ({np.sum(y_pred == 1)/len(y_pred)*100:.1f}%)")
    print(f" (-20):      {np.sum(y_pred == 2)} ({np.sum(y_pred == 2)/len(y_pred)*100:.1f}%)")

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    class_names = ['Valid', '\n(-10)', '\n(-20)']

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    ax1.set_xlabel('Predicted Class', fontsize=12)
    ax1.set_ylabel('Ground Truth Class', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14)

    # Normalized (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Proportion'}, annot_kws={'size': 14})
    ax2.set_xlabel('Predicted Class', fontsize=12)
    ax2.set_ylabel('Ground Truth Class', fontsize=12)
    ax2.set_title('Confusion Matrix (Normalized by Row)', fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_3class.png", dpi=300, bbox_inches='tight')
    # plt.show()

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


def plot_metrics_breakdown(metrics, output_dir="./"):
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

    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('TP/FP/FN per Class', fontsize=14)
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

    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Precision/Recall/F1 per Class', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_names)
    ax2.set_ylim([0, 1.1])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_breakdown_3class.png", dpi=300, bbox_inches='tight')
    plt.show()


def extract_all_phis_from_comparison(test_csv_path, original_csv_path):
    """
    Extract all phi values from both test and original data
    """
    with open(test_csv_path, 'r') as f:
        test_lines = [line.rstrip('\n\r') for line in f.readlines()]

    with open(original_csv_path, 'r') as f:
        original_lines = [line.rstrip('\n\r') for line in f.readlines()]

    test_indices = []
    test_data_lines = []

    for line in test_lines:
        if line.strip():
            parts = line.split(',', 1)
            if len(parts) >= 2:
                try:
                    test_indices.append(int(parts[0]))
                    test_data_lines.append(parts[1])
                except ValueError:
                    continue

    all_orig_phis = []
    all_test_phis = []

    for i, original_row_idx in enumerate(test_indices):
        if original_row_idx < len(original_lines):
            original_row_data = original_lines[original_row_idx]
            test_row_data = test_data_lines[i]

            # Extract phis using your existing function
            # Since you're storing phi in both x and z temporarily
            orig_x, orig_z = extract_pcl_points_from_row(
                original_row_data, 0.05988024, 0.1, 0.0, has_indices=False
            )
            test_x, test_z = extract_pcl_points_from_row(
                test_row_data, 0.05988024, 0.1, 0.0, has_indices=True
            )

            # Your code stores phi values in z (and temporarily in x too)
            all_orig_phis.extend(orig_z)
            all_test_phis.extend(test_z)

    return np.array(all_orig_phis), np.array(all_test_phis)


# Update your main function
def main():
    # predictions_with_indices_path = "./data_splits/test_predictions_final.csv"
    predictions_with_indices_path = "./data_splits/test_predictions_final.csv"
    original_csv_path = "/Users/farhang/Downloads/fls_all_with_phi.csv"
    output_dir = "./confusion_matrix_results"

    os.makedirs(output_dir, exist_ok=True)

    # Extract all phis
    print("Extracting phi values from all rows...")
    original_phis, predicted_phis = extract_all_phis_from_comparison(
        predictions_with_indices_path,
        original_csv_path
    )

    print(f"\nExtracted {len(original_phis)} phi measurements")

    # Create 3-class confusion matrix and get metrics
    cm, cm_norm, metrics = plot_3class_confusion_matrix(
        original_phis,
        predicted_phis,
        output_dir=output_dir
    )

    # Plot detailed metrics breakdown
    plot_metrics_breakdown(metrics, output_dir=output_dir)

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()