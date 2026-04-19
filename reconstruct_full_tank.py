import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_xz_from_row(row_data, has_indices=False):
    """Extract X, Z points from a single row"""
    if isinstance(row_data, str):
        values = row_data.split(',')
    else:
        values = row_data

    def safe_float(val):
        try:
            if isinstance(val, str):
                val = val.strip()
                if val == '' or val == 'nan':
                    return float('nan')
            return float(val)
        except (ValueError, TypeError):
            return float('nan')

    numeric_values = [safe_float(v) for v in values]

    # Extract phi angles
    if has_indices:
        phis = numeric_values[3342:6014]
    else:
        phis = numeric_values[3341:6013]

    x_points = []
    z_points = []

    max_range = 40.0
    min_range = 0.5
    azimuth = 0.0

    for point_idx in range(668):
        reverse_idx = 667 - point_idx
        range_val = min_range + (reverse_idx / 668) * (max_range - min_range)
        phi_start_idx = point_idx * 4

        for beam_idx in range(4):
            phi_idx = phi_start_idx + beam_idx
            if phi_idx < len(phis):
                phi_rad = phis[phi_idx]
                if not pd.isna(phi_rad) and abs(phi_rad + 10.0) > 0.01 and abs(phi_rad + 20.0) > 0.01:
                    x = range_val * np.cos(azimuth) * np.cos(phi_rad)
                    z = range_val * np.sin(phi_rad)
                    x_points.append(x)
                    z_points.append(z)

    return x_points, z_points


def plot_all_rows(original_csv_path, predictions_csv_path):
    """Stack all rows and plot ground truth vs predictions"""

    print("Loading data...")
    with open(predictions_csv_path, 'r') as f:
        pred_lines = [line.rstrip('\n\r') for line in f.readlines()]

    with open(original_csv_path, 'r') as f:
        orig_lines = [line.rstrip('\n\r') for line in f.readlines()]

    pred_indices = []
    pred_data_lines = []

    for line in pred_lines:
        if line.strip():
            parts = line.split(',', 1)
            if len(parts) >= 2:
                try:
                    pred_indices.append(int(parts[0]))
                    pred_data_lines.append(parts[1])
                except ValueError:
                    continue

    print(f"Processing {len(pred_indices)} rows...")

    gt_x, gt_z = [], []
    pred_x, pred_z = [], []

    for i, orig_idx in enumerate(pred_indices):
        if i % 100 == 0:
            print(f"Row {i}/{len(pred_indices)}")

        if orig_idx < len(orig_lines):
            x, z = extract_xz_from_row(orig_lines[orig_idx], has_indices=False)
            gt_x.extend(x)
            gt_z.extend(z)

        x, z = extract_xz_from_row(pred_data_lines[i], has_indices=True)
        pred_x.extend(x)
        pred_z.extend(z)

    print(f"GT points: {len(gt_x)}, Pred points: {len(pred_x)}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.scatter(gt_x, gt_z, c='blue', s=0.1, alpha=0.5)
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Z (meters)')
    ax1.set_title('Ground Truth')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    ax2.scatter(pred_x, pred_z, c='red', s=0.1, alpha=0.5)
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Z (meters)')
    ax2.set_title('Predictions')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    print("Combining TEST + TRAIN predictions...")

    test_path = "./data_splits/test_predictions_full_three_stage.csv"
    train_path = "./data_splits/train_predictions_full_three_stage.csv"
    original_path = "/home/farhang/Downloads/fls_all_with_phi_long.csv"

    # Load original data
    with open(original_path, 'r') as f:
        orig_lines = [line.rstrip('\n\r') for line in f.readlines()]

    # Combine test + train indices and predictions
    all_indices = []
    all_pred_lines = []

    for pred_path in [test_path, train_path]:
        with open(pred_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.split(',', 1)
                    if len(parts) >= 2:
                        try:
                            all_indices.append(int(parts[0]))
                            all_pred_lines.append(parts[1].rstrip('\n\r'))
                        except ValueError:
                            continue

    print(f"Total combined rows: {len(all_indices)}")

    # Extract points
    gt_x, gt_z = [], []
    pred_x, pred_z = [], []

    for i, orig_idx in enumerate(all_indices):
        if i % 100 == 0:
            print(f"Row {i}/{len(all_indices)}")

        if orig_idx < len(orig_lines):
            x, z = extract_xz_from_row(orig_lines[orig_idx], has_indices=False)
            gt_x.extend(x)
            gt_z.extend(z)

        x, z = extract_xz_from_row(all_pred_lines[i], has_indices=True)
        pred_x.extend(x)
        pred_z.extend(z)

    print(f"GT points: {len(gt_x)}, Pred points: {len(pred_x)}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.scatter(gt_x, gt_z, c='blue', s=0.1, alpha=0.5)
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Z (meters)')
    ax1.set_title('Ground Truth (Test + Train)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    ax2.scatter(pred_x, pred_z, c='red', s=0.1, alpha=0.5)
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Z (meters)')
    ax2.set_title('Predictions (Test + Train)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
