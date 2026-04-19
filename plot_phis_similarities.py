import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def calculate_phi_similarity(orig_phis, test_phis):
    """
    Calculate exact similarity percentage - pure array comparison.
    """
    orig_phis = np.array(orig_phis)
    test_phis = np.array(test_phis)

    if orig_phis.shape != test_phis.shape:
        return 0.0

    # Direct exact comparison
    exact_matches = (orig_phis == test_phis)

    # Handle NaN comparisons (NaN != NaN in numpy)
    both_nan = np.isnan(orig_phis) & np.isnan(test_phis)
    exact_matches = exact_matches | both_nan

    similarity_percentage = (np.sum(exact_matches) / len(orig_phis)) * 100
    return similarity_percentage

def process_original_phis(phis):
    """
    Process original phi values according to the 4-column grouping rule:
    - If all 4 phis in a group are NaN -> replace all with -20.0
    - If at least one phi in a group is non-NaN -> keep non-NaN values, replace NaN with -10.0
    """
    phis = np.array(phis, dtype=float)
    processed_phis = phis.copy()

    # Process in groups of 4 (668 points * 4 beams = 2672 total)
    for i in range(0, len(phis), 4):
        group = phis[i:i+4]

        # Check if all 4 are NaN
        if np.all(np.isnan(group)):
            # Replace all with -20.0
            processed_phis[i:i+4] = -20.0
        else:
            # At least one is non-NaN, replace only NaN values with -10.0
            nan_mask = np.isnan(group)
            processed_phis[i:i+4][nan_mask] = -10.0

    return processed_phis

def extract_phis_from_row(row_data, is_original=False):
    """
    Extract phi values from a CSV row.
    """
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
    phis = numeric_values[3341:6013]

    # Apply special processing only for original file
    if is_original:
        phis = process_original_phis(phis)

    return phis

def plot_phi_similarities(original_csv_path, predicted_csv_path, output_dir="./phi_similarity_plots"):
    """
    Plot exact similarity between phi values in two CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Reading CSV files...")

    # Read both files
    with open(original_csv_path, 'r') as f:
        original_lines = [line.rstrip('\n\r') for line in f.readlines()]

    with open(predicted_csv_path, 'r') as f:
        predicted_lines = [line.rstrip('\n\r') for line in f.readlines()]

    print(f"Processing {min(len(original_lines), len(predicted_lines))} rows...")
    print("Applying NaN processing rules to original file...")

    # Calculate exact similarities
    row_numbers = []
    similarities = []

    max_rows = min(len(original_lines), len(predicted_lines))

    for i in range(max_rows):
        if i % 100 == 0:
            print(f"Processing row {i}/{max_rows}...")

        try:
            # Extract and process original phis (with special NaN handling)
            orig_phis = extract_phis_from_row(original_lines[i], is_original=True)
            # Extract predicted phis (no special processing)
            pred_phis = extract_phis_from_row(predicted_lines[i], is_original=False)

            similarity = calculate_phi_similarity(orig_phis, pred_phis)

            row_numbers.append(i)
            similarities.append(similarity)

        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue

    # Create the 2 plots
    plt.figure(figsize=(15, 8))

    # Plot 1: Exact similarity by row
    plt.subplot(2, 1, 1)
    plt.scatter(row_numbers, similarities, alpha=0.5, color='steelblue', s=2)
    plt.xlabel('Row Number')
    plt.ylabel('Exact Similarity (%)')
    plt.title('Phi Exact Similarity Between Processed Original and Predicted Data by Row')
    plt.grid(True, alpha=0.3, axis='y')

    # Add reference lines
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect Match (100%)')
    plt.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% Threshold')
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
    plt.legend(loc='lower right')

    # Plot 2: Histogram
    plt.subplot(2, 1, 2)
    plt.hist(similarities, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Exact Similarity (%)')
    plt.ylabel('Number of Rows')
    plt.title('Distribution of Phi Exact Similarities (with NaN processing)')
    plt.grid(True, alpha=0.3, axis='y')

    # Add statistics
    mean_similarity = np.mean(similarities)
    median_similarity = np.median(similarities)
    std_similarity = np.std(similarities)

    stats_text = f'Mean: {mean_similarity:.2f}%\n'
    stats_text += f'Median: {median_similarity:.2f}%\n'
    stats_text += f'Std: {std_similarity:.2f}%\n'
    stats_text += f'Min: {np.min(similarities):.2f}%\n'
    stats_text += f'Max: {np.max(similarities):.2f}%'

    plt.text(0.75, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/phi_similarity_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nExact Similarity Analysis Complete:")
    print(f"  Total rows analyzed: {len(similarities)}")
    print(f"  Mean exact similarity: {mean_similarity:.2f}%")
    print(f"  Rows with >90% similarity: {np.sum(np.array(similarities) > 90)}/{len(similarities)}")
    print(f"  Rows with >95% similarity: {np.sum(np.array(similarities) > 95)}/{len(similarities)}")
    print(f"  Rows with 100% similarity: {np.sum(np.array(similarities) == 100)}/{len(similarities)}")

    return row_numbers, similarities

if __name__ == "__main__":
    original_csv = "/Users/farhang/Downloads/fls_all_with_phis_long.csv"
    predicted_csv = "full_predictions.csv"

    row_numbers, similarities = plot_phi_similarities(original_csv, predicted_csv)

    above_90_percentage = (np.sum(np.array(similarities) > 90) / len(similarities)) * 100
    print(f"Percentage of rows with >90% similarity: {above_90_percentage:.1f}%")