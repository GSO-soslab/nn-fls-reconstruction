import math

def extract_phi_from_test_data(test_csv_path, output_path, start_col, end_col, phi_start_col=3341):
    """
    Read test data, extract phi values (radians) from given column range
    (relative to phi_start_col), and save to output CSV.
    """
    print(f"Reading test data: {test_csv_path}")
    with open(test_csv_path, 'r') as f:
        lines = [line.rstrip('\n\r') for line in f.readlines()]

    print(f"Loaded {len(lines)} rows from test data")

    def safe_float(val):
        try:
            if isinstance(val, str):
                val = val.strip()
                if val == '' or val.lower() == 'nan':
                    return float('nan')
            float_val = float(val)
            if float_val in [-10.0, -20.0]:
                return float('nan')
            return float_val
        except (ValueError, TypeError):
            return float('nan')

    # Header: row_index, timestamp, phi columns
    header_parts = ["row_index", "timestamp"]
    for i in range(start_col, end_col + 1):
        header_parts.append(f"phi_col_{phi_start_col + i}_rad")
    header = ",".join(header_parts)
    output_lines = [header]

    for line in lines:
        if not line.strip():
            continue
        values = line.split(',')

        try:
            idx = int(values[0])
            timestamp = safe_float(values[1])
        except (ValueError, IndexError):
            continue

        phi_values_rad = []
        for i in range(start_col, end_col + 1):
            col_index = phi_start_col + i
            phi_rad = safe_float(values[col_index]) if col_index < len(values) else float('nan')
            phi_values_rad.append(phi_rad)

        row = [str(idx), str(timestamp)]
        for rad_val in phi_values_rad:
            row.append("nan" if math.isnan(rad_val) else f"{rad_val:.6f}")

        output_lines.append(",".join(row))

    print(f"Saving to CSV: {output_path}")
    with open(output_path, 'w') as f:
        f.write("\n".join(output_lines) + "\n")

    print(f"Saved {len(output_lines)-1} rows")
    return output_lines


def main():
    start_col = 441
    end_col = 445

    test_csv_path = "./data_splits/test_predictions_with_indices.csv"
    output_path = "./phi_columns_from_test_pred.csv"
    extract_phi_from_test_data(test_csv_path, output_path, start_col, end_col)

    test_csv_path = "./data_splits/test_data_with_indices.csv"
    output_path = "./phi_columns_from_test_data.csv"
    extract_phi_from_test_data(test_csv_path, output_path, start_col, end_col)


if __name__ == "__main__":
    main()