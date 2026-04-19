import torch
import numpy as np
import matplotlib.pyplot as plt
from cnn_separate_fls_bathymetry_separate import BathymetryDataset

# Load your dataset
csv_file = "/Users/farhang/Downloads/fls_all_with_phis_long.csv"
dataset = BathymetryDataset(csv_file, prediction_type='phi')

print(f"Total samples: {len(dataset)}")

# Separate samples by validity percentage
valid_samples = []  # >50% valid measurements
invalid_samples = []  # >90% invalid (-20)
mixed_samples = []

for i in range(len(dataset)):
    intensities, targets = dataset[i]

    # Count each type
    valid_count = torch.sum((targets != -10) & (targets != -20)).item()
    invalid_count = torch.sum(targets == -20).item()
    no_return_count = torch.sum(targets == -10).item()
    total = len(targets)

    valid_pct = valid_count / total * 100
    invalid_pct = invalid_count / total * 100

    if valid_pct > 50:  # Majority valid
        valid_samples.append((intensities, targets))
    elif invalid_pct > 90:  # Mostly invalid
        invalid_samples.append((intensities, targets))
    else:
        mixed_samples.append((intensities, targets))

print(f"Samples with >50% valid phi: {len(valid_samples)}")
print(f"Samples with >90% invalid (-20): {len(invalid_samples)}")
print(f"Mixed samples: {len(mixed_samples)}")

# Plot intensity distributions
if len(valid_samples) > 0 and len(invalid_samples) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Compare first valid vs first invalid
    valid_intensities = valid_samples[0][0].numpy()
    invalid_intensities = invalid_samples[0][0].numpy()

    # Histogram comparison
    axes[0, 0].hist(valid_intensities, bins=50, alpha=0.5, label='Valid-rich sample', color='green')
    axes[0, 0].hist(invalid_intensities, bins=50, alpha=0.5, label='Invalid-rich sample', color='red')
    axes[0, 0].set_xlabel('Intensity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Intensity Distribution: Valid-rich vs Invalid-rich')
    axes[0, 0].legend()

    # Time series comparison
    axes[0, 1].plot(valid_intensities, label='Valid-rich sample', alpha=0.7, color='green')
    axes[0, 1].plot(invalid_intensities, label='Invalid-rich sample', alpha=0.7, color='red')
    axes[0, 1].set_xlabel('Position index')
    axes[0, 1].set_ylabel('Intensity')
    axes[0, 1].set_title('Intensity Pattern: Valid-rich vs Invalid-rich')
    axes[0, 1].legend()

    # Statistics comparison
    valid_means = [s[0].mean().item() for s in valid_samples[:min(100, len(valid_samples))]]
    invalid_means = [s[0].mean().item() for s in invalid_samples[:min(100, len(invalid_samples))]]

    axes[1, 0].hist(valid_means, bins=30, alpha=0.5, label=f'Valid-rich ({len(valid_means)} samples)', color='green')
    axes[1, 0].hist(invalid_means, bins=30, alpha=0.5, label=f'Invalid-rich ({len(invalid_means)} samples)', color='red')
    axes[1, 0].set_xlabel('Mean intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Mean Intensity Distribution')
    axes[1, 0].legend()

    # Show where valid phi values occur
    valid_targets = valid_samples[0][1].numpy()
    valid_mask = (valid_targets != -10) & (valid_targets != -20)

    axes[1, 1].scatter(np.arange(len(valid_intensities)), valid_intensities,
                       c=['green' if m else 'gray' for m in valid_mask],
                       s=1, alpha=0.5)
    axes[1, 1].set_xlabel('Position index')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].set_title('Valid-rich sample: Green=valid phi, Gray=invalid phi')

    plt.tight_layout()
    plt.savefig('data_investigation.png', dpi=300)
    plt.show()

    print("\nSaved visualization to 'data_investigation.png'")
else:
    print("Not enough samples to compare!")
    print("Trying alternative: compare positions within mixed samples...")

    # Alternative: look at individual positions within samples
    if len(mixed_samples) > 0:
        sample_intensities, sample_targets = mixed_samples[0]

        print(f"Intensities shape: {sample_intensities.shape}")
        print(f"Targets shape: {sample_targets.shape}")

        # Each intensity (668) corresponds to 4 phi values (668*4=2672)
        # Group targets by their corresponding intensity
        targets_reshaped = sample_targets.reshape(668, 4)  # [668, 4]

        # For each intensity position, check if ANY of the 4 beams has valid phi
        has_valid_phi = torch.any((targets_reshaped != -10) & (targets_reshaped != -20), dim=1)
        all_invalid_phi = torch.all(targets_reshaped == -20, dim=1)

        valid_positions = has_valid_phi.nonzero(as_tuple=True)[0]
        invalid_positions = all_invalid_phi.nonzero(as_tuple=True)[0]

        if len(valid_positions) > 0 and len(invalid_positions) > 0:
            valid_intensities_at_pos = sample_intensities[valid_positions].numpy()
            invalid_intensities_at_pos = sample_intensities[invalid_positions].numpy()

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Histogram
            axes[0].hist(valid_intensities_at_pos, bins=50, alpha=0.5,
                        label=f'Positions with valid phi (n={len(valid_positions)})', color='green')
            axes[0].hist(invalid_intensities_at_pos, bins=50, alpha=0.5,
                        label=f'Positions with all invalid phi (n={len(invalid_positions)})', color='red')
            axes[0].set_xlabel('Intensity')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Intensity Distribution: Valid vs Invalid Phi Positions')
            axes[0].legend()

            # Box plot comparison
            axes[1].boxplot([valid_intensities_at_pos, invalid_intensities_at_pos],
                        labels=['Valid phi positions', 'Invalid phi positions'])
            axes[1].set_ylabel('Intensity')
            axes[1].set_title('Intensity Comparison')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('data_investigation_positions.png', dpi=300)
            plt.show()

            print(f"\nSaved position-based comparison to 'data_investigation_positions.png'")
            print(f"Positions with valid phi: {len(valid_positions)}")
            print(f"Positions with all invalid phi: {len(invalid_positions)}")
            print(f"Mean intensity at valid positions: {valid_intensities_at_pos.mean():.4f}")
            print(f"Mean intensity at invalid positions: {invalid_intensities_at_pos.mean():.4f}")
        else:
            print("No valid or invalid positions found in sample")