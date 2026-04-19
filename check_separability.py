import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch

def visualize_class_separability(dataset, num_samples_per_class=500):
    """
    Check if classes are separable using PCA visualization
    """
    # First, discover what labels actually exist
    print("Discovering label values...")
    sample_labels = []
    for i in range(min(1000, len(dataset))):
        _, label = dataset[i]
        if torch.is_tensor(label):
            label = label.item() if label.numel() == 1 else label.argmax().item()
        sample_labels.append(int(label))

    unique_labels = sorted(set(sample_labels))
    print(f"Found unique labels: {unique_labels}")
    print(f"Label counts: {[(l, sample_labels.count(l)) for l in unique_labels]}")

    # Use actual labels found
    class_samples = {label: [] for label in unique_labels}
    class_names = {unique_labels[i]: f'Class_{i}' for i in range(len(unique_labels))}

    # If you have 3 classes, map them
    if len(unique_labels) == 3:
        class_names = {unique_labels[0]: 'Valid', unique_labels[1]: '(-10)', unique_labels[2]: '(-20)'}

    print("\nCollecting samples from dataset...")
    for i in range(len(dataset)):
        data, label = dataset[i]

        # Convert to numpy and flatten
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        if torch.is_tensor(label):
            label = label.item() if label.numel() == 1 else label.argmax().item()

        label = int(label)

        if label in class_samples and len(class_samples[label]) < num_samples_per_class:
            class_samples[label].append(data.flatten())

        # Stop when we have enough samples
        if all(len(samples) >= num_samples_per_class for samples in class_samples.values()):
            break

    # Print class sample counts
    for label, samples in class_samples.items():
        print(f"Class {class_names[label]}: {len(samples)} samples")

    # Rest of the function stays the same...
    all_samples = []
    all_labels = []
    for label, samples in class_samples.items():
        all_samples.extend(samples)
        all_labels.extend([label] * len(samples))

    all_samples = np.array(all_samples)
    all_labels = np.array(all_labels)

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Feature dimension: {all_samples.shape[1]}")

    # Apply PCA
    print("\nApplying PCA...")
    pca = PCA(n_components=2)
    samples_2d = pca.fit_transform(all_samples)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # Plot
    plt.figure(figsize=(12, 5))

    # Plot 1: Scatter plot
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, label in enumerate(sorted(class_samples.keys())):
        mask = all_labels == label
        plt.scatter(samples_2d[mask, 0], samples_2d[mask, 1],
                   c=colors[i % len(colors)], label=class_names[label], alpha=0.5, s=20)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA: Class Separability')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Density plot
    plt.subplot(1, 2, 2)
    for i, label in enumerate(sorted(class_samples.keys())):
        mask = all_labels == label
        plt.hist(samples_2d[mask, 0], bins=50, alpha=0.5,
                label=class_names[label], color=colors[i % len(colors)])

    plt.xlabel('PC1')
    plt.ylabel('Count')
    plt.title('PC1 Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('class_separability_analysis.png', dpi=150)
    print("\nPlot saved as 'class_separability_analysis.png'")
    plt.show()

    # Calculate class statistics
    print("\n" + "="*60)
    print("CLASS STATISTICS")
    print("="*60)
    for label in sorted(class_samples.keys()):
        mask = all_labels == label
        pc1_mean = samples_2d[mask, 0].mean()
        pc1_std = samples_2d[mask, 0].std()
        pc2_mean = samples_2d[mask, 1].mean()
        pc2_std = samples_2d[mask, 1].std()

        print(f"\n{class_names[label]}:")
        print(f"  PC1: {pc1_mean:.3f} ± {pc1_std:.3f}")
        print(f"  PC2: {pc2_mean:.3f} ± {pc2_std:.3f}")