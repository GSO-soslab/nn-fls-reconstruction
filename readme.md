# Bathymetry Neural Network

A PyTorch implementation of 1D U-Net models for predicting bathymetric angles (phi and tangent) from sonar intensity data.

## Requirements

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn
```

## Data Format

The input CSV should contain:
- Column 0: timestamp
- Columns 1-668: intensity values
- Columns 669-3340: tangent angles (2672 values, 4 beams × 668 points)
- Columns 3341-6012: phi angles (2672 values, 4 beams × 668 points)

## Usage

### Basic Training and Testing

```python
python cnn_fls_bathymetry.py
```


### Key Configuration

Update these paths in the `main()` function:

```python
csv_file = '/path/to/your/fls_all_with_phi.csv'
```

### Model Parameters

- **Prediction types**: 'phi', 'tangent', or 'combined'
- **Architecture**: 1D U-Net with residual blocks
- **Loss function**: Combined classification + regression loss
- **Training**: AdamW optimizer with OneCycleLR scheduler

### Output Files

The script generates:
- `best_bathymetry_model_phi.pth` - trained phi model
- `best_bathymetry_model_tangent.pth` - trained tangent model
- `training_curves_*.png` - loss plots
- `test_outputs_*/` - prediction results and CSVs

### Data Splits

First run creates CSV splits in `./data_splits/`:
- `train_data.csv`
- `val_data.csv`
- `test_data.csv`