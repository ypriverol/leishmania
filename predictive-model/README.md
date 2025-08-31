# Leishmania Species Classifier - Random Forest

A streamlined Random Forest model for classifying *Leishmania* species based on protein expression profiles from proteins that map uniquely to single genes.

## Overview

This model uses only proteins with unambiguous single-gene mapping to classify *Leishmania* samples into four species:
- **Lb**: *Leishmania braziliensis*
- **Lg**: *Leishmania guyanensis*
- **Ln**: *Leishmania naiffi*
- **Lp**: *Leishmania panamensis*

## Key Features

- **Strict Gene Mapping**: Only proteins with unique single-gene mapping
- **Stress Testing**: Comprehensive protein coverage testing (50-5000 proteins)
- **Real-world Scenarios**: Simulates samples with limited protein detection
- **Feature Importance**: Identifies most discriminatory genes

## Model Performance

### Training Results
- **Dataset**: 50 samples (24 Lb, 11 Lg, 5 Ln, 10 Lp)
- **Features**: 8,144 proteins with unique single-gene mapping
- **Accuracy**: 100% (1.0000)

### Stress Test Results
- **Minimum for 90% Accuracy**: 100 proteins
- **Minimum for 95% Accuracy**: 200 proteins
- **Perfect Performance**: 200+ proteins

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python random_forest_classifier.py
```

This will:
- Load and preprocess protein expression data
- Filter for unique single-gene mapping proteins
- Train Random Forest classifier
- Perform stress testing with different protein coverage levels
- Generate visualizations and save the model

### Making Predictions

```bash
python predict_with_random_forest.py --sample path/to/sample.csv --output results.json
```

## Input Data Format

### Training Data
- CSV file with protein expression data
- Columns: `Protein_IDs`, `Unique_Gene_Mapping`, `Gene_Names`, `Intensity Lb_*`, `Intensity Lg_*`, `Intensity Ln_*`, `Intensity Lp_*`
- Only proteins with `Unique_Gene_Mapping == True` and single gene mapping are used

### Prediction Data
- CSV file with protein expression values
- Same protein order as training data
- Single sample per file

## Output Files

### Model Files
- `leishmania_random_forest_classifier.joblib`: Trained model
- `leishmania_random_forest_classifier_scaler.joblib`: Feature scaler
- `leishmania_random_forest_classifier_label_encoder.joblib`: Label encoder
- `leishmania_random_forest_classifier_info.joblib`: Model metadata

### Visualizations
- `random_forest_confusion_matrix.png`: Confusion matrix
- `random_forest_feature_importance.png`: Feature importance plot
- `protein_coverage_stress_test.png`: Stress test results

## Applications

### Clinical Diagnostics
- Rapid species identification from patient samples
- Minimum 200 proteins for 95% accuracy
- Quality assessment based on protein count

### Research Applications
- Automated species classification in large-scale studies
- Feature importance analysis for biological insights
- Flexible protein coverage requirements

## Technical Details

### Model Architecture
- **Algorithm**: Random Forest
- **Parameters**: 100 trees, max_depth=10
- **Feature Selection**: Optional top-k selection
- **Preprocessing**: Log2 transformation, standardization

### Performance
- **Training Time**: ~30 seconds
- **Prediction Time**: <1 second per sample
- **Memory Usage**: ~500MB during training

## Dependencies

- scikit-learn >= 1.1.0
- pandas >= 1.4.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- joblib >= 1.1.0

## Biological Validation

The model successfully identifies distinct protein expression patterns for each *Leishmania* species using only proteins with unambiguous single-gene mapping. This ensures reliable, biologically meaningful classifications based on clear genetic relationships.

## Citation

When using this model in research, please cite the comprehensive proteomic analysis manuscript and acknowledge the use of this Random Forest model for species classification.
