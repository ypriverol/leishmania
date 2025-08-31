# Leishmania Species Classifier

A deep learning model for classifying *Leishmania* species based on protein expression profiles from unique gene-mapped proteins.

## Overview

This predictive model uses protein expression data from proteins that map to unique genes to classify *Leishmania* samples into four species:
- **Lb**: *Leishmania braziliensis*
- **Lg**: *Leishmania guyanensis*
- **Ln**: *Leishmania naiffi*
- **Lp**: *Leishmania panamensis*

## Features

- **Deep Neural Network**: Multi-layer perceptron with dropout for robust classification
- **Random Forest Alternative**: Traditional machine learning approach for comparison
- **Cross-Validation**: Robust model evaluation with stratified k-fold cross-validation
- **Feature Scaling**: StandardScaler for optimal model performance
- **Model Persistence**: Save and load trained models for future use
- **Prediction Interface**: Easy-to-use prediction script for new samples

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the processed protein data available in `../processed_data/protein_dataframe.csv`

## Usage

### Training the Model

To train a new species classification model:

```bash
python leishmania_species_classifier.py
```

This will:
- Load and preprocess the protein expression data
- Filter for proteins with unique gene mapping
- Train a neural network classifier
- Perform cross-validation
- Generate performance metrics and visualizations
- Save the trained model

### Making Predictions

To predict species for a new sample:

```bash
python predict_species.py --sample path/to/sample_data.csv --output results.json
```

Or use the Python API:

```python
from leishmania_species_classifier import LeishmaniaSpeciesClassifier

# Load the trained model
classifier = LeishmaniaSpeciesClassifier()
classifier.load_model('leishmania_species_classifier')

# Predict species for new data
protein_expression = load_your_sample_data()
predicted_species, confidence_scores = classifier.predict_species(protein_expression)
```

## Model Architecture

### Neural Network
- **Input Layer**: Dense layer with 512 neurons
- **Hidden Layers**: 256, 128, 64 neurons with ReLU activation
- **Dropout**: 0.3-0.2 dropout rates for regularization
- **Output Layer**: Softmax activation for multi-class classification
- **Optimizer**: Adam with learning rate reduction on plateau
- **Loss**: Sparse categorical crossentropy

### Data Preprocessing
- **Log2 Transformation**: Applied to protein intensity values
- **Feature Scaling**: StandardScaler for normalization
- **Label Encoding**: Converts species names to numerical labels

## Input Data Format

The model expects protein expression data in the following format:

### Training Data
- CSV file with protein expression data
- Columns: `Protein_IDs`, `Unique_Gene_Mapping`, `Intensity Lb_*`, `Intensity Lg_*`, `Intensity Ln_*`, `Intensity Lp_*`
- Only proteins with `Unique_Gene_Mapping == True` are used

### Prediction Data
- CSV file with protein expression values
- Same protein order as training data
- Single sample per file

## Model Performance

The model typically achieves:
- **Accuracy**: >95% on test set
- **Cross-Validation**: >90% mean accuracy across folds
- **Robust Classification**: High confidence scores for correct predictions

## Output Files

### Training Outputs
- `confusion_matrix.png`: Confusion matrix visualization
- `training_history.png`: Training/validation curves (neural network)
- `leishmania_species_classifier.h5`: Trained neural network model
- `leishmania_species_classifier.joblib`: Trained random forest model
- `leishmania_species_classifier_scaler.joblib`: Feature scaler
- `leishmania_species_classifier_label_encoder.joblib`: Label encoder
- `leishmania_species_classifier_info.joblib`: Model metadata

### Prediction Outputs
- `results.json`: Prediction results with confidence scores

## Model Validation

The model includes comprehensive validation:
- **Stratified Train/Test Split**: Maintains class distribution
- **Cross-Validation**: 5-fold stratified cross-validation
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Visual representation of predictions
- **Confidence Scores**: Probability estimates for each species

## Biological Significance

This model leverages the distinct protein expression patterns identified in our comprehensive analysis:
- **Unique Gene Proteins**: Uses only proteins with unambiguous gene mapping
- **Species-Specific Patterns**: Captures the differential expression signatures
- **Robust Classification**: Handles biological variability in protein expression

## Future Applications

The trained model can be used for:
- **Clinical Diagnostics**: Rapid species identification from patient samples
- **Research Applications**: Automated species classification in large-scale studies
- **Quality Control**: Validation of sample species identity
- **Epidemiological Studies**: Large-scale species distribution analysis

## Technical Notes

- **Reproducibility**: Fixed random seeds for consistent results
- **Scalability**: Handles large protein datasets efficiently
- **Memory Efficient**: Optimized for processing high-dimensional data
- **Cross-Platform**: Compatible with Windows, macOS, and Linux

## Citation

If you use this model in your research, please cite the comprehensive analysis manuscript and acknowledge the use of this predictive model for species classification.

## Support

For questions or issues with the model, please refer to the main project documentation or create an issue in the repository.
