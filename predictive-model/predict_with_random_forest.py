#!/usr/bin/env python3
"""
Predict Species with Random Forest
=================================

Simple script to predict Leishmania species using the trained Random Forest model.
"""

import pandas as pd
import numpy as np
import argparse
from random_forest_classifier import LeishmaniaRandomForestClassifier

def load_new_sample_data(file_path):
    """
    Load protein expression data for a new sample.
    """
    print(f"Loading sample data from {file_path}...")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Check if it's the same format as training data
    if 'Protein_IDs' in df.columns:
        # This is the full dataframe format
        # Filter for unique gene mapping
        df = df[df['Unique_Gene_Mapping'] == True].copy()
        
        # Additional check: ensure no protein groups with multiple genes
        if 'Gene_Names' in df.columns:
            multi_gene_mask = df['Gene_Names'].str.contains(';|\\|', na=False)
            df = df[~multi_gene_mask].copy()
        
        # Get intensity columns (assuming single sample)
        intensity_cols = [col for col in df.columns if col.startswith('Intensity ')]
        
        if len(intensity_cols) == 0:
            raise ValueError("No intensity columns found in the data")
        
        # Use the first intensity column (or you can specify which one)
        sample_values = df[intensity_cols[0]].values
        
    else:
        # Assume it's already in the correct format (protein expression values)
        sample_values = df.values.flatten()
    
    # Apply log2 transformation
    sample_values_log2 = np.log2(sample_values + 1)
    
    print(f"Sample loaded with {len(sample_values_log2)} proteins")
    print(f"Proteins with non-zero expression: {np.sum(sample_values_log2 > 0)}")
    
    return sample_values_log2

def predict_species_from_file(model_path, sample_file, output_file=None):
    """
    Predict species for a new sample from file.
    """
    # Load the trained model
    classifier = LeishmaniaRandomForestClassifier()
    classifier.load_model(model_path)
    
    # Load the new sample data
    sample_data = load_new_sample_data(sample_file)
    
    # Make prediction
    predicted_species, confidence_scores = classifier.predict_species(sample_data)
    
    # Display results
    print("\n" + "=" * 50)
    print("SPECIES PREDICTION RESULTS")
    print("=" * 50)
    print(f"Predicted Species: {predicted_species}")
    print(f"Confidence: {confidence_scores[classifier.label_encoder.transform([predicted_species])[0]]:.4f}")
    
    print(f"\nConfidence Scores for All Species:")
    for i, species in enumerate(classifier.class_names):
        confidence = confidence_scores[i]
        print(f"  {species}: {confidence:.4f}")
    
    # Save results if output file specified
    if output_file:
        results = {
            'predicted_species': predicted_species,
            'confidence': confidence_scores[classifier.label_encoder.transform([predicted_species])[0]],
            'all_confidence_scores': dict(zip(classifier.class_names, confidence_scores))
        }
        
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    return predicted_species, confidence_scores

def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description='Predict Leishmania species using Random Forest')
    parser.add_argument('--model', type=str, default='leishmania_random_forest_classifier_complete',
                       help='Path to the trained Random Forest model (without .joblib extension)')
    parser.add_argument('--sample', type=str, required=True,
                       help='Path to the sample data file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save prediction results')
    
    args = parser.parse_args()
    
    try:
        predicted_species, confidence_scores = predict_species_from_file(
            args.model, args.sample, args.output
        )
        
        print(f"\nPrediction completed successfully!")
        print(f"Species: {predicted_species}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
