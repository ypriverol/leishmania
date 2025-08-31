#!/usr/bin/env python3
"""
Leishmania Species Prediction Script
====================================

Script for predicting Leishmania species from new protein expression data
using the trained deep learning model.

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import argparse
from leishmania_species_classifier import LeishmaniaSpeciesClassifier

def load_new_sample_data(file_path):
    """
    Load protein expression data for a new sample.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file containing protein expression data
        
    Returns:
    --------
    numpy.ndarray
        Protein expression values
    """
    print(f"Loading sample data from {file_path}...")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Check if it's the same format as training data
    if 'Protein_IDs' in df.columns:
        # This is the full dataframe format
        # Filter for unique gene mapping
        df = df[df['Unique_Gene_Mapping'] == True].copy()
        
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
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    sample_file : str
        Path to the sample data file
    output_file : str, optional
        Path to save prediction results
    """
    # Load the trained model
    classifier = LeishmaniaSpeciesClassifier()
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

def predict_species_from_values(model_path, protein_values):
    """
    Predict species from protein expression values array.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    protein_values : numpy.ndarray
        Protein expression values
        
    Returns:
    --------
    tuple
        (predicted_species, confidence_scores)
    """
    # Load the trained model
    classifier = LeishmaniaSpeciesClassifier()
    classifier.load_model(model_path)
    
    # Apply log2 transformation
    protein_values_log2 = np.log2(protein_values + 1)
    
    # Make prediction
    predicted_species, confidence_scores = classifier.predict_species(protein_values_log2)
    
    return predicted_species, confidence_scores

def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description='Predict Leishmania species from protein expression data')
    parser.add_argument('--model', type=str, default='leishmania_species_classifier',
                       help='Path to the trained model')
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
