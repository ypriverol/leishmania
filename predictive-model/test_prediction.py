#!/usr/bin/env python3
"""
Test Prediction Script
=====================

Simple script to test the trained Leishmania species classifier.
"""

import numpy as np
from leishmania_species_classifier_sklearn import LeishmaniaSpeciesClassifier

def test_prediction():
    """
    Test the prediction functionality with sample data.
    """
    print("Testing Leishmania Species Classifier...")
    
    # Load the trained model
    classifier = LeishmaniaSpeciesClassifier(model_type='random_forest')
    classifier.load_model('leishmania_species_classifier')
    
    print(f"Model loaded successfully!")
    print(f"Model type: {classifier.model_type}")
    print(f"Number of features: {len(classifier.feature_names)}")
    print(f"Classes: {classifier.class_names}")
    
    # Create a test sample (random protein expression values)
    # In practice, this would be real protein expression data
    test_sample = np.random.rand(8144) * 1000  # Random intensity values
    
    # Apply log2 transformation
    test_sample_log2 = np.log2(test_sample + 1)
    
    print(f"\nTest sample shape: {test_sample_log2.shape}")
    print(f"Test sample range: {test_sample_log2.min():.2f} to {test_sample_log2.max():.2f}")
    
    # Make prediction
    predicted_species, confidence_scores = classifier.predict_species(test_sample_log2)
    
    print(f"\nPrediction Results:")
    print(f"Predicted Species: {predicted_species}")
    print(f"Confidence: {confidence_scores[classifier.label_encoder.transform([predicted_species])[0]]:.4f}")
    
    print(f"\nConfidence Scores for All Species:")
    for i, species in enumerate(classifier.class_names):
        confidence = confidence_scores[i]
        print(f"  {species}: {confidence:.4f}")
    
    print(f"\nTest completed successfully!")

if __name__ == "__main__":
    test_prediction()
