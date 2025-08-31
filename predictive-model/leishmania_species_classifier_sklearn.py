#!/usr/bin/env python3
"""
Leishmania Species Classifier (Scikit-learn Version)
====================================================

Machine learning model for classifying Leishmania species based on protein expression profiles.
Uses protein expression data from proteins that map to unique genes to predict species.

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class LeishmaniaSpeciesClassifier:
    def __init__(self, model_type='random_forest'):
        """
        Initialize the Leishmania species classifier.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest', 'mlp', 'svm', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        
    def load_and_preprocess_data(self, data_path='../processed_data/protein_dataframe.csv'):
        """
        Load and preprocess the protein expression data.
        
        Parameters:
        -----------
        data_path : str
            Path to the protein dataframe CSV file
        """
        print("Loading protein expression data...")
        
        # Load the data
        df = pd.read_csv(data_path)
        
        # Filter for proteins with unique gene mapping
        df = df[df['Unique_Gene_Mapping'] == True].copy()
        print(f"Proteins with unique gene mapping: {len(df)}")
        
        # Get intensity columns for each species
        lb_cols = [col for col in df.columns if col.startswith('Intensity Lb_')]
        lg_cols = [col for col in df.columns if col.startswith('Intensity Lg_')]
        ln_cols = [col for col in df.columns if col.startswith('Intensity Ln_')]
        lp_cols = [col for col in df.columns if col.startswith('Intensity Lp_')]
        
        # Create sample-level data
        samples_data = []
        sample_labels = []
        
        # Process each species
        species_mapping = {
            'Lb': lb_cols,
            'Lg': lg_cols,
            'Ln': ln_cols,
            'Lp': lp_cols
        }
        
        for species, intensity_cols in species_mapping.items():
            print(f"Processing {species} samples...")
            
            for col in intensity_cols:
                # Get protein expression values for this sample
                sample_values = df[col].values
                
                # Apply log2 transformation
                sample_values_log2 = np.log2(sample_values + 1)
                
                # Only include samples with at least some non-zero values
                if np.sum(sample_values_log2 > 0) > 100:  # At least 100 proteins detected
                    samples_data.append(sample_values_log2)
                    sample_labels.append(species)
        
        # Convert to numpy arrays
        X = np.array(samples_data)
        y = np.array(sample_labels)
        
        print(f"Total samples: {len(X)}")
        print(f"Sample shape: {X.shape}")
        print(f"Species distribution: {np.bincount(self.label_encoder.fit_transform(y))}")
        
        self.feature_names = [f"Protein_{i}" for i in range(X.shape[1])]
        self.class_names = self.label_encoder.classes_
        
        return X, y
    
    def create_model(self, random_state=42):
        """
        Create the specified machine learning model.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        elif self.model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=200,
                random_state=random_state
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=random_state
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train the species classification model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Protein expression data
        y : numpy.ndarray
            Species labels
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
        """
        print(f"Training {self.model_type} species classification model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of classes: {len(self.class_names)}")
        
        # Create and train model
        self.create_model(random_state)
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {self.model_type.upper()} Species Classification')
        plt.xlabel('Predicted Species')
        plt.ylabel('True Species')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, y_test_encoded, y_pred, y_pred_proba
    
    def cross_validate_model(self, X, y, n_splits=5):
        """
        Perform cross-validation to assess model robustness.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Protein expression data
        y : numpy.ndarray
            Species labels
        n_splits : int
            Number of cross-validation folds
        """
        print(f"Performing {n_splits}-fold cross-validation...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.transform(y)
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_encoded)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            X_train_fold = X_scaled[train_idx]
            X_val_fold = X_scaled[val_idx]
            y_train_fold = y_encoded[train_idx]
            y_val_fold = y_encoded[val_idx]
            
            # Create and train model for this fold
            self.create_model()
            self.model.fit(X_train_fold, y_train_fold)
            
            # Evaluate
            y_pred_fold = self.model.predict(X_val_fold)
            
            # Calculate accuracy
            fold_accuracy = accuracy_score(y_val_fold, y_pred_fold)
            cv_scores.append(fold_accuracy)
            
            print(f"  Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")
        
        print(f"\nCross-Validation Results:")
        print(f"Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        return cv_scores
    
    def save_model(self, model_path='leishmania_species_classifier'):
        """
        Save the trained model and preprocessing components.
        
        Parameters:
        -----------
        model_path : str
            Path to save the model
        """
        print(f"Saving model to {model_path}...")
        
        import joblib
        
        # Save model
        joblib.dump(self.model, f"{model_path}_{self.model_type}.joblib")
        
        # Save preprocessing components
        joblib.dump(self.scaler, f"{model_path}_{self.model_type}_scaler.joblib")
        joblib.dump(self.label_encoder, f"{model_path}_{self.model_type}_label_encoder.joblib")
        
        # Save feature information
        feature_info = {
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'model_type': self.model_type
        }
        joblib.dump(feature_info, f"{model_path}_{self.model_type}_info.joblib")
        
        print("Model saved successfully!")
    
    def load_model(self, model_path='leishmania_species_classifier'):
        """
        Load a trained model and preprocessing components.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        """
        print(f"Loading model from {model_path}...")
        
        import joblib
        
        # Load preprocessing components
        self.scaler = joblib.load(f"{model_path}_{self.model_type}_scaler.joblib")
        self.label_encoder = joblib.load(f"{model_path}_{self.model_type}_label_encoder.joblib")
        
        # Load feature information
        feature_info = joblib.load(f"{model_path}_{self.model_type}_info.joblib")
        self.feature_names = feature_info['feature_names']
        self.class_names = feature_info['class_names']
        
        # Load model
        self.model = joblib.load(f"{model_path}_{self.model_type}.joblib")
        
        print("Model loaded successfully!")
    
    def predict_species(self, protein_expression):
        """
        Predict species for new protein expression data.
        
        Parameters:
        -----------
        protein_expression : numpy.ndarray
            Protein expression values for a new sample
            
        Returns:
        --------
        tuple
            (predicted_species, confidence_scores)
        """
        # Ensure correct shape
        if len(protein_expression.shape) == 1:
            protein_expression = protein_expression.reshape(1, -1)
        
        # Scale the data
        protein_expression_scaled = self.scaler.transform(protein_expression)
        
        # Make prediction
        prediction_proba = self.model.predict_proba(protein_expression_scaled)
        predicted_class = self.model.predict(protein_expression_scaled)
        
        # Convert back to species names
        predicted_species = self.label_encoder.inverse_transform(predicted_class)
        confidence_scores = prediction_proba[0]
        
        return predicted_species[0], confidence_scores

def main():
    """
    Main function to train and evaluate the Leishmania species classifier.
    """
    print("=" * 60)
    print("LEISHMANIA SPECIES CLASSIFIER (SCIKIT-LEARN)")
    print("=" * 60)
    
    # Initialize classifier with Random Forest
    classifier = LeishmaniaSpeciesClassifier(model_type='random_forest')
    
    # Load and preprocess data
    X, y = classifier.load_and_preprocess_data()
    
    # Train the model
    accuracy, y_true, y_pred, y_pred_proba = classifier.train_model(X, y)
    
    # Perform cross-validation
    cv_scores = classifier.cross_validate_model(X, y)
    
    # Save the model
    classifier.save_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print("\nModel saved and ready for predictions!")

if __name__ == "__main__":
    main()
