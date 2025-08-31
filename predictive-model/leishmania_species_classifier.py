#!/usr/bin/env python3
"""
Leishmania Species Classifier
=============================

Deep learning model for classifying Leishmania species based on protein expression profiles.
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class LeishmaniaSpeciesClassifier:
    def __init__(self, model_type='neural_network'):
        """
        Initialize the Leishmania species classifier.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('neural_network' or 'random_forest')
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
        
        # Calculate median expression for each species per protein
        df['Lb_median'] = df[lb_cols].median(axis=1)
        df['Lg_median'] = df[lg_cols].median(axis=1)
        df['Ln_median'] = df[ln_cols].median(axis=1)
        df['Lp_median'] = df[lp_cols].median(axis=1)
        
        # Apply log2 transformation
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            df[f'{species}_log2'] = np.log2(df[f'{species}_median'] + 1)
        
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
    
    def create_neural_network(self, input_dim, num_classes):
        """
        Create a neural network model for species classification.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features (proteins)
        num_classes : int
            Number of species classes
        """
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
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
        print("Training species classification model...")
        
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
        
        if self.model_type == 'neural_network':
            # Create and train neural network
            self.model = self.create_neural_network(X_train.shape[1], len(self.class_names))
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
            
            # Train the model
            history = self.model.fit(
                X_train_scaled, y_train_encoded,
                epochs=100,
                batch_size=32,
                validation_data=(X_test_scaled, y_test_encoded),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluate the model
            y_pred_proba = self.model.predict(X_test_scaled)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
        elif self.model_type == 'random_forest':
            # Train Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
            
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
        plt.title('Confusion Matrix - Species Classification')
        plt.xlabel('Predicted Species')
        plt.ylabel('True Species')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot training history for neural network
        if self.model_type == 'neural_network':
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
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
            
            if self.model_type == 'neural_network':
                # Create and train model for this fold
                fold_model = self.create_neural_network(X_train_fold.shape[1], len(self.class_names))
                
                early_stopping = EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True
                )
                
                fold_model.fit(
                    X_train_fold, y_train_fold,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val_fold, y_val_fold),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Evaluate
                y_pred_fold = np.argmax(fold_model.predict(X_val_fold), axis=1)
                
            elif self.model_type == 'random_forest':
                # Train Random Forest for this fold
                fold_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                fold_model.fit(X_train_fold, y_train_fold)
                y_pred_fold = fold_model.predict(X_val_fold)
            
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
        
        if self.model_type == 'neural_network':
            self.model.save(f"{model_path}.h5")
        else:
            import joblib
            joblib.dump(self.model, f"{model_path}.joblib")
        
        # Save preprocessing components
        import joblib
        joblib.dump(self.scaler, f"{model_path}_scaler.joblib")
        joblib.dump(self.label_encoder, f"{model_path}_label_encoder.joblib")
        
        # Save feature information
        feature_info = {
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'model_type': self.model_type
        }
        joblib.dump(feature_info, f"{model_path}_info.joblib")
        
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
        self.scaler = joblib.load(f"{model_path}_scaler.joblib")
        self.label_encoder = joblib.load(f"{model_path}_label_encoder.joblib")
        
        # Load feature information
        feature_info = joblib.load(f"{model_path}_info.joblib")
        self.feature_names = feature_info['feature_names']
        self.class_names = feature_info['class_names']
        self.model_type = feature_info['model_type']
        
        # Load model
        if self.model_type == 'neural_network':
            self.model = keras.models.load_model(f"{model_path}.h5")
        else:
            self.model = joblib.load(f"{model_path}.joblib")
        
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
        if self.model_type == 'neural_network':
            prediction_proba = self.model.predict(protein_expression_scaled)
            predicted_class = np.argmax(prediction_proba, axis=1)
        else:
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
    print("LEISHMANIA SPECIES CLASSIFIER")
    print("=" * 60)
    
    # Initialize classifier
    classifier = LeishmaniaSpeciesClassifier(model_type='neural_network')
    
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
