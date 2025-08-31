#!/usr/bin/env python3
"""
Random Forest Leishmania Species Classifier
===========================================

Streamlined Random Forest model for Leishmania species classification.
Uses only proteins that map uniquely to single genes and includes stress testing
for different protein coverage scenarios.

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class LeishmaniaRandomForestClassifier:
    def __init__(self):
        """
        Initialize the Random Forest classifier for Leishmania species.
        """
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.feature_names = None
        self.class_names = None
        self.gene_names = None
        
    def load_and_preprocess_data(self, data_path='../processed_data/protein_dataframe.csv'):
        """
        Load and preprocess protein expression data with strict unique gene mapping.
        """
        print("Loading protein expression data...")
        
        # Load the data
        df = pd.read_csv(data_path)
        
        # Filter for proteins with unique gene mapping ONLY
        df = df[df['Unique_Gene_Mapping'] == True].copy()
        print(f"Proteins with unique gene mapping: {len(df)}")
        
        # Additional check: ensure no protein groups with multiple genes
        if 'Gene_Names' in df.columns:
            # Check for proteins with multiple genes (separated by ; or |)
            multi_gene_mask = df['Gene_Names'].str.contains(';|\\|', na=False)
            df = df[~multi_gene_mask].copy()
            print(f"Proteins with single gene mapping: {len(df)}")
        
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
                if np.sum(sample_values_log2 > 0) > 50:  # At least 50 proteins detected
                    samples_data.append(sample_values_log2)
                    sample_labels.append(species)
        
        # Convert to numpy arrays
        X = np.array(samples_data)
        y = np.array(sample_labels)
        
        print(f"Total samples: {len(X)}")
        print(f"Sample shape: {X.shape}")
        print(f"Species distribution: {np.bincount(self.label_encoder.fit_transform(y))}")
        
        # Store gene names for feature importance
        if 'Gene_Names' in df.columns:
            self.gene_names = df['Gene_Names'].values
        elif 'Gene_IDs' in df.columns:
            self.gene_names = df['Gene_IDs'].values
        elif 'Protein_IDs' in df.columns:
            # Extract gene names from protein IDs
            self.gene_names = []
            for protein_id in df['Protein_IDs']:
                if pd.isna(protein_id):
                    self.gene_names.append("Unknown_Gene")
                else:
                    protein_str = str(protein_id)
                    # Try to extract gene name from various formats
                    if '|' in protein_str:
                        # Format: gene_name|protein_id
                        gene_name = protein_str.split('|')[0]
                    elif ';' in protein_str:
                        # Format: gene_name;protein_id
                        gene_name = protein_str.split(';')[0]
                    elif '_' in protein_str and len(protein_str.split('_')) > 1:
                        # Format: gene_name_protein_id
                        parts = protein_str.split('_')
                        if len(parts) >= 2:
                            gene_name = parts[0] + '_' + parts[1]
                        else:
                            gene_name = parts[0]
                    else:
                        # Use the protein ID as gene name
                        gene_name = protein_str
                    self.gene_names.append(gene_name)
        else:
            # If no gene information available, use protein index
            self.gene_names = [f"Protein_{i}" for i in range(len(df))]
        
        # Clean up gene names (remove any remaining generic patterns)
        cleaned_gene_names = []
        for gene_name in self.gene_names:
            if pd.isna(gene_name):
                cleaned_gene_names.append("Unknown_Gene")
            else:
                gene_str = str(gene_name).strip()
                # Remove any remaining generic patterns
                if gene_str.startswith('Gene_') and gene_str[5:].isdigit():
                    cleaned_gene_names.append(f"Protein_{gene_str[5:]}")
                else:
                    cleaned_gene_names.append(gene_str)
        
        self.gene_names = cleaned_gene_names
        self.feature_names = self.gene_names
        self.class_names = self.label_encoder.classes_
        
        return X, y
    
    def train_model(self, X, y, n_features=None, test_size=0.2, random_state=42):
        """
        Train the Random Forest model with optional feature selection.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Protein expression data
        y : numpy.ndarray
            Species labels
        n_features : int, optional
            Number of top features to select (None for all features)
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
        """
        print(f"Training Random Forest classifier...")
        if n_features:
            print(f"Using top {n_features} features")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Feature selection if specified
        if n_features and n_features < X_train.shape[1]:
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
            X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            X_test_selected = self.feature_selector.transform(X_test)
            
            # Update feature names
            selected_indices = self.feature_selector.get_support()
            self.feature_names = [self.gene_names[i] for i in range(len(self.gene_names)) if selected_indices[i]]
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            self.feature_names = self.gene_names
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Encode labels
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Number of features: {X_train_scaled.shape[1]}")
        print(f"Number of classes: {len(self.class_names)}")
        
        # Create and train Random Forest
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
        plt.title('Confusion Matrix - Random Forest Species Classification')
        plt.xlabel('Predicted Species')
        plt.ylabel('True Species')
        plt.tight_layout()
        plt.savefig('random_forest_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, y_test_encoded, y_pred, y_pred_proba
    
    def stress_test_protein_coverage(self, X, y, protein_coverage_levels=None):
        """
        Stress test the model with different protein coverage scenarios.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Full protein expression data
        y : numpy.ndarray
            Species labels
        protein_coverage_levels : list, optional
            List of protein counts to test (default: [50, 100, 200, 500, 1000, 2000, 5000])
        """
        if protein_coverage_levels is None:
            protein_coverage_levels = [50, 100, 200, 500, 1000, 2000, 5000]
        
        print(f"\nStress Testing Protein Coverage...")
        print(f"Testing coverage levels: {protein_coverage_levels}")
        
        results = {}
        
        for n_proteins in protein_coverage_levels:
            if n_proteins > X.shape[1]:
                print(f"Skipping {n_proteins} proteins (exceeds available features)")
                continue
                
            print(f"\nTesting with {n_proteins} proteins...")
            
            # Cross-validation with feature selection
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Feature selection
                selector = SelectKBest(score_func=f_classif, k=n_proteins)
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_val_selected = selector.transform(X_val)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_selected)
                X_val_scaled = scaler.transform(X_val_selected)
                
                # Encode labels
                label_encoder = LabelEncoder()
                y_train_encoded = label_encoder.fit_transform(y_train)
                y_val_encoded = label_encoder.transform(y_val)
                
                # Train and evaluate
                model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                model.fit(X_train_scaled, y_train_encoded)
                y_pred = model.predict(X_val_scaled)
                score = accuracy_score(y_val_encoded, y_pred)
                scores.append(score)
            
            mean_accuracy = np.mean(scores)
            std_accuracy = np.std(scores)
            results[n_proteins] = {'mean': mean_accuracy, 'std': std_accuracy}
            
            print(f"  Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        return results
    
    def plot_stress_test_results(self, results):
        """
        Plot stress test results.
        """
        protein_counts = list(results.keys())
        accuracies = [results[n]['mean'] for n in protein_counts]
        stds = [results[n]['std'] for n in protein_counts]
        
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy vs protein count
        plt.errorbar(protein_counts, accuracies, yerr=stds, 
                    marker='o', linewidth=2, markersize=8, capsize=5)
        
        # Add horizontal line at 95% accuracy
        plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Accuracy Threshold')
        
        # Add horizontal line at 90% accuracy
        plt.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% Accuracy Threshold')
        
        plt.xlabel('Number of Proteins')
        plt.ylabel('Classification Accuracy')
        plt.title('Random Forest Performance vs Protein Coverage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Add text annotations for key points
        for i, (n_proteins, acc) in enumerate(zip(protein_counts, accuracies)):
            if acc >= 0.95:
                plt.annotate(f'{acc:.3f}', (n_proteins, acc), 
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontweight='bold', color='green')
            elif acc >= 0.90:
                plt.annotate(f'{acc:.3f}', (n_proteins, acc), 
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontweight='bold', color='orange')
            else:
                plt.annotate(f'{acc:.3f}', (n_proteins, acc), 
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontweight='bold', color='red')
        
        plt.tight_layout()
        plt.savefig('protein_coverage_stress_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\nStress Test Summary:")
        print(f"Minimum proteins for 95% accuracy: {min([n for n, r in results.items() if r['mean'] >= 0.95], default='Not achieved')}")
        print(f"Minimum proteins for 90% accuracy: {min([n for n, r in results.items() if r['mean'] >= 0.90], default='Not achieved')}")
    
    def plot_feature_importance(self):
        """
        Plot feature importance with gene names.
        """
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return
        
        feature_importance = self.model.feature_importances_
        
        # Create feature importance dataframe
        feature_df = pd.DataFrame({
            'Gene_Name': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Plot top 20 features
        top_features = feature_df.head(20)
        
        # Create figure with larger size to accommodate longer gene names
        plt.figure(figsize=(14, 10))
        bars = plt.barh(range(len(top_features)), top_features['Importance'])
        
        # Use actual gene names and handle long names
        gene_names_display = []
        for gene_name in top_features['Gene_Name']:
            # Truncate very long names for display
            if len(str(gene_name)) > 40:
                gene_names_display.append(str(gene_name)[:37] + "...")
            else:
                gene_names_display.append(str(gene_name))
        
        plt.yticks(range(len(top_features)), gene_names_display)
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Genes for Species Classification')
        plt.gca().invert_yaxis()  # Show highest importance at top
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
            plt.text(importance + 0.001, i, f'{importance:.4f}', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top features with full names
        print(f"\nTop 10 Most Important Genes:")
        for i, row in top_features.head(10).iterrows():
            gene_name = str(row['Gene_Name'])
            # Show full gene name in console output
            print(f"{i+1:2d}. {gene_name:<50} Importance: {row['Importance']:.4f}")
        
        return feature_df
    
    def save_model(self, model_path='leishmania_random_forest_classifier'):
        """
        Save the trained model and preprocessing components.
        """
        print(f"Saving model to {model_path}...")
        
        import joblib
        
        # Save model
        joblib.dump(self.model, f"{model_path}.joblib")
        
        # Save preprocessing components
        joblib.dump(self.scaler, f"{model_path}_scaler.joblib")
        joblib.dump(self.label_encoder, f"{model_path}_label_encoder.joblib")
        
        if self.feature_selector:
            joblib.dump(self.feature_selector, f"{model_path}_feature_selector.joblib")
        
        # Save feature information
        feature_info = {
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'gene_names': self.gene_names
        }
        joblib.dump(feature_info, f"{model_path}_info.joblib")
        
        print("Model saved successfully!")
    
    def predict_species(self, protein_expression):
        """
        Predict species for new protein expression data.
        """
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return None, None
        
        # Ensure correct shape
        if len(protein_expression.shape) == 1:
            protein_expression = protein_expression.reshape(1, -1)
        
        # Apply feature selection if used during training
        if self.feature_selector:
            protein_expression = self.feature_selector.transform(protein_expression)
        
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
    Main function to train and evaluate the Random Forest classifier.
    """
    print("=" * 60)
    print("RANDOM FOREST LEISHMANIA SPECIES CLASSIFIER")
    print("=" * 60)
    
    # Initialize classifier
    classifier = LeishmaniaRandomForestClassifier()
    
    # Load and preprocess data
    X, y = classifier.load_and_preprocess_data()
    
    # Train the model with all features
    accuracy, y_true, y_pred, y_pred_proba = classifier.train_model(X, y)
    
    # Plot feature importance
    feature_df = classifier.plot_feature_importance()
    
    # Stress test protein coverage
    stress_results = classifier.stress_test_protein_coverage(X, y)
    classifier.plot_stress_test_results(stress_results)
    
    # Save the model
    classifier.save_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print(f"Total proteins used: {len(classifier.feature_names)}")
    print(f"All proteins map uniquely to single genes")
    print("\nModel saved and ready for predictions!")

if __name__ == "__main__":
    main()
