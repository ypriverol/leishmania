#!/usr/bin/env python3
"""
Model Comparison Script
======================

Compare different machine learning algorithms for Leishmania species classification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(data_path='../processed_data/protein_dataframe.csv'):
    """
    Load and preprocess the protein expression data.
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
    
    return X, y

def compare_models(X, y):
    """
    Compare different machine learning models.
    """
    print("Comparing machine learning models...")
    
    # Define models to compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64), activation='relu', 
                                          solver='adam', alpha=0.001, batch_size=32, learning_rate='adaptive', 
                                          max_iter=200, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
    }
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        results[name] = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"  Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results

def plot_comparison_results(results):
    """
    Plot the comparison results.
    """
    # Prepare data for plotting
    model_names = list(results.keys())
    mean_accuracies = [results[name]['mean_accuracy'] for name in model_names]
    std_accuracies = [results[name]['std_accuracy'] for name in model_names]
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    
    # Bar plot
    bars = plt.bar(model_names, mean_accuracies, yerr=std_accuracies, 
                   capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for bar, mean_acc in zip(bars, mean_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Machine Learning Model Comparison for Leishmania Species Classification', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Cross-Validation Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add text box with summary
    best_model = max(results.keys(), key=lambda x: results[x]['mean_accuracy'])
    best_accuracy = results[best_model]['mean_accuracy']
    
    textstr = f'Best Model: {best_model}\nAccuracy: {best_accuracy:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_model

def detailed_analysis(X, y, best_model_name):
    """
    Perform detailed analysis of the best model.
    """
    print(f"\nPerforming detailed analysis of {best_model_name}...")
    
    # Define the best model
    if best_model_name == 'Random Forest':
        best_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    elif best_model_name == 'MLP Neural Network':
        best_model = MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64), activation='relu', 
                                  solver='adam', alpha=0.001, batch_size=32, learning_rate='adaptive', 
                                  max_iter=200, random_state=42)
    elif best_model_name == 'SVM (RBF)':
        best_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    elif best_model_name == 'Logistic Regression':
        best_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
    
    # Train the model
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance (for Random Forest)
    if best_model_name == 'Random Forest':
        feature_importance = best_model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(20), feature_importance[top_features_idx])
        plt.yticks(range(20), [f'Protein_{i}' for i in top_features_idx])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Proteins for Species Classification')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function for model comparison.
    """
    print("=" * 60)
    print("MACHINE LEARNING MODEL COMPARISON")
    print("=" * 60)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Compare models
    results = compare_models(X, y)
    
    # Plot results
    best_model = plot_comparison_results(results)
    
    # Detailed analysis of best model
    detailed_analysis(X, y, best_model)
    
    print(f"\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print("=" * 60)
    print(f"Best performing model: {best_model}")
    print(f"Best accuracy: {results[best_model]['mean_accuracy']:.4f}")
    print("\nResults saved to 'model_comparison.png' and 'feature_importance.png'")

if __name__ == "__main__":
    main()
