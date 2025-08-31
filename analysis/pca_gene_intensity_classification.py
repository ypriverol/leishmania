#!/usr/bin/env python3
"""
t-SNE Gene Intensity Classification Script
=========================================

This script performs t-SNE (t-Distributed Stochastic Neighbor Embedding) on gene-filtered protein intensity data
to classify samples by species. It uses the same data processing approach as the gene-filtered
protein intensity analysis but focuses on t-SNE visualization for sample classification.

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import sys
import os

# Add parent directory to path to import data_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processor import DataProcessor

warnings.filterwarnings('ignore')

class TSNEGeneIntensityClassifier:
    def __init__(self, data_file="../raw_data.txt.gz"):
        """
        Initialize the t-SNE classifier with data processor.
        
        Args:
            data_file (str): Path to the raw data file (supports .txt and .gz formats)
        """
        self.data_file = data_file
        self.processor = None
        self.protein_df = None
        self.gene_df = None
        self.sample_info_df = None
        
    def load_and_process_data(self):
        """Load and process data using the common data processor with gene filtering."""
        print("🔄 Loading and processing data with gene filtering...")
        
        # Initialize data processor
        self.processor = DataProcessor(self.data_file)
        
        # Load and process data
        self.protein_df, self.gene_df = self.processor.load_and_process_data()
        self.sample_info_df = self.processor.sample_info_df
        
        # Use gene_df (which contains only proteins with unique gene mapping)
        self.df = self.gene_df
        self.intensity_columns = self.processor.intensity_columns
        self.species_mapping = self.processor.species_mapping
        
        print(f"✅ Loaded {len(self.df)} proteins with unique gene mapping")
        
    def prepare_data_for_tsne(self):
        """Prepare intensity data for t-SNE analysis."""
        print("🔬 Preparing data for t-SNE analysis...")
        
        # Get intensity columns (sample data)
        intensity_data = self.df[self.intensity_columns].copy()
        
        # Replace zeros with NaN for better handling
        intensity_data = intensity_data.replace(0, np.nan)
        
        # Log2 transform the data (add 1 to avoid log(0))
        intensity_data_log2 = np.log2(intensity_data.fillna(0) + 1)
        
        # Remove proteins with too many missing values (>50% missing)
        missing_threshold = len(intensity_data.columns) * 0.5
        proteins_with_data = intensity_data_log2.notna().sum(axis=1) >= missing_threshold
        intensity_data_filtered = intensity_data_log2[proteins_with_data]
        
        print(f"✅ Prepared {len(intensity_data_filtered)} proteins for t-SNE analysis")
        print(f"✅ Using {len(intensity_data_filtered.columns)} samples")
        
        return intensity_data_filtered
    
    def perform_tsne_analysis(self, intensity_data, output_file="gene-based-pca/tsne_gene_intensity_classification.png"):
        """Perform t-SNE analysis and create visualization."""
        print("📊 Performing t-SNE analysis...")
        
        # Transpose data so samples are rows and proteins are columns
        data_for_tsne = intensity_data.T
        
        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_for_tsne)
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data_scaled)-1), n_iter=1000)
        tsne_result = tsne.fit_transform(data_scaled)
        
        # Create DataFrame with t-SNE results
        tsne_df = pd.DataFrame(
            data=tsne_result,
            columns=['tSNE1', 'tSNE2'],
            index=data_for_tsne.index
        )
        
        # Add species information
        tsne_df['Species'] = tsne_df.index.map(self.species_mapping)
        tsne_df['Sample'] = tsne_df.index
        
        # Create Leishmania color scheme
        leishmania_colors = {
            'Lb': '#1f77b4',  # Blue
            'Lg': '#2ca02c',  # Green
            'Ln': '#d62728',  # Red
            'Lp': '#ff7f0e'   # Orange
        }
        
        # Create single figure with only the main t-SNE plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.suptitle('t-SNE Analysis of Gene-Filtered Protein Intensity Data\n(Only proteins mapping to unique genes)', 
                     fontsize=16, fontweight='bold')
        
        # t-SNE Scatter Plot
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            species_data = tsne_df[tsne_df['Species'] == species]
            if len(species_data) > 0:
                ax.scatter(species_data['tSNE1'], species_data['tSNE2'], 
                          c=leishmania_colors[species], label=species, 
                          s=120, alpha=0.8, edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title('t-SNE: Sample Classification by Species', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add information text box
        info_text = f"""t-SNE Parameters:
• Perplexity: {min(30, len(data_scaled)-1)}
• Iterations: 1000
• Random State: 42
• Components: 2

Data Information:
• Total Samples: {len(tsne_df)}
• Total Proteins: {intensity_data.shape[0]}
• Species: {', '.join(tsne_df['Species'].unique())}

t-SNE preserves local structure
and reveals clusters in high-dimensional data."""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ t-SNE analysis plots saved to: {output_file}")
        
        # Save t-SNE results
        tsne_results_file = output_file.replace('.png', '_results.csv')
        tsne_df.to_csv(tsne_results_file)
        print(f"✅ t-SNE results saved to: {tsne_results_file}")
        
        return tsne_df
    
    def perform_statistical_analysis(self, tsne_df, output_file="gene-based-pca/tsne_statistical_analysis.txt"):
        """Perform statistical analysis on t-SNE results."""
        print("🔬 Performing statistical analysis on t-SNE results...")
        
        results = []
        results.append("t-SNE GENE INTENSITY CLASSIFICATION - STATISTICAL ANALYSIS")
        results.append("=" * 60)
        results.append("")
        
        # ANOVA test for tSNE1 and tSNE2 by species
        for component in ['tSNE1', 'tSNE2']:
            results.append(f"ANOVA TEST FOR {component} BY SPECIES:")
            results.append("-" * 40)
            
            # Prepare data for ANOVA
            species_data = {}
            for species in ['Lb', 'Lg', 'Ln', 'Lp']:
                species_data[species] = tsne_df[tsne_df['Species'] == species][component].values
            
            # Perform one-way ANOVA
            try:
                f_stat, p_value = stats.f_oneway(*species_data.values())
                results.append(f"F-statistic: {f_stat:.4f}")
                results.append(f"p-value: {p_value:.4e}")
                results.append(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
                results.append("")
                
                # Post-hoc tests (Tukey's HSD)
                results.append("POST-HOC PAIRWISE COMPARISONS (Tukey's HSD):")
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                
                # Prepare data for Tukey's test
                all_values = []
                all_labels = []
                for species, values in species_data.items():
                    all_values.extend(values)
                    all_labels.extend([species] * len(values))
                
                tukey_result = pairwise_tukeyhsd(all_values, all_labels)
                results.append(str(tukey_result))
                results.append("")
                
            except Exception as e:
                results.append(f"Statistical test failed: {e}")
                results.append("")
        
        # Save results
        with open(output_file, 'w') as f:
            f.write('\n'.join(results))
        
        print(f"✅ Statistical analysis saved to: {output_file}")
    
    def run_complete_analysis(self):
        """Run the complete t-SNE analysis pipeline."""
        print("🚀 Starting t-SNE gene intensity classification analysis...")
        print("=" * 60)
        
        # Load and process data
        self.load_and_process_data()
        
        # Prepare data for t-SNE
        intensity_data = self.prepare_data_for_tsne()
        
        # Create output directory
        output_dir = "gene-based-pca"
        os.makedirs(output_dir, exist_ok=True)
        
        # Perform t-SNE analysis
        tsne_df = self.perform_tsne_analysis(
            intensity_data, 
            os.path.join(output_dir, "tsne_gene_intensity_classification.png")
        )
        
        # Perform statistical analysis
        self.perform_statistical_analysis(
            tsne_df, 
            os.path.join(output_dir, "tsne_statistical_analysis.txt")
        )
        
        # Print summary
        print("\n📊 t-SNE ANALYSIS SUMMARY:")
        print("=" * 40)
        print(f"Total samples analyzed: {len(tsne_df)}")
        print(f"Total proteins used: {intensity_data.shape[0]}")
        print(f"t-SNE components: 2")
        print(f"Perplexity: {min(30, len(intensity_data.columns)-1)}")
        print(f"Iterations: 1000")
        
        # Sample distribution
        species_counts = tsne_df['Species'].value_counts()
        print("\n📋 Sample Distribution:")
        for species, count in species_counts.items():
            print(f"  {species}: {count} samples")
        
        print("\n✅ t-SNE gene intensity classification analysis complete!")
        print(f"📁 Results saved to: {output_dir}/")

if __name__ == "__main__":
    # Create and run the t-SNE classifier
    tsne_classifier = TSNEGeneIntensityClassifier()
    tsne_classifier.run_complete_analysis()
