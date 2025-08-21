#!/usr/bin/env python3
"""
Gene-Filtered Protein Intensity Analysis
=======================================

This script performs protein intensity analysis using ONLY protein groups
that map to unique genes (same filtering as gene-based analysis).

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import mannwhitneyu, kruskal
import os

class GeneFilteredProteinIntensityAnalyzer:
    def __init__(self, data_file="../raw_data.txt.gz"):
        """
        Initialize the analyzer with raw data file.
        
        Args:
            data_file (str): Path to the raw_data.txt.gz file
        """
        self.data_file = data_file
        self.df = None
        self.intensity_columns = []
        self.species_mapping = {}
        
    def extract_gene_from_fasta_header(self, fasta_header):
        """Extract gene names from Fasta header."""
        genes = []
        if 'GN=' in fasta_header:
            gene_matches = re.findall(r'GN=([^=]+)', fasta_header)
            genes.extend(gene_matches)
        return genes
    
    def load_and_process_data(self):
        """Load and process the raw data file with gene filtering."""
        print("Loading and processing data with gene filtering...")
        
        # Read the data file (handle gzipped file)
        if self.data_file.endswith('.gz'):
            import gzip
            with gzip.open(self.data_file, 'rt') as f:
                lines = f.readlines()
        else:
            with open(self.data_file, 'r') as f:
                lines = f.readlines()
        
        # Parse header to find intensity columns
        header = lines[0].strip().split('\t')
        self.intensity_columns = [col for col in header if col.startswith('Intensity ')]
        
        # Create species mapping
        for col in self.intensity_columns:
            if 'Lb_' in col:
                self.species_mapping[col] = 'Lb'
            elif 'Lg_' in col:
                self.species_mapping[col] = 'Lg'
            elif 'Ln_' in col:
                self.species_mapping[col] = 'Ln'
            elif 'Lp_' in col:
                self.species_mapping[col] = 'Lp'
        
        print(f"Found {len(self.intensity_columns)} intensity columns")
        species_counts = {}
        for sp in set(self.species_mapping.values()):
            species_counts[sp] = sum(1 for v in self.species_mapping.values() if v == sp)
        print(f"Species mapping: {species_counts}")
        
        # Parse data lines with gene filtering
        data_rows = []
        gene_filtered_count = 0
        total_count = 0
        
        for line in lines[1:]:
            total_count += 1
            parts = line.strip().split('\t')
            if len(parts) < len(header):
                continue
                
            # Check for decoys and contaminants
            reverse = parts[272].strip() if len(parts) > 272 else ''
            contaminant = parts[273].strip() if len(parts) > 273 else ''
            
            if reverse == '+' or contaminant == '+':
                continue
            
            # Extract protein info
            protein_ids = parts[0].strip()
            if not protein_ids:
                continue
            
            # GENE FILTERING: Check if protein group maps to unique gene
            fasta_header = parts[5].strip() if len(parts) > 5 else ''  # Fasta headers column
            genes = self.extract_gene_from_fasta_header(fasta_header)
            unique_genes = list(set(genes))
            
            # Only include if maps to exactly one gene
            if len(unique_genes) != 1:
                continue
            
            gene_filtered_count += 1
            
            # Extract intensities
            row_data = {
                'Protein_IDs': protein_ids,
                'Gene_ID': unique_genes[0],
                'Fasta_Header': fasta_header
            }
            
            for i, col in enumerate(self.intensity_columns):
                col_idx = header.index(col)
                if col_idx < len(parts):
                    try:
                        intensity = float(parts[col_idx]) if parts[col_idx].strip() else np.nan
                    except ValueError:
                        intensity = np.nan
                    row_data[col] = intensity
            
            data_rows.append(row_data)
        
        self.df = pd.DataFrame(data_rows)
        print(f"Total proteins processed: {total_count}")
        print(f"Proteins after gene filtering: {gene_filtered_count}")
        print(f"Loaded {len(self.df)} proteins after filtering decoys/contaminants AND gene filtering")
        
        # Filter out undetected proteins (zero intensity across all species)
        print("Filtering out undetected proteins...")
        total_intensity = self.df[self.intensity_columns].sum(axis=1)
        self.df = self.df[total_intensity > 0]
        print(f"Removed undetected proteins. Remaining proteins: {len(self.df)}")
        
        # Calculate median intensities per protein
        self._calculate_median_intensities()
        
    def _calculate_median_intensities(self):
        """Calculate median intensity per protein per species, then log transform (excluding zeros)."""
        print("Calculating median intensities per protein per species (excluding zeros)...")
        
        # Calculate median for each protein per species (excluding zeros)
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            species_cols = [col for col in self.intensity_columns if self.species_mapping[col] == species]
            if species_cols:
                # Calculate median excluding zeros
                species_data = self.df[species_cols].copy()
                # Replace zeros with NaN for median calculation
                species_data = species_data.replace(0, np.nan)
                self.df[f'Median_{species}'] = species_data.median(axis=1)
                # Only apply log2 to non-zero values
                self.df[f'Log2_Median_{species}'] = np.log2(self.df[f'Median_{species}'].fillna(0) + 1)
        
        # Also calculate overall median for reference (excluding zeros)
        overall_data = self.df[self.intensity_columns].copy()
        overall_data = overall_data.replace(0, np.nan)
        self.df['Median_Intensity_Overall'] = overall_data.median(axis=1)
        self.df['Log2_Median_Intensity_Overall'] = np.log2(self.df['Median_Intensity_Overall'].fillna(0) + 1)
        
        # Print summary statistics
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            if f'Median_{species}' in self.df.columns:
                median_col = f'Median_{species}'
                log_col = f'Log2_Median_{species}'
                non_zero_medians = self.df[median_col].dropna()
                non_zero_log = self.df[log_col][self.df[median_col].notna()]
                
                print(f"{species} - Non-zero median intensity range: {non_zero_medians.min():.2f} - {non_zero_medians.max():.2f}")
                print(f"{species} - Non-zero log2 median intensity range: {non_zero_log.min():.2f} - {non_zero_log.max():.2f}")
                print(f"{species} - Proteins with non-zero median: {len(non_zero_medians)} / {len(self.df)} ({len(non_zero_medians)/len(self.df)*100:.1f}%)")
        
        non_zero_overall = self.df['Median_Intensity_Overall'].dropna()
        non_zero_log_overall = self.df['Log2_Median_Intensity_Overall'][self.df['Median_Intensity_Overall'].notna()]
        print(f"Overall - Non-zero median intensity range: {non_zero_overall.min():.2f} - {non_zero_overall.max():.2f}")
        print(f"Overall - Non-zero log2 median intensity range: {non_zero_log_overall.min():.2f} - {non_zero_log_overall.max():.2f}")
        print(f"Overall - Proteins with non-zero median: {len(non_zero_overall)} / {len(self.df)} ({len(non_zero_overall)/len(self.df)*100:.1f}%)")
    
    def create_violin_plots(self, output_file="gene_filtered_intensity_violin_plots.png"):
        """Create violin plots showing intensity distribution per species using median intensities."""
        print("Creating violin plots using species-specific medians (gene-filtered)...")
        
        # Prepare data for violin plot using species-specific medians (non-zero only)
        plot_data = []
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            median_col = f'Median_{species}'
            if median_col in self.df.columns:
                non_zero_data = self.df[median_col].dropna()
                for value in non_zero_data:
                    plot_data.append({'Species': species, 'Log2_Median_Intensity': np.log2(value + 1)})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create violin plot
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=plot_df, x='Species', y='Log2_Median_Intensity', 
                      palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        plt.title('Gene-Filtered Protein Intensity Distribution by Species\n(Only proteins mapping to unique genes)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Species', fontsize=12)
        plt.ylabel('Log2 Median Intensity', fontsize=12)
        
        # Add statistics
        for i, species in enumerate(['Lb', 'Lg', 'Ln', 'Lp']):
            species_data = plot_df[plot_df['Species'] == species]['Log2_Median_Intensity']
            if len(species_data) > 0:
                mean_val = species_data.mean()
                median_val = species_data.median()
                plt.text(i, plt.ylim()[1] * 0.95, f'n={len(species_data)}\nμ={mean_val:.2f}\nmed={median_val:.2f}', 
                        ha='center', va='top', fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Violin plots saved to: {output_file}")
    
    def create_histograms(self, output_file="gene_filtered_intensity_histograms.png"):
        """Create histograms showing intensity distribution per species."""
        print("Creating intensity histograms (gene-filtered)...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, species in enumerate(['Lb', 'Lg', 'Ln', 'Lp']):
            median_col = f'Median_{species}'
            if median_col in self.df.columns:
                non_zero_data = self.df[median_col].dropna()
                log_data = np.log2(non_zero_data + 1)
                
                axes[i].hist(log_data, bins=50, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][i])
                axes[i].set_title(f'{species} - Gene-Filtered Protein Intensity Distribution', fontsize=12)
                axes[i].set_xlabel('Log2 Median Intensity', fontsize=10)
                axes[i].set_ylabel('Frequency', fontsize=10)
                
                # Add statistics
                mean_val = log_data.mean()
                median_val = log_data.median()
                axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
                axes[i].legend()
                axes[i].text(0.02, 0.98, f'n={len(log_data)}', transform=axes[i].transAxes, 
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Histograms saved to: {output_file}")
    
    def create_ma_plots(self, output_file="gene_filtered_ma_plots.png"):
        """Create MA plots for differential expression analysis."""
        print("Creating MA plots for differential expression analysis (gene-filtered)...")
        
        # Create MA plots for each species pair
        species_pairs = [('Lb', 'Lg'), ('Lb', 'Ln'), ('Lb', 'Lp'), ('Lg', 'Ln'), ('Lg', 'Lp'), ('Ln', 'Lp')]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, (sp1, sp2) in enumerate(species_pairs):
            median_col1 = f'Median_{sp1}'
            median_col2 = f'Median_{sp2}'
            
            if median_col1 in self.df.columns and median_col2 in self.df.columns:
                # Get non-zero data for both species
                valid_data = self.df[[median_col1, median_col2]].dropna()
                
                if len(valid_data) > 0:
                    # Calculate M and A values
                    log2_sp1 = np.log2(valid_data[median_col1] + 1)
                    log2_sp2 = np.log2(valid_data[median_col2] + 1)
                    
                    M = log2_sp1 - log2_sp2  # Log fold change
                    A = (log2_sp1 + log2_sp2) / 2  # Average expression
                    
                    # Color code based on fold change
                    colors = []
                    for m_val in M:
                        if abs(m_val) > 1:  # Log2 fold change > 2
                            if m_val > 0:
                                colors.append('red')  # Overexpressed in sp1
                            else:
                                colors.append('blue')  # Downregulated in sp1
                        else:
                            colors.append('grey')  # No significant change
                    
                    axes[idx].scatter(A, M, c=colors, alpha=0.6, s=20)
                    axes[idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                    axes[idx].axhline(y=1, color='red', linestyle='--', alpha=0.5)
                    axes[idx].axhline(y=-1, color='blue', linestyle='--', alpha=0.5)
                    
                    axes[idx].set_title(f'{sp1} vs {sp2} - Gene-Filtered MA Plot', fontsize=12)
                    axes[idx].set_xlabel('A (Average Expression)', fontsize=10)
                    axes[idx].set_ylabel('M (Log Fold Change)', fontsize=10)
                    
                    # Add statistics
                    overexpressed = sum(1 for m in M if m > 1)
                    downregulated = sum(1 for m in M if m < -1)
                    total = len(M)
                    axes[idx].text(0.02, 0.98, f'Overexpressed: {overexpressed}\nDownregulated: {downregulated}\nTotal: {total}', 
                                 transform=axes[idx].transAxes, fontsize=9, verticalalignment='top',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"MA plots saved to: {output_file}")
    
    def perform_statistical_tests(self, output_file="gene_filtered_statistical_tests.txt"):
        """Perform statistical tests comparing intensity distributions between species."""
        print("Performing statistical tests (gene-filtered)...")
        
        results = []
        results.append("GENE-FILTERED PROTEIN INTENSITY ANALYSIS - STATISTICAL TESTS")
        results.append("=" * 70)
        results.append("")
        
        # Prepare data for statistical tests
        species_data = {}
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            median_col = f'Median_{species}'
            if median_col in self.df.columns:
                non_zero_data = self.df[median_col].dropna()
                log_data = np.log2(non_zero_data + 1)
                species_data[species] = log_data
                results.append(f"{species}: n={len(log_data)}, mean={log_data.mean():.3f}, median={log_data.median():.3f}")
        
        results.append("")
        
        # Kruskal-Wallis test for all species
        if len(species_data) >= 3:
            all_data = list(species_data.values())
            all_labels = list(species_data.keys())
            
            try:
                h_stat, p_value = kruskal(*all_data)
                results.append(f"KRUSKAL-WALLIS TEST (All species):")
                results.append(f"H-statistic: {h_stat:.4f}")
                results.append(f"p-value: {p_value:.4e}")
                results.append(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
                results.append("")
            except Exception as e:
                results.append(f"Kruskal-Wallis test failed: {e}")
                results.append("")
        
        # Mann-Whitney U tests for species pairs
        species_pairs = [('Lb', 'Lg'), ('Lb', 'Ln'), ('Lb', 'Lp'), ('Lg', 'Ln'), ('Lg', 'Lp'), ('Ln', 'Lp')]
        
        results.append("MANN-WHITNEY U TESTS (Species pairs):")
        results.append("-" * 50)
        
        for sp1, sp2 in species_pairs:
            if sp1 in species_data and sp2 in species_data:
                try:
                    u_stat, p_value = mannwhitneyu(species_data[sp1], species_data[sp2], alternative='two-sided')
                    results.append(f"{sp1} vs {sp2}:")
                    results.append(f"  U-statistic: {u_stat:.4f}")
                    results.append(f"  p-value: {p_value:.4e}")
                    results.append(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
                    results.append("")
                except Exception as e:
                    results.append(f"{sp1} vs {sp2}: Test failed - {e}")
                    results.append("")
        
        # Write results to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(results))
        
        print(f"Statistical tests saved to: {output_file}")
    
    def save_processed_data(self, output_file="gene_filtered_processed_intensity_data.csv"):
        """Save the processed data to CSV."""
        print("Saving processed data...")
        self.df.to_csv(output_file, index=False)
        print(f"Processed data saved to: {output_file}")
    
    def create_comprehensive_report(self, output_file="gene_filtered_comprehensive_analysis_report.png"):
        """Create a comprehensive analysis report with multiple plots."""
        print("Creating comprehensive analysis report (gene-filtered)...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Violin plot (top left)
        ax1 = plt.subplot(2, 3, 1)
        plot_data = []
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            median_col = f'Median_{species}'
            if median_col in self.df.columns:
                non_zero_data = self.df[median_col].dropna()
                for value in non_zero_data:
                    plot_data.append({'Species': species, 'Log2_Median_Intensity': np.log2(value + 1)})
        
        plot_df = pd.DataFrame(plot_data)
        sns.violinplot(data=plot_df, x='Species', y='Log2_Median_Intensity', 
                      palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], ax=ax1)
        ax1.set_title('Gene-Filtered Intensity Distribution', fontsize=12, fontweight='bold')
        
        # 2. Box plot (top middle)
        ax2 = plt.subplot(2, 3, 2)
        sns.boxplot(data=plot_df, x='Species', y='Log2_Median_Intensity', 
                   palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], ax=ax2)
        ax2.set_title('Gene-Filtered Intensity Box Plot', fontsize=12, fontweight='bold')
        
        # 3. Histogram (top right)
        ax3 = plt.subplot(2, 3, 3)
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            median_col = f'Median_{species}'
            if median_col in self.df.columns:
                non_zero_data = self.df[median_col].dropna()
                log_data = np.log2(non_zero_data + 1)
                ax3.hist(log_data, bins=30, alpha=0.6, label=species)
        ax3.set_title('Gene-Filtered Intensity Histograms', fontsize=12, fontweight='bold')
        ax3.legend()
        
        # 4. MA plot example (bottom left)
        ax4 = plt.subplot(2, 3, 4)
        median_col1 = 'Median_Lb'
        median_col2 = 'Median_Lg'
        if median_col1 in self.df.columns and median_col2 in self.df.columns:
            valid_data = self.df[[median_col1, median_col2]].dropna()
            if len(valid_data) > 0:
                log2_sp1 = np.log2(valid_data[median_col1] + 1)
                log2_sp2 = np.log2(valid_data[median_col2] + 1)
                M = log2_sp1 - log2_sp2
                A = (log2_sp1 + log2_sp2) / 2
                ax4.scatter(A, M, alpha=0.6, s=20)
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax4.set_title('Lb vs Lg - Gene-Filtered MA Plot', fontsize=12, fontweight='bold')
                ax4.set_xlabel('A (Average Expression)')
                ax4.set_ylabel('M (Log Fold Change)')
        
        # 5. Summary statistics (bottom middle)
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        stats_text = "GENE-FILTERED ANALYSIS SUMMARY\n\n"
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            median_col = f'Median_{species}'
            if median_col in self.df.columns:
                non_zero_data = self.df[median_col].dropna()
                log_data = np.log2(non_zero_data + 1)
                stats_text += f"{species}:\n"
                stats_text += f"  n = {len(log_data)}\n"
                stats_text += f"  mean = {log_data.mean():.2f}\n"
                stats_text += f"  median = {log_data.median():.2f}\n\n"
        
        stats_text += f"Total proteins analyzed: {len(self.df)}\n"
        stats_text += f"All proteins mapping to unique genes"
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        # 6. Correlation heatmap (bottom right)
        ax6 = plt.subplot(2, 3, 6)
        correlation_data = []
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            median_col = f'Median_{species}'
            if median_col in self.df.columns:
                correlation_data.append(self.df[median_col].fillna(0))
        
        if len(correlation_data) >= 2:
            corr_df = pd.DataFrame(correlation_data, index=['Lb', 'Lg', 'Ln', 'Lp'][:len(correlation_data)]).T
            corr_matrix = corr_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax6)
            ax6.set_title('Gene-Filtered Intensity Correlation', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comprehensive report saved to: {output_file}")
    
    def run_complete_analysis(self):
        """Run the complete gene-filtered protein intensity analysis."""
        print("🚀 Starting gene-filtered protein intensity analysis...")
        print("=" * 60)
        
        # Load and process data
        self.load_and_process_data()
        
        # Create output directory
        output_dir = "gene_filtered_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all analyses
        print("\n📊 Generating analyses...")
        
        # Create visualizations
        self.create_violin_plots(os.path.join(output_dir, "gene_filtered_intensity_violin_plots.png"))
        self.create_histograms(os.path.join(output_dir, "gene_filtered_intensity_histograms.png"))
        self.create_ma_plots(os.path.join(output_dir, "gene_filtered_ma_plots.png"))
        self.create_comprehensive_report(os.path.join(output_dir, "gene_filtered_comprehensive_analysis_report.png"))
        
        # Perform statistical tests
        self.perform_statistical_tests(os.path.join(output_dir, "gene_filtered_statistical_tests.txt"))
        
        # Save processed data
        self.save_processed_data(os.path.join(output_dir, "gene_filtered_processed_intensity_data.csv"))
        
        print(f"\n✅ Gene-filtered protein intensity analysis complete!")
        print(f"📁 Results saved to: {output_dir}/")
        
        return self.df

if __name__ == "__main__":
    analyzer = GeneFilteredProteinIntensityAnalyzer()
    df = analyzer.run_complete_analysis()
