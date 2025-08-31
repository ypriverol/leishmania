#!/usr/bin/env python3
"""
Protein Intensity Analysis Script
================================

This script performs comprehensive analysis of protein intensities across species:
1. Distribution of intensities per species (violin plots and histograms)
2. Median intensity calculation per protein across samples
3. MA plots for differential analysis between species
4. Statistical comparisons between species

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
import warnings
import sys
import os

# Add parent directory to path to import data_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processor import DataProcessor

warnings.filterwarnings('ignore')

class ProteinIntensityAnalyzer:
    def __init__(self, data_file="../raw_data.txt.gz"):
        """
        Initialize the analyzer with data processor.
        
        Args:
            data_file (str): Path to the raw data file (supports .txt and .gz formats)
        """
        self.data_file = data_file
        self.processor = None
        self.protein_df = None
        self.gene_df = None
        self.sample_info_df = None
        
    def load_and_process_data(self):
        """Load and process data using the common data processor."""
        print("🔄 Loading and processing data using common data processor...")
        
        # Initialize data processor
        self.processor = DataProcessor(self.data_file)
        
        # Load and process data
        self.protein_df, self.gene_df = self.processor.load_and_process_data()
        self.sample_info_df = self.processor.sample_info_df
        
        # Set up for compatibility with existing methods
        self.df = self.protein_df  # Use protein_df as main dataframe
        self.intensity_columns = self.processor.intensity_columns
        self.species_mapping = self.processor.species_mapping
        
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
    
    def create_comprehensive_intensity_plots(self, output_file="analysis/intensity_distributions.png"):
        """Create comprehensive intensity distribution plots with correct Leishmania color scheme."""
        print("Creating comprehensive intensity distribution plots with Leishmania color scheme...")
        
        # Define Leishmania color scheme (using better hex colors from radial phylogenetic analyzer)
        leishmania_colors = {
            'Lb': '#1f77b4',   # Leishmania braziliensis - blue
            'Lg': '#ff7f0e',   # L. guyanensis - orange
            'Ln': '#d62728',   # L. naiffi - red
            'Lp': '#2ca02c'    # L. panamensis - green
        }
        
        # Prepare data for plots using species-specific medians (non-zero only)
        plot_data = []
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            median_col = f'Median_{species}'
            log_col = f'Log2_Median_{species}'
            
            if median_col in self.df.columns:
                # Only use non-zero medians
                non_zero_mask = self.df[median_col].notna() & (self.df[median_col] > 0)
                medians = self.df[median_col][non_zero_mask]
                log_medians = self.df[log_col][non_zero_mask]
                
                for median, log_median in zip(medians, log_medians):
                    plot_data.append({
                        'Species': species,
                        'Median_Intensity': median,
                        'Log2_Median_Intensity': log_median
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create figure with subplots (2x1 layout - removed redundant panels)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Leishmania Protein Intensity Distributions (Log2-Transformed)', fontsize=16, fontweight='bold')
        
        # 1. Log2 intensity violin plot (top)
        sns.violinplot(data=plot_df, x='Species', y='Log2_Median_Intensity', 
                      palette=leishmania_colors, ax=axes[0], alpha=0.7)
        axes[0].set_title('Log2 Median Intensity Distribution (Violin Plot)', fontweight='bold')
        axes[0].set_ylabel('Log2(Median Intensity + 1)')
        axes[0].grid(True, alpha=0.2)
        
        # 2. Log2 intensity histogram (bottom)
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            species_data = plot_df[plot_df['Species'] == species]['Log2_Median_Intensity']
            if len(species_data) > 0:
                n_bins = int(np.ceil(1 + 3.322 * np.log10(len(species_data))))
                axes[1].hist(species_data, bins=n_bins, alpha=0.7, 
                           label=species, density=True, color=leishmania_colors[species],
                           edgecolor='white', linewidth=0.5)
        
        axes[1].set_title('Log2 Median Intensity Distribution (Histogram)', fontweight='bold')
        axes[1].set_xlabel('Log2(Median Intensity + 1)')
        axes[1].set_ylabel('Density')
        axes[1].legend(framealpha=0.9)
        axes[1].grid(True, alpha=0.2)
        
        # Save statistical summary to separate file
        stats_data = plot_df.groupby('Species')['Log2_Median_Intensity'].agg(['count', 'mean', 'std', 'median', 'min', 'max']).round(3)
        stats_file = output_file.replace('.png', '_statistics.csv')
        stats_data.to_csv(stats_file)
        print(f"✅ Statistical summary saved to: {stats_file}")
        
        # Add color legend
        legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=f'{species}') 
                          for species, color in leishmania_colors.items()]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=4, title='Leishmania Species', framealpha=0.9, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive intensity plots saved to: {output_file}")
        return plot_df
    

    
    def create_ma_plots(self, output_file="analysis/proteins_ma_plots.png"):
        """Create MA plots for differential analysis between species pairs using pre-calculated medians with statistical testing."""
        print("Creating MA plots using species-specific medians with statistical testing...")
        
        # Use pre-calculated species medians (non-zero only)
        species_medians = {}
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            median_col = f'Median_{species}'
            if median_col in self.df.columns:
                # Only use non-zero medians
                non_zero_mask = self.df[median_col].notna() & (self.df[median_col] > 0)
                species_medians[species] = self.df[median_col][non_zero_mask]
        
        # Create MA plots for all species pairs
        species_pairs = [('Lb', 'Lg'), ('Lb', 'Ln'), ('Lb', 'Lp'), 
                        ('Lg', 'Ln'), ('Lg', 'Lp'), ('Ln', 'Lp')]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MA Plots for Differential Analysis Between Species - Statistical Significance with Benjamini-Hochberg', fontsize=16, fontweight='bold')
        
        # Collect data for export
        export_data = []
        
        for idx, (sp1, sp2) in enumerate(species_pairs):
            if sp1 in species_medians and sp2 in species_medians:
                row = idx // 3
                col = idx % 3
                ax = axes[row, col]
                
                # Get non-zero medians for both species
                intensity1 = species_medians[sp1]
                intensity2 = species_medians[sp2]
                
                # Find common proteins (proteins present in both species)
                common_proteins = intensity1.index.intersection(intensity2.index)
                intensity1_common = intensity1[common_proteins]
                intensity2_common = intensity2[common_proteins]
                
                if len(intensity1_common) > 0:
                    # Calculate M and A
                    M = np.log2(intensity1_common + 1) - np.log2(intensity2_common + 1)
                    A = (np.log2(intensity1_common + 1) + np.log2(intensity2_common + 1)) / 2
                    
                    # Perform statistical testing for each protein using individual sample data
                    p_values = []
                    for protein_id in intensity1_common.index:
                        # Get all sample columns for each species
                        sp1_cols = [col for col in self.df.columns if col.startswith('Intensity ') and f'{sp1}_' in col]
                        sp2_cols = [col for col in self.df.columns if col.startswith('Intensity ') and f'{sp2}_' in col]
                        
                        if sp1_cols and sp2_cols:
                            # Get individual sample intensities (not medians)
                            sp1_values = self.df.loc[protein_id, sp1_cols].dropna()
                            sp2_values = self.df.loc[protein_id, sp2_cols].dropna()
                            
                            # Filter out zero values (not detected in that sample)
                            sp1_values = sp1_values[sp1_values > 0]
                            sp2_values = sp2_values[sp2_values > 0]
                            
                            # Only test if we have enough data points (at least 2 samples per species)
                            if len(sp1_values) >= 2 and len(sp2_values) >= 2:
                                # Convert to numpy arrays for statistical testing
                                sp1_array = sp1_values.values.astype(float)
                                sp2_array = sp2_values.values.astype(float)
                                
                                # Perform Mann-Whitney U test on individual sample intensities
                                try:
                                    stat, p_val = stats.mannwhitneyu(sp1_array, sp2_array, alternative='two-sided')
                                    p_values.append(p_val)
                                except Exception as e:
                                    # If test fails, try with log-transformed values
                                    try:
                                        log_sp1 = np.log2(sp1_array + 1)
                                        log_sp2 = np.log2(sp2_array + 1)
                                        stat, p_val = stats.mannwhitneyu(log_sp1, log_sp2, alternative='two-sided')
                                        p_values.append(p_val)
                                    except:
                                        p_values.append(1.0)  # No significant difference
                            else:
                                p_values.append(1.0)  # Not enough data
                        else:
                            p_values.append(1.0)  # No data
                    
                    # Apply Benjamini-Hochberg correction
                    if p_values:
                        rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
                    else:
                        rejected = [False] * len(M)
                        p_corrected = [1.0] * len(M)
                    
                    # Define fold change threshold (0.05 = log2 fold change of ~-4.32)
                    fc_threshold = 0.05
                    log2_fc_threshold = np.log2(fc_threshold)
                    
                    # Color coding based on fold change AND statistical significance
                    colors = []
                    for i, (protein_id, m_val, p_corr) in enumerate(zip(intensity1_common.index, M, p_corrected)):
                        # Get the actual protein accession from the Protein_IDs column
                        actual_protein_accession = self.df.loc[protein_id, 'Protein_IDs']
                        
                        # Determine category based on fold change AND statistical significance
                        if p_corr > 0.05:  # Not statistically significant
                            colors.append('lightgrey')  # Not significant
                            category = 'Not significant'
                        elif abs(m_val) < abs(log2_fc_threshold):
                            colors.append('grey')  # Significant but small fold change
                            category = 'Significant (small FC)'
                        elif m_val > abs(log2_fc_threshold):
                            colors.append('red')   # Overexpressed in sp1
                            category = 'Overexpressed'
                        else:
                            colors.append('blue')  # Downregulated in sp1
                            category = 'Downregulated'
                        
                        # Add to export data
                        export_data.append({
                            'Protein_Group_Accession': actual_protein_accession,
                            'Species_Comparison': f'{sp1}_vs_{sp2}',
                            'Fold_Change_Log2': m_val,
                            'P_Value': p_values[i] if i < len(p_values) else 1.0,
                            'P_Value_Corrected': p_corr,
                            'Statistically_Significant': p_corr <= 0.05,
                            'Category': category
                        })
                    
                    # Create scatter plot with colors
                    ax.scatter(A, M, c=colors, alpha=0.6, s=1)
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
                    ax.axhline(y=log2_fc_threshold, color='red', linestyle=':', alpha=0.5)
                    ax.axhline(y=-log2_fc_threshold, color='blue', linestyle=':', alpha=0.5)
                    ax.set_xlabel('A = (log2(Intensity1) + log2(Intensity2))/2')
                    ax.set_ylabel(f'M = log2({sp1}) - log2({sp2})')
                    ax.set_title(f'{sp1} vs {sp2}')
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    mean_M = np.mean(M)
                    std_M = np.std(M)
                    
                    # Count differentially expressed proteins
                    significant_overexpressed = sum(1 for m_val, p_corr in zip(M, p_corrected) 
                                                  if m_val > abs(log2_fc_threshold) and p_corr <= 0.05)
                    significant_downregulated = sum(1 for m_val, p_corr in zip(M, p_corrected) 
                                                  if m_val < -abs(log2_fc_threshold) and p_corr <= 0.05)
                    significant_small_fc = sum(1 for m_val, p_corr in zip(M, p_corrected) 
                                             if abs(m_val) < abs(log2_fc_threshold) and p_corr <= 0.05)
                    not_significant = sum(1 for p_corr in p_corrected if p_corr > 0.05)
                    
                    ax.text(0.05, 0.95, 
                           f'Mean M: {mean_M:.3f}\nStd M: {std_M:.3f}\n'
                           f'Sig. Overexpressed: {significant_overexpressed}\n'
                           f'Sig. Downregulated: {significant_downregulated}\n'
                           f'Sig. Small FC: {significant_small_fc}\n'
                           f'Not significant: {not_significant}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add legend for color coding
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey', markersize=8, label='Not significant'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=8, label='Significant (small FC)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Overexpressed'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Downregulated')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, 
                  title='Fold Change Threshold: 0.05 | Statistical Significance: Benjamini-Hochberg (α=0.05)')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Export protein differential expression data
        if export_data:
            export_df = pd.DataFrame(export_data)
            export_file = output_file.replace('.png', '_differential_expression.csv')
            export_df.to_csv(export_file, index=False)
            print(f"✅ Protein differential expression data with statistical testing saved to: {export_file}")
        
        print(f"✅ MA plots with statistical testing saved to: {output_file}")
    
    def perform_statistical_tests(self, output_file="analysis/statistical_tests.txt"):
        """Perform statistical tests to compare log2-transformed median intensity distributions between species."""
        print("Performing statistical tests using log2-transformed species-specific medians...")
        
        # Prepare data for statistical tests using log2-transformed medians (non-zero only)
        species_data = {}
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            log_col = f'Log2_Median_{species}'
            if log_col in self.df.columns:
                # Only use non-zero log2 medians
                non_zero_mask = self.df[f'Median_{species}'].notna() & (self.df[f'Median_{species}'] > 0)
                species_log_medians = self.df[log_col][non_zero_mask]
                species_data[species] = species_log_medians
        
        # Perform statistical tests
        results = []
        
        # Kruskal-Wallis test for all species
        if len(species_data) >= 3:
            all_data = [species_data[sp] for sp in species_data.keys()]
            h_stat, p_value = kruskal(*all_data)
            results.append(f"Kruskal-Wallis Test (All Species):")
            results.append(f"  H-statistic: {h_stat:.4f}")
            results.append(f"  p-value: {p_value:.4e}")
            results.append(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            results.append("")
        
        # Mann-Whitney U tests for species pairs
        species_pairs = [('Lb', 'Lg'), ('Lb', 'Ln'), ('Lb', 'Lp'), 
                        ('Lg', 'Ln'), ('Lg', 'Lp'), ('Ln', 'Lp')]
        
        results.append("Mann-Whitney U Tests (Species Pairs):")
        results.append("=" * 50)
        
        for sp1, sp2 in species_pairs:
            if sp1 in species_data and sp2 in species_data:
                data1 = species_data[sp1]
                data2 = species_data[sp2]
                
                if len(data1) > 0 and len(data2) > 0:
                    stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                    results.append(f"{sp1} vs {sp2}:")
                    results.append(f"  U-statistic: {stat:.4f}")
                    results.append(f"  p-value: {p_value:.4e}")
                    results.append(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
                    results.append(f"  Effect size (r): {abs(stat - len(data1)*len(data2)/2) / np.sqrt(len(data1)*len(data2)*(len(data1)+len(data2)+1)/12):.3f}")
                    results.append("")
        
        # Summary statistics
        results.append("Summary Statistics (Log2 Median Intensities):")
        results.append("=" * 50)
        
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            if species in species_data:
                data = species_data[species]
                results.append(f"{species}:")
                results.append(f"  Count: {len(data)}")
                results.append(f"  Mean Log2 Intensity: {np.mean(data):.3f}")
                results.append(f"  Median Log2 Intensity: {np.median(data):.3f}")
                results.append(f"  Std Log2 Intensity: {np.std(data):.3f}")
                results.append(f"  Min Log2 Intensity: {np.min(data):.3f}")
                results.append(f"  Max Log2 Intensity: {np.max(data):.3f}")
                results.append("")
        
        # Write results to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(results))
        
        print(f"✅ Statistical test results saved to: {output_file}")
        return results
    

    
    def generate_protein_gene_counts(self, output_file="analysis/protein_gene_counts_summary.csv"):
        """Generate protein group and unique gene counts using the data processor."""
        print("🔬 Generating protein group and unique gene counts using data processor...")
        
        # Check if we have the required data
        if self.processor is None:
            print("❌ No data loaded. Please run load_and_process_data() first.")
            return None
        
        # Use the data processor to get counts
        protein_counts = self.processor.get_protein_counts_per_sample()
        gene_counts = self.processor.get_gene_counts_per_sample()
        species_totals = self.processor.get_species_totals()
        
        # Combine results
        combined_results = protein_counts.merge(gene_counts, on=['Sample_Column', 'Sample_Name', 'Species'])
        
        # Add species totals
        for species, totals in species_totals.items():
            mask = combined_results['Species'] == species
            combined_results.loc[mask, 'Protein_Groups_Total'] = totals['Protein_Groups_Total']
            combined_results.loc[mask, 'Unique_Genes_Total'] = totals['Unique_Genes_Total']
        
        # Add overall totals
        overall_summary = {
            'Sample_Column': 'Overall_Summary',
            'Sample_Name': 'Overall_Summary',
            'Species': 'All',
            'Protein_Groups_Detected': f"{combined_results['Protein_Groups_Detected'].mean():.1f} ± {combined_results['Protein_Groups_Detected'].std():.1f}",
            'Unique_Genes_Detected': f"{combined_results['Unique_Genes_Detected'].mean():.1f} ± {combined_results['Unique_Genes_Detected'].std():.1f}",
            'Protein_Groups_Total': len(self.protein_df),
            'Unique_Genes_Total': len(self.gene_df)
        }
        
        final_df = pd.concat([combined_results, pd.DataFrame([overall_summary])], ignore_index=True)
        
        # Save results
        final_df.to_csv(output_file, index=False)
        
        # Print summary
        print("\n📊 PROTEIN AND GENE COUNT SUMMARY")
        print("=" * 50)
        print("Sample-level counts:")
        for _, row in combined_results.iterrows():
            print(f"  {row['Sample_Name']} ({row['Species']}):")
            print(f"    - Protein groups detected: {row['Protein_Groups_Detected']}")
            print(f"    - Unique genes detected: {row['Unique_Genes_Detected']}")
        
        print("\nSpecies-level collapsed totals:")
        for species, totals in species_totals.items():
            print(f"  {species}:")
            print(f"    - Collapsed protein groups (all samples): {totals['Protein_Groups_Total']}")
            print(f"    - Collapsed unique genes (all samples): {totals['Unique_Genes_Total']}")
        
        print(f"\nOverall:")
        print(f"  - Total protein groups: {overall_summary['Protein_Groups_Total']}")
        print(f"  - Total unique genes: {overall_summary['Unique_Genes_Total']}")
        
        print(f"\n✅ Protein and gene counts saved to: {output_file}")
        return final_df
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("🚀 Starting comprehensive protein intensity analysis...")
        print("=" * 60)
        
        # Create protein-based directory
        import os
        output_dir = "protein-based"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and process data using common data processor
        self.load_and_process_data()
        
        # Generate protein and gene counts using data processor
        self.generate_protein_gene_counts(os.path.join(output_dir, "protein_gene_counts_summary.csv"))
        
        # Create all visualizations
        self.create_comprehensive_intensity_plots(os.path.join(output_dir, "intensity_distributions.png"))
        self.create_ma_plots(os.path.join(output_dir, "proteins_ma_plots.png"))
        
        # Perform statistical tests
        self.perform_statistical_tests(os.path.join(output_dir, "statistical_tests.txt"))
        
        # Save processed data
        self.protein_df.to_csv(os.path.join(output_dir, "processed_intensity_data.csv"), index=False)
        
        print("=" * 60)
        print("✅ Complete analysis finished!")
        print(f"📁 Files generated in '{output_dir}' folder:")
        print("  - protein_gene_counts_summary.csv")
        print("  - intensity_distributions.png")
        print("  - intensity_distributions_statistics.csv")
        print("  - proteins_ma_plots.png")
        print("  - proteins_ma_plots_differential_expression.csv")
        print("  - statistical_tests.txt")
        print("  - processed_intensity_data.csv")

def main():
    """Main function to run the analysis."""
    analyzer = ProteinIntensityAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
