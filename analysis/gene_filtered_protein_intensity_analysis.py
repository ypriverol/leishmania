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
from statsmodels.stats.multitest import multipletests
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
    
    def create_comprehensive_intensity_plots(self, output_file="gene_filtered_intensity_distributions.png"):
        """Create comprehensive intensity distribution plots with correct Leishmania color scheme for gene-filtered data."""
        print("Creating comprehensive intensity distribution plots with Leishmania color scheme (gene-filtered)...")
        
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
        fig.suptitle('Leishmania Gene-Filtered Protein Intensity Distributions (Log2-Transformed)\n(Only proteins mapping to unique genes)', fontsize=16, fontweight='bold')
        
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
    
    def create_ma_plots(self, output_file="gene_ma_plots.png"):
        """Create MA plots for differential analysis between species pairs using pre-calculated medians for gene-filtered data with statistical testing."""
        print("Creating MA plots using species-specific medians with statistical testing (gene-filtered)...")
        
        # Import required statistical functions
        from scipy import stats
        from statsmodels.stats.multitest import multipletests
        
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
        fig.suptitle('MA Plots for Differential Analysis Between Species (Gene-Filtered)\n(Only proteins mapping to unique genes) - Statistical Significance with Benjamini-Hochberg', fontsize=16, fontweight='bold')
        
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
            print(f"✅ Gene-filtered protein differential expression data with statistical testing saved to: {export_file}")
            
            # Export specific data for overexpressed and downregulated proteins
            significant_changes = export_df[export_df['Category'].isin(['Overexpressed', 'Downregulated'])]
            if len(significant_changes) > 0:
                # Add gene accession information
                significant_changes_with_genes = []
                for _, row in significant_changes.iterrows():
                    # Find the protein in the main dataframe to get gene information
                    protein_accession = row['Protein_Group_Accession']
                    # Find the protein by matching Protein_IDs column
                    protein_mask = self.df['Protein_IDs'] == protein_accession
                    if protein_mask.any():
                        gene_accession = self.df.loc[protein_mask, 'Gene_ID'].iloc[0]
                    else:
                        gene_accession = "Unknown"
                    
                    significant_changes_with_genes.append({
                        'Species_Combination': row['Species_Comparison'],
                        'Protein_Group_Accession': protein_accession,
                        'Gene_Accession': gene_accession,
                        'Category': row['Category'],
                        'Fold_Change': round(row['Fold_Change_Log2'], 3),
                        'P_Value_Corrected': round(row['P_Value_Corrected'], 3)
                    })
                
                # Create and export the specific CSV
                specific_export_df = pd.DataFrame(significant_changes_with_genes)
                specific_export_file = output_file.replace('.png', '_overexpressed_downregulated.csv')
                specific_export_df.to_csv(specific_export_file, index=False)
                print(f"✅ Overexpressed and downregulated proteins exported to: {specific_export_file}")
                print(f"   - Total overexpressed proteins: {len(significant_changes[significant_changes['Category'] == 'Overexpressed'])}")
                print(f"   - Total downregulated proteins: {len(significant_changes[significant_changes['Category'] == 'Downregulated'])}")
            else:
                print("⚠️  No overexpressed or downregulated proteins found to export.")
        
        print(f"✅ MA plots with statistical testing saved to: {output_file}")
    
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
    

    
    def run_complete_analysis(self):
        """Run the complete gene-filtered protein intensity analysis."""
        print("🚀 Starting gene-filtered protein intensity analysis...")
        print("=" * 60)
        
        # Load and process data
        self.load_and_process_data()
        
        # Create output directory
        output_dir = "gene-based"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all analyses
        print("\n📊 Generating analyses...")
        
        # Create visualizations
        self.create_comprehensive_intensity_plots(os.path.join(output_dir, "intensity_distributions.png"))
        self.create_ma_plots(os.path.join(output_dir, "gene_ma_plots.png"))
        
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
