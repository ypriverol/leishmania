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
import warnings
import sys
import os

# Add parent directory to path to import data_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processor import DataProcessor

warnings.filterwarnings('ignore')

class ProteinIntensityAnalyzer:
    def __init__(self, data_file="raw_data.txt.gz"):
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
    
    def create_violin_plots(self, output_file="analysis/intensity_violin_plots.png"):
        """Create violin plots showing intensity distribution per species using median intensities."""
        print("Creating violin plots using species-specific medians...")
        
        # Prepare data for violin plot using species-specific medians (non-zero only)
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
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Protein Intensity Distribution by Species', fontsize=16, fontweight='bold')
        
        # Raw intensity violin plot
        sns.violinplot(data=plot_df, x='Species', y='Median_Intensity', ax=axes[0,0])
        axes[0,0].set_title('Median Intensity Distribution per Species')
        axes[0,0].set_ylabel('Median Intensity')
        axes[0,0].set_yscale('log')
        
        # Log2 intensity violin plot
        sns.violinplot(data=plot_df, x='Species', y='Log2_Median_Intensity', ax=axes[0,1])
        axes[0,1].set_title('Log2 Median Intensity Distribution')
        axes[0,1].set_ylabel('Log2(Median Intensity + 1)')
        
        # Box plot for comparison
        sns.boxplot(data=plot_df, x='Species', y='Log2_Median_Intensity', ax=axes[1,0])
        axes[1,0].set_title('Log2 Median Intensity Box Plot')
        axes[1,0].set_ylabel('Log2(Median Intensity + 1)')
        
        # Statistical summary
        stats_data = plot_df.groupby('Species')['Log2_Median_Intensity'].agg(['count', 'mean', 'std', 'median']).round(3)
        axes[1,1].axis('off')
        axes[1,1].text(0.1, 0.9, 'Statistical Summary (Log2 Intensities):', fontsize=12, fontweight='bold')
        axes[1,1].text(0.1, 0.8, stats_data.to_string(), fontsize=10, fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Violin plots saved to: {output_file}")
        return plot_df
    
    def create_histograms(self, output_file="analysis/intensity_histograms.png"):
        """Create histograms with optimal binning for median intensity distributions per species (non-zero only)."""
        print("Creating histograms using species-specific medians (non-zero only)...")
        
        # Prepare data using species-specific medians (non-zero only)
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
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Protein Intensity Histograms by Species', fontsize=16, fontweight='bold')
        
        # Raw intensity histograms
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            species_data = plot_df[plot_df['Species'] == species]['Median_Intensity']
            if len(species_data) > 0:
                # Use optimal binning (Sturges' rule)
                n_bins = int(np.ceil(1 + 3.322 * np.log10(len(species_data))))
                axes[0,0].hist(species_data, bins=n_bins, alpha=0.6, label=species, density=True)
        
        axes[0,0].set_title('Median Intensity Distribution per Species (Density)')
        axes[0,0].set_xlabel('Median Intensity')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_xscale('log')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Log2 intensity histograms
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            species_data = plot_df[plot_df['Species'] == species]['Log2_Median_Intensity']
            if len(species_data) > 0:
                n_bins = int(np.ceil(1 + 3.322 * np.log10(len(species_data))))
                axes[0,1].hist(species_data, bins=n_bins, alpha=0.6, label=species, density=True)
        
        axes[0,1].set_title('Log2 Median Intensity Distribution (Density)')
        axes[0,1].set_xlabel('Log2(Median Intensity + 1)')
        axes[0,1].set_ylabel('Density')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Cumulative distribution
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            species_data = plot_df[plot_df['Species'] == species]['Log2_Median_Intensity']
            if len(species_data) > 0:
                sorted_data = np.sort(species_data)
                y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                axes[1,0].plot(sorted_data, y, label=species, linewidth=2)
        
        axes[1,0].set_title('Cumulative Distribution (Log2 Median Intensities)')
        axes[1,0].set_xlabel('Log2(Median Intensity + 1)')
        axes[1,0].set_ylabel('Cumulative Probability')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Q-Q plot for normality check
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            species_data = plot_df[plot_df['Species'] == species]['Log2_Median_Intensity']
            if len(species_data) > 0:
                stats.probplot(species_data, dist="norm", plot=axes[1,1])
                axes[1,1].set_title(f'Q-Q Plot for {species} (Log2 Median Intensities)')
                break  # Only show one species for Q-Q plot
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Histograms saved to: {output_file}")
        return plot_df
    
    def create_ma_plots(self, output_file="analysis/ma_plots.png"):
        """Create MA plots for differential analysis between species pairs using pre-calculated medians."""
        print("Creating MA plots using species-specific medians...")
        
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
        fig.suptitle('MA Plots for Differential Analysis Between Species', fontsize=16, fontweight='bold')
        
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
                    
                    # Define fold change threshold (0.05 = log2 fold change of ~-4.32)
                    fc_threshold = 0.05
                    log2_fc_threshold = np.log2(fc_threshold)
                    
                    # Color coding based on fold change
                    colors = []
                    for m_val in M:
                        if abs(m_val) < abs(log2_fc_threshold):
                            colors.append('grey')  # Not overexpressed
                        elif m_val > abs(log2_fc_threshold):
                            colors.append('red')   # Overexpressed in sp1
                        else:
                            colors.append('blue')  # Downregulated in sp1 (overexpressed in sp2)
                    
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
                    overexpressed = sum(1 for m_val in M if m_val > abs(log2_fc_threshold))
                    downregulated = sum(1 for m_val in M if m_val < -abs(log2_fc_threshold))
                    not_changed = len(M) - overexpressed - downregulated
                    
                    ax.text(0.05, 0.95, 
                           f'Mean M: {mean_M:.3f}\nStd M: {std_M:.3f}\n'
                           f'Overexpressed: {overexpressed}\n'
                           f'Downregulated: {downregulated}\n'
                           f'Not changed: {not_changed}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add legend for color coding
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=8, label='Not changed'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Overexpressed'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Downregulated')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, title='Fold Change Threshold: 0.05')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ MA plots saved to: {output_file}")
    
    def perform_statistical_tests(self, output_file="analysis/statistical_tests.txt"):
        """Perform statistical tests to compare median intensity distributions between species."""
        print("Performing statistical tests using species-specific medians...")
        
        # Prepare data for statistical tests using pre-calculated medians (non-zero only)
        species_data = {}
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            median_col = f'Median_{species}'
            if median_col in self.df.columns:
                # Only use non-zero medians
                non_zero_mask = self.df[median_col].notna() & (self.df[median_col] > 0)
                species_medians = self.df[median_col][non_zero_mask]
                species_data[species] = species_medians
        
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
                results.append(f"  Mean: {np.mean(data):.3f}")
                results.append(f"  Median: {np.median(data):.3f}")
                results.append(f"  Std: {np.std(data):.3f}")
                results.append(f"  Min: {np.min(data):.3f}")
                results.append(f"  Max: {np.max(data):.3f}")
                results.append("")
        
        # Write results to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(results))
        
        print(f"✅ Statistical test results saved to: {output_file}")
        return results
    
    def create_comprehensive_report(self, output_file="analysis/comprehensive_analysis_report.png"):
        """Create a comprehensive analysis report with multiple visualizations using species-specific medians."""
        print("Creating comprehensive analysis report using species-specific medians...")
        
        # Prepare data using species-specific medians (non-zero only)
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
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Comprehensive Protein Intensity Analysis Report', fontsize=20, fontweight='bold')
        
        # 1. Violin plot (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        sns.violinplot(data=plot_df, x='Species', y='Log2_Median_Intensity', ax=ax1)
        ax1.set_title('Median Intensity Distribution by Species', fontweight='bold')
        ax1.set_ylabel('Log2(Median Intensity + 1)')
        
        # 2. Box plot (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        sns.boxplot(data=plot_df, x='Species', y='Log2_Median_Intensity', ax=ax2)
        ax2.set_title('Median Intensity Distribution (Box Plot)', fontweight='bold')
        ax2.set_ylabel('Log2(Median Intensity + 1)')
        
        # 3. Histogram (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            species_data = plot_df[plot_df['Species'] == species]['Log2_Median_Intensity']
            if len(species_data) > 0:
                ax3.hist(species_data, bins=30, alpha=0.6, label=species, density=True)
        ax3.set_title('Median Intensity Distribution (Histogram)', fontweight='bold')
        ax3.set_xlabel('Log2(Median Intensity + 1)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative distribution (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            species_data = plot_df[plot_df['Species'] == species]['Log2_Median_Intensity']
            if len(species_data) > 0:
                sorted_data = np.sort(species_data)
                y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax4.plot(sorted_data, y, label=species, linewidth=2)
        ax4.set_title('Cumulative Distribution (Median Intensities)', fontweight='bold')
        ax4.set_xlabel('Log2(Median Intensity + 1)')
        ax4.set_ylabel('Cumulative Probability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. MA plots (bottom)
        species_pairs = [('Lb', 'Lg'), ('Lb', 'Ln'), ('Lb', 'Lp'), ('Lg', 'Ln')]
        for idx, (sp1, sp2) in enumerate(species_pairs):
            ax = fig.add_subplot(gs[2:, idx])
            
            # Use pre-calculated species medians (non-zero only)
            sp1_median_col = f'Median_{sp1}'
            sp2_median_col = f'Median_{sp2}'
            
            if sp1_median_col in self.df.columns and sp2_median_col in self.df.columns:
                # Get non-zero medians for both species
                non_zero_mask1 = self.df[sp1_median_col].notna() & (self.df[sp1_median_col] > 0)
                non_zero_mask2 = self.df[sp2_median_col].notna() & (self.df[sp2_median_col] > 0)
                
                intensity1 = self.df[sp1_median_col][non_zero_mask1]
                intensity2 = self.df[sp2_median_col][non_zero_mask2]
                
                # Find common proteins (proteins present in both species)
                common_proteins = intensity1.index.intersection(intensity2.index)
                intensity1_common = intensity1[common_proteins]
                intensity2_common = intensity2[common_proteins]
                
                if len(intensity1_common) > 0:
                    M = np.log2(intensity1_common + 1) - np.log2(intensity2_common + 1)
                    A = (np.log2(intensity1_common + 1) + np.log2(intensity2_common + 1)) / 2
                    
                    # Define fold change threshold (0.05 = log2 fold change of ~-4.32)
                    fc_threshold = 0.05
                    log2_fc_threshold = np.log2(fc_threshold)
                    
                    # Color coding based on fold change
                    colors = []
                    for m_val in M:
                        if abs(m_val) < abs(log2_fc_threshold):
                            colors.append('grey')  # Not overexpressed
                        elif m_val > abs(log2_fc_threshold):
                            colors.append('red')   # Overexpressed in sp1
                        else:
                            colors.append('blue')  # Downregulated in sp1 (overexpressed in sp2)
                    
                    ax.scatter(A, M, c=colors, alpha=0.6, s=1)
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
                    ax.axhline(y=log2_fc_threshold, color='red', linestyle=':', alpha=0.5)
                    ax.axhline(y=-log2_fc_threshold, color='blue', linestyle=':', alpha=0.5)
                    ax.set_xlabel('A')
                    ax.set_ylabel(f'M ({sp1}-{sp2})')
                    ax.set_title(f'MA Plot: {sp1} vs {sp2}')
                    ax.grid(True, alpha=0.3)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive report saved to: {output_file}")
    
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
        
        # Ensure analysis directory exists
        import os
        os.makedirs("analysis", exist_ok=True)
        
        # Load and process data using common data processor
        self.load_and_process_data()
        
        # Generate protein and gene counts using data processor
        self.generate_protein_gene_counts("analysis/protein_gene_counts_summary.csv")
        
        # Create all visualizations
        self.create_violin_plots("analysis/intensity_violin_plots.png")
        self.create_histograms("analysis/intensity_histograms.png")
        self.create_ma_plots("analysis/ma_plots.png")
        self.create_comprehensive_report("analysis/comprehensive_analysis_report.png")
        
        # Perform statistical tests
        self.perform_statistical_tests("analysis/statistical_tests.txt")
        
        # Save processed data
        self.protein_df.to_csv("analysis/processed_intensity_data.csv", index=False)
        
        print("=" * 60)
        print("✅ Complete analysis finished!")
        print("📁 Files generated in 'analysis' folder:")
        print("  - protein_gene_counts_summary.csv")
        print("  - intensity_violin_plots.png")
        print("  - intensity_histograms.png") 
        print("  - ma_plots.png")
        print("  - comprehensive_analysis_report.png")
        print("  - statistical_tests.txt")
        print("  - processed_intensity_data.csv")

def main():
    """Main function to run the analysis."""
    analyzer = ProteinIntensityAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
