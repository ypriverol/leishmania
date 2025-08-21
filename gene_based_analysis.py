#!/usr/bin/env python3
"""
Gene-Based Analysis
==================

This script performs all analysis at the gene level:
1. Aggregates protein intensities by gene
2. Uses gene IDs for consistent analysis
3. Handles protein groups that map to unique genes
4. Provides gene-level statistics and visualizations

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu, kruskal
import os

class GeneBasedAnalyzer:
    def __init__(self, raw_data_path='raw_data.txt.gz'):
        self.raw_data_path = raw_data_path
        self.df = None
        self.gene_df = None
        self.species_prefixes = {
            "Lb": "Intensity Lb",
            "Lg": "Intensity Lg", 
            "Ln": "Intensity Ln",
            "Lp": "Intensity Lp",
        }
        
    def extract_gene_from_fasta_header(self, fasta_header):
        """Extract gene names from Fasta header."""
        genes = []
        if 'GN=' in fasta_header:
            gene_matches = re.findall(r'GN=([^=]+)', fasta_header)
            genes.extend(gene_matches)
        return genes
    
    def load_and_filter_data(self):
        """Load data and filter to gene-level analysis."""
        print("🔍 Loading and filtering data for gene-based analysis...")
        
        # Load raw data (handle gzipped file)
        if self.raw_data_path.endswith('.gz'):
            import gzip
            with gzip.open(self.raw_data_path, 'rt') as f:
                self.df = pd.read_csv(f, sep='\t', low_memory=False)
        else:
            self.df = pd.read_csv(self.raw_data_path, sep='\t', low_memory=False)
        print(f"Total proteins loaded: {len(self.df)}")
        
        # Filter out decoys and contaminants
        if "Reverse" in self.df.columns and "Potential contaminant" in self.df.columns:
            self.df = self.df[(self.df["Reverse"] != '+') & (self.df["Potential contaminant"] != '+')]
            print(f"Proteins after filtering decoys/contaminants: {len(self.df)}")
        
        # Identify species intensity columns
        species_cols = {}
        for sp, pref in self.species_prefixes.items():
            cols = [c for c in self.df.columns if c.startswith(pref)]
            if cols:
                species_cols[sp] = cols
                print(f"Found {len(cols)} intensity columns for {sp}")
        
        # Calculate total intensity for each protein
        all_intensity_cols = sum(species_cols.values(), [])
        self.df['total_intensity'] = self.df[all_intensity_cols].sum(axis=1)
        
        # Filter out undetected proteins
        self.df = self.df[self.df['total_intensity'] > 0]
        print(f"Proteins with detected intensity: {len(self.df)}")
        
        # Process protein groups and extract gene information
        self._process_protein_groups()
        
        return self.gene_df
    
    def _process_protein_groups(self):
        """Process protein groups and aggregate by gene."""
        print("🧬 Processing protein groups and aggregating by gene...")
        
        gene_data = []
        
        for idx, row in self.df.iterrows():
            protein_ids = row["Protein IDs"]
            fasta_header = row["Fasta headers"]
            
            # Extract genes from fasta header
            genes = self.extract_gene_from_fasta_header(fasta_header)
            unique_genes = list(set(genes))
            
            # Only include if maps to exactly one gene
            if len(unique_genes) == 1:
                gene = unique_genes[0]
                
                # Get intensity data
                intensity_data = {}
                for sp, pref in self.species_prefixes.items():
                    sp_cols = [c for c in self.df.columns if c.startswith(pref)]
                    if sp_cols:
                        # Calculate median intensity for this species
                        sp_intensities = row[sp_cols].values
                        sp_intensities = sp_intensities[sp_intensities > 0]  # Remove zeros
                        if len(sp_intensities) > 0:
                            intensity_data[sp] = np.median(sp_intensities)
                        else:
                            intensity_data[sp] = 0
                
                gene_data.append({
                    'Gene_ID': gene,
                    'Protein_Group': protein_ids,
                    'Protein_ID_Count': len(protein_ids.split(';')),
                    'Fasta_Header': fasta_header,
                    **intensity_data
                })
        
        # Create gene dataframe
        self.gene_df = pd.DataFrame(gene_data)
        
        # Aggregate by gene (in case multiple protein groups map to same gene)
        self._aggregate_by_gene()
        
        print(f"Genes after processing: {len(self.gene_df)}")
        
    def _aggregate_by_gene(self):
        """Aggregate multiple protein groups that map to the same gene."""
        print("🔄 Aggregating protein groups by gene...")
        
        # Group by gene and aggregate intensities
        gene_groups = self.gene_df.groupby('Gene_ID')
        
        aggregated_data = []
        for gene_id, group in gene_groups:
            if len(group) == 1:
                # Single protein group for this gene
                aggregated_data.append(group.iloc[0].to_dict())
            else:
                # Multiple protein groups for this gene - aggregate intensities
                print(f"Gene {gene_id} has {len(group)} protein groups - aggregating...")
                
                # Combine protein groups
                combined_proteins = ';'.join(group['Protein_Group'].tolist())
                protein_count = group['Protein_ID_Count'].sum()
                
                # Aggregate intensities (mean across protein groups)
                intensity_data = {}
                for sp in self.species_prefixes.keys():
                    if sp in group.columns:
                        sp_intensities = group[sp].values
                        sp_intensities = sp_intensities[sp_intensities > 0]  # Remove zeros
                        if len(sp_intensities) > 0:
                            intensity_data[sp] = np.mean(sp_intensities)
                        else:
                            intensity_data[sp] = 0
                
                aggregated_data.append({
                    'Gene_ID': gene_id,
                    'Protein_Group': combined_proteins,
                    'Protein_ID_Count': protein_count,
                    'Fasta_Header': group['Fasta_Header'].iloc[0],  # Use first header
                    **intensity_data
                })
        
        self.gene_df = pd.DataFrame(aggregated_data)
        print(f"Final unique genes: {len(self.gene_df)}")
    
    def create_gene_intensity_matrix(self):
        """Create gene intensity matrix for analysis."""
        print("📊 Creating gene intensity matrix...")
        
        # Select intensity columns
        intensity_cols = [sp for sp in self.species_prefixes.keys() if sp in self.gene_df.columns]
        
        # Create matrix
        gene_matrix = self.gene_df[intensity_cols].copy()
        gene_matrix.index = self.gene_df['Gene_ID']
        
        # Apply log transformation
        gene_matrix = np.log2(gene_matrix + 1)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='constant', fill_value=0.0)
        gene_matrix_imputed = pd.DataFrame(
            imputer.fit_transform(gene_matrix),
            index=gene_matrix.index,
            columns=gene_matrix.columns
        )
        
        return gene_matrix_imputed
    
    def calculate_gene_uniqueness(self):
        """Calculate gene uniqueness across species."""
        print("🔍 Calculating gene uniqueness...")
        
        gene_matrix = self.create_gene_intensity_matrix()
        species_cols = list(self.species_prefixes.keys())
        
        uniqueness_data = {}
        
        for species in species_cols:
            if species in gene_matrix.columns:
                # Get genes detected in this species
                species_genes = gene_matrix[gene_matrix[species] > 0].index.tolist()
                
                # Calculate uniqueness
                unique_genes = []
                shared_genes = []
                
                for gene in species_genes:
                    # Check if gene is detected in other species
                    other_species_intensities = []
                    for other_sp in species_cols:
                        if other_sp != species and other_sp in gene_matrix.columns:
                            other_species_intensities.append(gene_matrix.loc[gene, other_sp])
                    
                    # If gene is only detected in this species, it's unique
                    if all(intensity == 0 for intensity in other_species_intensities):
                        unique_genes.append(gene)
                    else:
                        shared_genes.append(gene)
                
                uniqueness_data[species] = {
                    'total_genes': len(species_genes),
                    'unique_genes': len(unique_genes),
                    'shared_genes': len(shared_genes),
                    'unique_gene_list': unique_genes,
                    'shared_gene_list': shared_genes
                }
        
        return uniqueness_data
    
    def create_gene_statistics_report(self):
        """Create comprehensive gene statistics report."""
        print("📈 Creating gene statistics report...")
        
        gene_matrix = self.create_gene_intensity_matrix()
        uniqueness_data = self.calculate_gene_uniqueness()
        
        # Create output directory
        output_dir = "gene_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate statistics
        stats = []
        for species in self.species_prefixes.keys():
            if species in gene_matrix.columns:
                species_intensities = gene_matrix[species].values
                species_intensities = species_intensities[species_intensities > 0]
                
                if len(species_intensities) > 0:
                    stats.append({
                        'Species': species,
                        'Total_Genes': len(species_intensities),
                        'Mean_Intensity': np.mean(species_intensities),
                        'Median_Intensity': np.median(species_intensities),
                        'Std_Intensity': np.std(species_intensities),
                        'Min_Intensity': np.min(species_intensities),
                        'Max_Intensity': np.max(species_intensities),
                        'Unique_Genes': uniqueness_data[species]['unique_genes'],
                        'Shared_Genes': uniqueness_data[species]['shared_genes']
                    })
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(f"{output_dir}/gene_statistics.csv", index=False)
        
        # Create visualizations
        self._create_gene_visualizations(gene_matrix, uniqueness_data, output_dir)
        
        return stats_df
    
    def _create_gene_visualizations(self, gene_matrix, uniqueness_data, output_dir):
        """Create gene-level visualizations."""
        print("🎨 Creating gene visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Gene intensity distribution per species
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, species in enumerate(self.species_prefixes.keys()):
            if species in gene_matrix.columns:
                species_intensities = gene_matrix[species].values
                species_intensities = species_intensities[species_intensities > 0]
                
                if len(species_intensities) > 0:
                    # Violin plot
                    axes[i].violinplot(species_intensities, positions=[1])
                    axes[i].set_title(f'{species} - Gene Intensity Distribution')
                    axes[i].set_ylabel('Log2 Intensity')
                    axes[i].set_xlabel('Species')
                    
                    # Add statistics
                    mean_int = np.mean(species_intensities)
                    median_int = np.median(species_intensities)
                    axes[i].text(0.7, 0.9, f'Mean: {mean_int:.2f}\nMedian: {median_int:.2f}', 
                               transform=axes[i].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gene_intensity_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Gene uniqueness heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create uniqueness matrix
        species_list = list(self.species_prefixes.keys())
        uniqueness_matrix = np.zeros((len(species_list), len(species_list)))
        
        for i, sp1 in enumerate(species_list):
            for j, sp2 in enumerate(species_list):
                if sp1 in uniqueness_data and sp2 in uniqueness_data:
                    # Count genes shared between species
                    sp1_genes = set(uniqueness_data[sp1]['unique_gene_list'] + uniqueness_data[sp1]['shared_gene_list'])
                    sp2_genes = set(uniqueness_data[sp2]['unique_gene_list'] + uniqueness_data[sp2]['shared_gene_list'])
                    shared_count = len(sp1_genes.intersection(sp2_genes))
                    uniqueness_matrix[i, j] = shared_count
        
        sns.heatmap(uniqueness_matrix, annot=True, fmt='.0f', cmap='Blues', 
                   xticklabels=species_list, yticklabels=species_list, ax=ax)
        ax.set_title('Gene Sharing Between Species')
        ax.set_xlabel('Species')
        ax.set_ylabel('Species')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gene_sharing_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Gene uniqueness summary
        fig, ax = plt.subplots(figsize=(10, 6))
        
        species_names = []
        unique_counts = []
        shared_counts = []
        
        for species in self.species_prefixes.keys():
            if species in uniqueness_data:
                species_names.append(species)
                unique_counts.append(uniqueness_data[species]['unique_genes'])
                shared_counts.append(uniqueness_data[species]['shared_genes'])
        
        x = np.arange(len(species_names))
        width = 0.35
        
        ax.bar(x - width/2, unique_counts, width, label='Unique Genes', alpha=0.8)
        ax.bar(x + width/2, shared_counts, width, label='Shared Genes', alpha=0.8)
        
        ax.set_xlabel('Species')
        ax.set_ylabel('Number of Genes')
        ax.set_title('Gene Uniqueness Summary')
        ax.set_xticks(x)
        ax.set_xticklabels(species_names)
        ax.legend()
        
        # Add value labels on bars
        for i, (unique, shared) in enumerate(zip(unique_counts, shared_counts)):
            ax.text(i - width/2, unique + 1, str(unique), ha='center', va='bottom')
            ax.text(i + width/2, shared + 1, str(shared), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gene_uniqueness_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Gene visualizations saved to {output_dir}/")
    
    def save_gene_data(self):
        """Save processed gene data."""
        output_dir = "gene_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save gene dataframe
        self.gene_df.to_csv(f"{output_dir}/gene_level_data.csv", index=False)
        
        # Save gene intensity matrix
        gene_matrix = self.create_gene_intensity_matrix()
        gene_matrix.to_csv(f"{output_dir}/gene_intensity_matrix.csv")
        
        print(f"💾 Gene data saved to {output_dir}/")
    
    def run_complete_analysis(self):
        """Run complete gene-based analysis."""
        print("🚀 Starting complete gene-based analysis...")
        print("=" * 60)
        
        # Load and process data
        self.load_and_filter_data()
        
        # Generate statistics and visualizations
        stats_df = self.create_gene_statistics_report()
        
        # Save processed data
        self.save_gene_data()
        
        # Print summary
        print("\n📊 GENE-BASED ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total unique genes analyzed: {len(self.gene_df)}")
        print(f"Species analyzed: {list(self.species_prefixes.keys())}")
        
        for _, row in stats_df.iterrows():
            print(f"\n{row['Species']}:")
            print(f"  Total genes: {row['Total_Genes']}")
            print(f"  Unique genes: {row['Unique_Genes']}")
            print(f"  Shared genes: {row['Shared_Genes']}")
            print(f"  Mean intensity: {row['Mean_Intensity']:.2f}")
        
        print(f"\n✅ Analysis complete! Results saved to gene_analysis/")
        
        return self.gene_df, stats_df

if __name__ == "__main__":
    analyzer = GeneBasedAnalyzer()
    gene_df, stats_df = analyzer.run_complete_analysis()
