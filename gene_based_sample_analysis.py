#!/usr/bin/env python3
"""
Gene-Based Sample-Level Analysis
===============================

This script:
1. Uses gene-level data (proteins mapped to unique genes)
2. Keeps the original sample-level analysis (50 samples)
3. Maintains the same clustering and visualization approach

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
from scipy.spatial.distance import pdist, squareform
import os

class GeneSampleAnalyzer:
    def __init__(self, raw_data_path='raw_data.txt'):
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
    
    def load_and_process_to_genes(self):
        """Load data and filter to unique gene mappings."""
        print("🔍 Loading data and mapping to unique genes...")
        
        # Load raw data
        self.df = pd.read_csv(self.raw_data_path, sep='\t', low_memory=False)
        print(f"Total proteins loaded: {len(self.df)}")
        
        # Filter out decoys and contaminants
        if "Reverse" in self.df.columns and "Potential contaminant" in self.df.columns:
            self.df = self.df[(self.df["Reverse"] != '+') & (self.df["Potential contaminant"] != '+')]
            print(f"Proteins after filtering decoys/contaminants: {len(self.df)}")
        
        # Calculate total intensity and filter out undetected proteins
        intensity_cols = [col for col in self.df.columns if col.startswith("Intensity ")]
        self.df['total_intensity'] = self.df[intensity_cols].sum(axis=1)
        self.df = self.df[self.df['total_intensity'] > 0]
        print(f"Proteins with detected intensity: {len(self.df)}")
        
        # Filter to proteins/protein groups that map to unique genes
        self._filter_to_unique_genes()
        
        print(f"Final dataset for analysis: {len(self.df)} entries mapping to unique genes")
        
        return self.df
    
    def _filter_to_unique_genes(self):
        """Filter to only proteins/protein groups that map to exactly one gene."""
        print("🧬 Filtering to proteins/protein groups with unique gene mapping...")
        
        valid_indices = []
        
        for idx, row in self.df.iterrows():
            fasta_header = row["Fasta headers"]
            
            # Extract genes from fasta header
            genes = self.extract_gene_from_fasta_header(fasta_header)
            unique_genes = list(set(genes))
            
            # Only keep if maps to exactly one gene
            if len(unique_genes) == 1:
                valid_indices.append(idx)
        
        # Filter dataframe to only valid gene mappings
        self.df = self.df.loc[valid_indices]
        
        # Add gene information
        gene_ids = []
        for idx, row in self.df.iterrows():
            fasta_header = row["Fasta headers"]
            genes = self.extract_gene_from_fasta_header(fasta_header)
            unique_genes = list(set(genes))
            gene_ids.append(unique_genes[0] if unique_genes else "Unknown")
        
        self.df['Gene_ID'] = gene_ids
        
        print(f"Proteins/protein groups mapping to unique genes: {len(self.df)}")
    
    def create_sample_matrix(self):
        """Create sample-level matrix (same as original analysis)."""
        print("📊 Creating sample-level intensity matrix...")
        
        # Get all intensity columns (samples)
        intensity_cols = [col for col in self.df.columns if col.startswith("Intensity ")]
        
        # Create matrix: genes x samples
        sample_matrix = self.df[intensity_cols].copy()
        sample_matrix.index = self.df['Gene_ID']
        
        # Apply log transformation
        sample_matrix = np.log1p(sample_matrix)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='constant', fill_value=0.0)
        sample_matrix_imputed = pd.DataFrame(
            imputer.fit_transform(sample_matrix),
            index=sample_matrix.index,
            columns=sample_matrix.columns
        )
        
        # Transpose to get samples as rows (for clustering)
        sample_matrix_final = sample_matrix_imputed.T
        
        print(f"Sample matrix shape: {sample_matrix_final.shape} (samples x genes)")
        return sample_matrix_final
    
    def calculate_gene_uniqueness_by_samples(self, sample_matrix):
        """Calculate gene uniqueness using sample-level data."""
        print("🔍 Calculating gene uniqueness across species...")
        
        # Get sample labels and map to species
        sample_labels = list(sample_matrix.index)
        species_cols = {sp: [label for label in sample_labels if label.startswith(pref)] 
                       for sp, pref in self.species_prefixes.items()}
        species_cols = {sp: cols for sp, cols in species_cols.items() if len(cols) > 0}
        
        # Transpose back to get genes as rows for uniqueness calculation
        gene_matrix = sample_matrix.T
        
        uniqueness_data = {}
        
        for species, sample_cols in species_cols.items():
            # Get genes detected in this species (any sample)
            species_data = gene_matrix[sample_cols]
            species_genes = gene_matrix.index[species_data.sum(axis=1) > 0].tolist()
            
            # Calculate uniqueness
            unique_genes = []
            shared_genes = []
            
            for gene in species_genes:
                # Check if gene is detected in other species
                other_species_detected = []
                for other_sp, other_cols in species_cols.items():
                    if other_sp != species:
                        other_data = gene_matrix.loc[gene, other_cols]
                        sum_value = np.sum(other_data.values)
                        other_species_detected.append(sum_value > 0)
                
                # If gene is only detected in this species, it's unique
                if not any(other_species_detected):
                    unique_genes.append(gene)
                else:
                    shared_genes.append(gene)
            
            uniqueness_data[species] = {
                'total_genes': len(species_genes),
                'unique_genes': len(unique_genes),
                'shared_genes': len(shared_genes),
                'unique_gene_list': unique_genes,
                'shared_gene_list': shared_genes,
                'sample_count': len(sample_cols)
            }
        
        return uniqueness_data
    
    def braycurtis_upgma(self, data, labels, tag=""):
        """Calculate Bray-Curtis distance and UPGMA clustering."""
        print(f"🌳 Calculating Bray-Curtis distance and UPGMA clustering for {len(labels)} {tag}...")
        
        # Calculate Bray-Curtis distance
        from scipy.spatial.distance import braycurtis
        D = pdist(data, metric=braycurtis)
        D_matrix = squareform(D)
        
        # UPGMA clustering
        Z = linkage(D, method='average')
        
        return Z, D_matrix
    
    def spearman_upgma(self, data, labels, tag=""):
        """Calculate Spearman distance and UPGMA clustering."""
        print(f"🌳 Calculating Spearman distance and UPGMA clustering for {len(labels)} {tag}...")
        
        # Calculate Spearman correlation distance
        from scipy.spatial.distance import correlation
        D = pdist(data, metric=correlation)
        D_matrix = squareform(D)
        
        # UPGMA clustering
        Z = linkage(D, method='average')
        
        return Z, D_matrix
    
    def plot_radial_phylogenetic_tree(self, Z, labels, uniqueness_data, output_path):
        """Create radial phylogenetic tree with gene uniqueness data."""
        print("🎨 Creating gene-based radial phylogenetic tree...")
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})
        
        # Map sample labels to species
        species_angles = {}
        species_colors = {
            'Lb': '#1f77b4',  # Blue
            'Lg': '#ff7f0e',  # Orange  
            'Ln': '#2ca02c',  # Green
            'Lp': '#d62728',  # Red
        }
        
        # Calculate angles for each species
        species_list = list(self.species_prefixes.keys())
        angles = np.linspace(0, 2*np.pi, len(species_list), endpoint=False)
        
        for i, species in enumerate(species_list):
            species_angles[species] = angles[i]
        
        # Create circles for gene uniqueness
        outer_radius = 1.2
        inner_radius = 0.8
        inner_inner_radius = 0.4
        
        # Plot outer circle (genes shared with other species)
        for species, angle in species_angles.items():
            if species in uniqueness_data:
                shared_count = uniqueness_data[species]['shared_genes']
                total_count = uniqueness_data[species]['total_genes']
                
                if total_count > 0:
                    # Outer circle - shared genes
                    shared_angle = 2 * np.pi * shared_count / total_count if total_count > 0 else 0
                    ax.bar(angle, outer_radius - inner_radius, width=shared_angle, 
                          bottom=inner_radius, color=species_colors.get(species, 'gray'), 
                          alpha=0.8, edgecolor='black', linewidth=1)
                    
                    # Add text for shared genes
                    if shared_angle > 0:
                        text_angle = angle + shared_angle/2
                        ax.text(text_angle, inner_radius + (outer_radius - inner_radius)/2, 
                               str(shared_count), ha='center', va='center', 
                               fontsize=10, fontweight='bold', color='black')
        
        # Plot inner circles (unique genes and species-only shared)
        for species, angle in species_angles.items():
            if species in uniqueness_data:
                unique_count = uniqueness_data[species]['unique_genes']
                shared_count = uniqueness_data[species]['shared_genes']
                total_count = uniqueness_data[species]['total_genes']
                
                if total_count > 0:
                    # Inner circle - species-only shared (main pie)
                    species_only_angle = 2 * np.pi * shared_count / total_count if total_count > 0 else 0
                    ax.bar(angle, inner_radius - inner_inner_radius, width=species_only_angle, 
                          bottom=inner_inner_radius, color=species_colors.get(species, 'gray'), 
                          alpha=0.6, edgecolor='black', linewidth=1)
                    
                    # Add text for species-only shared
                    if species_only_angle > 0:
                        text_angle = angle + species_only_angle/2
                        ax.text(text_angle, inner_inner_radius + (inner_radius - inner_inner_radius)/2, 
                               str(shared_count), ha='center', va='center', 
                               fontsize=10, fontweight='bold', color='black')
                    
                    # Inner circle - unique genes (offset pie)
                    unique_angle = 2 * np.pi * unique_count / total_count if total_count > 0 else 0
                    if unique_angle > 0:
                        # Use lighter color for unique genes
                        unique_color = plt.cm.colors.to_rgba(species_colors.get(species, 'gray'), alpha=0.3)
                        ax.bar(angle + species_only_angle, inner_radius - inner_inner_radius, 
                              width=unique_angle, bottom=inner_inner_radius, 
                              color=unique_color, edgecolor='black', linewidth=1)
                        
                        # Add text for unique genes
                        text_angle = angle + species_only_angle + unique_angle/2
                        ax.text(text_angle, inner_inner_radius + (inner_radius - inner_inner_radius)/2, 
                               str(unique_count), ha='center', va='center', 
                               fontsize=10, fontweight='bold', color='black')
        
        # Customize the plot
        ax.set_title('Gene-Based Radial Phylogenetic Tree\n(Using Sample-Level Analysis)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, outer_radius + 0.1)
        ax.set_xticks(angles)
        ax.set_xticklabels(species_list, fontsize=12, fontweight='bold')
        
        # Add legend
        legend_elements = []
        for species, color in species_colors.items():
            if species in uniqueness_data:
                sample_count = uniqueness_data[species]['sample_count']
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, 
                                                   label=f'{species} ({sample_count} samples)'))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gene-based radial phylogenetic tree saved to: {output_path}")
    
    def plot_traditional_dendrogram(self, Z, labels, output_path):
        """Create traditional dendrogram."""
        print("🌳 Creating traditional dendrogram...")
        
        # Clean labels (remove intensity prefix and add species info)
        cleaned_labels = []
        for label in labels:
            if label.startswith('Intensity '):
                # Extract species and sample number
                parts = label.split('_')
                if len(parts) >= 2:
                    species = parts[0].replace('Intensity ', '')
                    sample_id = parts[1]
                    cleaned_labels.append(f"{sample_id} ({species})")
                else:
                    cleaned_labels.append(label.replace('Intensity ', ''))
            else:
                cleaned_labels.append(label)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create dendrogram
        dendrogram(Z, labels=cleaned_labels, ax=ax, orientation='top', 
                  leaf_rotation=90, leaf_font_size=8, show_leaf_counts=True)
        
        ax.set_title('Gene-Based Traditional Phylogenetic Tree (Sample Level)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Samples', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Traditional dendrogram saved to: {output_path}")
    
    def run_gene_sample_analysis(self):
        """Run complete gene-based sample-level analysis."""
        print("🚀 Starting gene-based sample-level analysis...")
        print("=" * 60)
        
        # Load and process data to gene level
        self.load_and_process_to_genes()
        
        # Create sample matrix (genes x samples, then transpose to samples x genes)
        sample_matrix = self.create_sample_matrix()
        
        # Calculate gene uniqueness
        uniqueness_data = self.calculate_gene_uniqueness_by_samples(sample_matrix)
        
        # Create output directory
        output_dir = "gene_sample_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Build phylogenetic trees (sample-level clustering)
        print("🌳 Building sample-level phylogenetic trees...")
        
        # Bray-Curtis distance
        Z_bray, D_bray = self.braycurtis_upgma(sample_matrix.values, sample_matrix.index, tag="samples")
        
        # Spearman distance
        Z_spear, D_spear = self.spearman_upgma(sample_matrix.values, sample_matrix.index, tag="samples")
        
        # Generate plots
        print("🎨 Generating phylogenetic plots...")
        
        # Radial phylogenetic tree
        self.plot_radial_phylogenetic_tree(Z_bray, sample_matrix.index, uniqueness_data, 
                                         os.path.join(output_dir, "gene_sample_radial_tree.png"))
        
        # Traditional dendrogram
        self.plot_traditional_dendrogram(Z_bray, sample_matrix.index, 
                                       os.path.join(output_dir, "gene_sample_traditional_dendrogram.png"))
        
        # Save data
        self.df.to_csv(os.path.join(output_dir, "gene_filtered_data.csv"), index=False)
        sample_matrix.to_csv(os.path.join(output_dir, "gene_sample_matrix.csv"))
        
        # Print summary
        print("\n📊 GENE-BASED SAMPLE-LEVEL ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total entries analyzed: {len(self.df)} (mapped to unique genes)")
        print(f"Sample matrix: {sample_matrix.shape[0]} samples x {sample_matrix.shape[1]} genes")
        print(f"Clustering performed on: {sample_matrix.shape[0]} samples (same as original)")
        
        for species in self.species_prefixes.keys():
            if species in uniqueness_data:
                data = uniqueness_data[species]
                print(f"\n{species} ({data['sample_count']} samples):")
                print(f"  Total genes: {data['total_genes']}")
                print(f"  Unique genes: {data['unique_genes']}")
                print(f"  Shared genes: {data['shared_genes']}")
        
        print(f"\n✅ Gene-based sample-level analysis complete!")
        print(f"📁 Results saved to: {output_dir}/")
        
        return self.df, uniqueness_data

if __name__ == "__main__":
    analyzer = GeneSampleAnalyzer()
    gene_df, uniqueness_data = analyzer.run_gene_sample_analysis()
