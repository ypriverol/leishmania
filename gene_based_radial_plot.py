#!/usr/bin/env python3
"""
Gene-Based Radial Phylogenetic Plot
==================================

This script creates the same radial plot as the original radial_phylo_analyzer.py
but uses gene-based data (proteins mapped to unique genes).

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
import math
import re
from collections import defaultdict
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import os

class GeneBasedRadialPlot:
    def __init__(self, raw_data_path='raw_data.txt'):
        self.raw_data_path = raw_data_path
        self.df = None
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
    
    def load_and_filter_to_genes(self):
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
        """Create sample-level matrix for phylogenetic analysis."""
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
    
    def calculate_uniqueness_data(self, sample_matrix):
        """Calculate uniqueness data for each sample (same as original)."""
        print("🔍 Calculating uniqueness data for each sample...")
        
        # Transpose to get genes as rows
        gene_matrix = sample_matrix.T
        
        uniqueness_data = {}
        
        for sample in sample_matrix.index:
            # Get genes detected in this sample
            sample_intensities = gene_matrix[sample]
            detected_genes = gene_matrix.index[sample_intensities > 0].tolist()
            
            # Calculate uniqueness
            unique_sample = 0
            species_only = 0
            
            for gene in detected_genes:
                # Check if gene is detected in other samples
                other_samples = [s for s in sample_matrix.index if s != sample]
                other_intensities = gene_matrix.loc[gene, other_samples]
                
                # Check if gene is unique to this sample
                if np.sum(other_intensities.values) == 0:
                    unique_sample += 1
                else:
                    # Check if gene is only in samples from the same species
                    sample_species = self._get_species_from_sample(sample)
                    same_species_samples = [s for s in other_samples if self._get_species_from_sample(s) == sample_species]
                    other_species_samples = [s for s in other_samples if self._get_species_from_sample(s) != sample_species]
                    
                    same_species_intensities = gene_matrix.loc[gene, same_species_samples]
                    other_species_intensities = gene_matrix.loc[gene, other_species_samples]
                    
                    if np.sum(same_species_intensities.values) > 0 and np.sum(other_species_intensities.values) == 0:
                        species_only += 1
            
            total = len(detected_genes)
            uniqueness_data[sample] = {
                'total': total,
                'unique_sample': unique_sample,
                'species_only': species_only
            }
        
        return uniqueness_data
    
    def _get_species_from_sample(self, sample_name):
        """Extract species from sample name."""
        for species, prefix in self.species_prefixes.items():
            if sample_name.startswith(prefix):
                return species
        return "Unknown"
    
    def braycurtis_upgma(self, matrix, labels, tag):
        """Calculate Bray-Curtis distance and UPGMA clustering."""
        print(f"🌳 Calculating Bray-Curtis distance and UPGMA clustering for {len(labels)} {tag}...")
        
        D = pdist(matrix, metric="braycurtis")
        Z = linkage(D, method="average")
        return Z, D
    
    def plot_radial_phylogenetic_tree(self, Z, labels, uniqueness_data, out_png):
        """Create radial phylogenetic tree (same as original)."""
        print("🎨 Creating gene-based radial phylogenetic tree...")
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        
        # Define colors for each species
        species_colors = {
            "Lb": "#1f77b4",  # Blue
            "Lg": "#ff7f0e",  # Orange
            "Ln": "#2ca02c",  # Green
            "Lp": "#d62728",  # Red
        }
        
        def strain_to_species(strain):
            """Extract species from strain name."""
            for species, prefix in self.species_prefixes.items():
                if strain.startswith(prefix):
                    return species
            return "Unknown"
        
        def clean_label(label):
            """Clean sample label for display - remove species prefix."""
            if label.startswith('Intensity '):
                # Remove 'Intensity ' prefix and extract just the sample ID
                sample_part = label.replace('Intensity ', '')
                # Extract the sample ID (everything after the species prefix)
                for species, prefix in self.species_prefixes.items():
                    if sample_part.startswith(species):
                        # Remove species prefix and any underscore
                        sample_id = sample_part.replace(species, '').lstrip('_')
                        return sample_id
                return sample_part
            return label
        
        # Count samples per species
        species_counts = defaultdict(int)
        for label in labels:
            species = strain_to_species(label)
            species_counts[species] += 1
        
        total_samples = len(labels)
        
        # Define section angles proportional to sample counts
        current_angle = 0
        section_angles = {}
        for species in ["Lb", "Lg", "Ln", "Lp"]:
            if species in species_counts:
                angle_span = (species_counts[species] / total_samples) * 360
                section_angles[species] = (current_angle, current_angle + angle_span)
                current_angle += angle_span
            else:
                section_angles[species] = (current_angle, current_angle)
        
        print(f"Species sample counts: {species_counts}")
        print(f"Section angles: {section_angles}")
        
        # Plot background sections
        for species, (start_angle, end_angle) in section_angles.items():
            color = species_colors[species]
            alpha = 0.06
            wedge = Wedge((0, 0), 1.1, start_angle, end_angle, 
                         facecolor=color, alpha=alpha, edgecolor='none')
            ax.add_patch(wedge)
            
            # Add species label
            mid_angle = math.radians((start_angle + end_angle) / 2)
            x = 0.8 * math.cos(mid_angle)
            y = 0.8 * math.sin(mid_angle)
            ax.text(x, y, species, fontsize=16, fontweight='bold', 
                   ha='center', va='center', color=color)
        
        # Determine leaf order from dendrogram, then place leaves by species sections
        dendro_info = dendrogram(Z, labels=list(range(len(labels))), no_plot=True)
        leaf_order = dendro_info['leaves']  # indices 0..n-1

        # Build ordered indices per species preserving dendrogram order
        species_to_indices = {"Lb": [], "Lg": [], "Ln": [], "Lp": []}
        for idx in leaf_order:
            lab = labels[idx]
            sp = strain_to_species(lab)
            if sp in species_to_indices:
                species_to_indices[sp].append(idx)

        # Assign angles to each leaf within species sections
        leaf_angle_deg = {}
        leaf_rank_in_species = {}
        species_step_deg = {}
        margin = 8  # degrees margin within each section
        for sp, idx_list in species_to_indices.items():
            if not idx_list:
                continue
            a0, a1 = section_angles[sp]
            a0 += margin
            a1 -= margin
            if len(idx_list) == 1:
                leaf_angle_deg[idx_list[0]] = (a0 + a1) / 2
                leaf_rank_in_species[idx_list[0]] = 0
                species_step_deg[sp] = (a1 - a0)
            else:
                step = (a1 - a0) / (len(idx_list) - 1)
                species_step_deg[sp] = step
                for j, leaf_idx in enumerate(idx_list):
                    leaf_angle_deg[leaf_idx] = a0 + j * step
                    leaf_rank_in_species[leaf_idx] = j

        # Map node -> species color (pure or majority)
        n_leaves = len(labels)
        max_dist = float(Z[:, 2].max()) if len(Z) > 0 else 1.0
        inner_radius = 0.15
        outer_radius = 0.95

        # Precompute species sets for each node id (leaf ids 0..n-1, internal ids n..n+Z-1)
        cluster_species = {}

        def get_node_species(node_id):
            if node_id in cluster_species:
                return cluster_species[node_id]
            if node_id < n_leaves:
                sp = strain_to_species(labels[node_id])
                cluster_species[node_id] = {sp}
            else:
                zrow = Z[node_id - n_leaves]
                left = int(zrow[0])
                right = int(zrow[1])
                cluster_species[node_id] = get_node_species(left) | get_node_species(right)
            return cluster_species[node_id]

        # Radial coordinate for a node given its distance
        def radius_for_dist(dist):
            if max_dist == 0:
                return outer_radius
            # leaves at outer_radius (distance 0), root near inner_radius (distance max)
            return outer_radius - (dist / max_dist) * (outer_radius - inner_radius)

        # Draw recursively: for each merge draw two radial segments up to parent radius, then an arc
        def draw_node(node_id):
            if node_id < n_leaves:
                theta_deg = leaf_angle_deg.get(node_id, 0.0)
                theta = math.radians(theta_deg)
                r = outer_radius
                return r, theta
            # internal node
            zrow = Z[node_id - n_leaves]
            left = int(zrow[0])
            right = int(zrow[1])
            height = float(zrow[2])
            r_parent = radius_for_dist(height)

            r_left, th_left = draw_node(left)
            r_right, th_right = draw_node(right)

            # determine branch color
            species_set = get_node_species(node_id)
            if len(species_set) == 1:
                sp = list(species_set)[0]
                color = species_colors.get(sp, "black")
            else:
                # choose majority by counting leaves
                def count_species(nid):
                    if nid < n_leaves:
                        return {strain_to_species(labels[nid]): 1}
                    l = int(Z[nid - n_leaves][0])
                    r = int(Z[nid - n_leaves][1])
                    counts = {}
                    for child in (l, r):
                        cc = count_species(child)
                        for k, v in cc.items():
                            counts[k] = counts.get(k, 0) + v
                    return counts
                counts = count_species(node_id)
                majority_sp = max(counts, key=counts.get)
                color = species_colors.get(majority_sp, "black")

            # radial segments from children up to parent radius
            for r_child, th_child in ((r_left, th_left), (r_right, th_right)):
                x1, y1 = r_child * math.cos(th_child), r_child * math.sin(th_child)
                x2, y2 = r_parent * math.cos(th_child), r_parent * math.sin(th_child)
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, alpha=0.9)

            # arc connecting children at parent radius
            th1_deg = math.degrees(min(th_left, th_right))
            th2_deg = math.degrees(max(th_left, th_right))
            arc = mpatches.Arc((0, 0), 2*r_parent, 2*r_parent, angle=0,
                               theta1=th1_deg, theta2=th2_deg, color=color, linewidth=1.5, alpha=0.9)
            ax.add_patch(arc)

            # return parent's polar coordinate (angle mid of children)
            th_parent = (th_left + th_right) / 2.0
            return r_parent, th_parent

        # Kick off drawing from root id
        if len(Z) > 0:
            root_id = n_leaves + len(Z) - 1
            draw_node(root_id)

        # After branches are drawn, overlay leaf labels and pies
        for leaf_idx in range(len(labels)):
            if leaf_idx not in leaf_angle_deg:
                continue
            theta = math.radians(leaf_angle_deg[leaf_idx])
            r_label = outer_radius * 1.08
            x = r_label * math.cos(theta)
            y = r_label * math.sin(theta)
            species = strain_to_species(labels[leaf_idx])
            color = species_colors.get(species, "black")

            # DUAL-PIE CHART: Outer pie + inner bar
            rank = leaf_rank_in_species.get(leaf_idx, 0)
            ring = rank % 3
            r_pie = outer_radius * (1.00 - 0.05 * ring)
            xp = r_pie * math.cos(theta)
            yp = r_pie * math.sin(theta)
            lab_full = labels[leaf_idx]
            
            # Get unique percentage for label
            unique_pct = 0
            if lab_full in uniqueness_data:
                data = uniqueness_data[lab_full]
                total = data['total']
                if total > 0:
                    unique_pct = (data['unique_sample'] / total) * 100
            
            # Draw sample name and unique percentage below it
            ax.text(x, y, clean_label(labels[leaf_idx]), fontsize=8, ha='center', va='center',
                    color=color, fontweight='bold', rotation=leaf_angle_deg[leaf_idx]-90,
                    rotation_mode='anchor', path_effects=[])
            
            # Position unique percentage below the name
            unique_r = r_label - 0.02
            px = unique_r * math.cos(theta)
            py = unique_r * math.sin(theta)
            ax.text(px, py, f"({unique_pct:.1f}%)", fontsize=6, ha='center', va='center',
                    color=color, fontweight='normal', rotation=leaf_angle_deg[leaf_idx]-90,
                    rotation_mode='anchor')
            
            if lab_full in uniqueness_data:
                data = uniqueness_data[lab_full]
                total = data['total']
                if total > 0:
                    # NEW LAYOUT: Outer circle + Inner two pies
                    
                    # Calculate values for the three sections
                    unique_sample = data.get('unique_sample', 0)
                    species_only = data.get('species_only', 0)
                    shared_other_species = total - unique_sample - species_only
                    
                    # Calculate percentages
                    unique_pct = (unique_sample / total) * 100 if total > 0 else 0
                    species_only_pct = (species_only / total) * 100 if total > 0 else 0
                    shared_other_pct = (shared_other_species / total) * 100 if total > 0 else 0
                    
                    # OUTER CIRCLE: Genes shared with other species
                    outer_radius_pie = 0.045
                    if shared_other_species > 0:
                        # Create outer circle showing genes shared with other species
                        ax.add_patch(Wedge((xp, yp), outer_radius_pie, 0, 360,
                                           facecolor='lightgray', alpha=0.7,
                                           edgecolor='black', linewidth=1))
                        
                        # Add number for outer circle
                        ax.text(xp, yp + outer_radius_pie * 0.7, f"{shared_other_species}",
                               fontsize=8, ha='center', va='center', fontweight='bold',
                               color='black')
                    
                    # INNER TWO PIES: Species-only shared + Unique to sample
                    inner_radius_pie = 0.025
                    
                    # First inner pie: Species-only shared (main position)
                    if species_only > 0:
                        ax.add_patch(Wedge((xp, yp), inner_radius_pie, 0, 360,
                                           facecolor=color, alpha=0.4,
                                           edgecolor='black', linewidth=0.8))
                        
                        # Add number for species-only
                        ax.text(xp, yp + inner_radius_pie * 0.5, f"{species_only}",
                               fontsize=6, ha='center', va='center', fontweight='bold',
                               color='black')
                    
                    # Second inner pie: Unique to sample (smaller, offset, lighter color)
                    if unique_sample > 0:
                        # Offset the second pie slightly
                        offset_x = xp + inner_radius_pie * 0.3
                        offset_y = yp - inner_radius_pie * 0.3
                        # Use lighter version of species color
                        light_color = color  # We'll make it lighter with alpha
                        ax.add_patch(Wedge((offset_x, offset_y), inner_radius_pie * 0.7, 0, 360,
                                           facecolor=light_color, alpha=0.2,
                                           edgecolor='black', linewidth=0.8))
                        
                        # Add number for unique sample
                        ax.text(offset_x, offset_y, f"{unique_sample}",
                               fontsize=6, ha='center', va='center', fontweight='bold',
                               color='black')

            # leader line from pie center to label for readability
            ax.plot([xp, x*0.98], [yp, y*0.98], color=color, alpha=0.3, linewidth=0.8)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightgray', alpha=0.7, label='Outer circle: Genes shared with other species'),
            mpatches.Patch(color='blue', alpha=0.4, label='Inner circle: Species-only shared'),
            mpatches.Patch(color='blue', alpha=0.2, label='Inner circle: Unique to sample'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.4, 1))
        
        # Add title
        ax.set_title('Gene-Based Leishmania Phylogenetic Tree\n(Using Unique Gene Mappings)', 
                    fontsize=16, pad=20)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(out_png, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gene-based radial phylogenetic tree saved to: {out_png}")
    
    def run_gene_based_radial_analysis(self):
        """Run complete gene-based radial analysis."""
        print("🚀 Starting gene-based radial phylogenetic analysis...")
        print("=" * 60)
        
        # Load and process data to gene level
        self.load_and_filter_to_genes()
        
        # Create sample matrix
        sample_matrix = self.create_sample_matrix()
        
        # Calculate uniqueness data
        uniqueness_data = self.calculate_uniqueness_data(sample_matrix)
        
        # Create output directory
        output_dir = "gene_radial_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Build phylogenetic tree
        print("🌳 Building phylogenetic tree...")
        Z_bray, D_bray = self.braycurtis_upgma(sample_matrix.values, sample_matrix.index, tag="samples")
        
        # Generate radial plot
        print("🎨 Generating radial phylogenetic plot...")
        self.plot_radial_phylogenetic_tree(Z_bray, sample_matrix.index, uniqueness_data, 
                                         os.path.join(output_dir, "gene_based_radial_tree.png"))
        
        # Save data
        sample_matrix.to_csv(os.path.join(output_dir, "gene_sample_matrix.csv"))
        
        # Print summary
        print("\n📊 GENE-BASED RADIAL ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total entries analyzed: {len(self.df)} (mapped to unique genes)")
        print(f"Sample matrix: {sample_matrix.shape[0]} samples x {sample_matrix.shape[1]} genes")
        print(f"Clustering performed on: {sample_matrix.shape[0]} samples")
        
        # Print some uniqueness statistics
        total_unique = sum(data['unique_sample'] for data in uniqueness_data.values())
        total_species_only = sum(data['species_only'] for data in uniqueness_data.values())
        total_shared = sum(data['total'] - data['unique_sample'] - data['species_only'] for data in uniqueness_data.values())
        
        print(f"\nGene uniqueness summary:")
        print(f"  Total unique to samples: {total_unique}")
        print(f"  Total species-only shared: {total_species_only}")
        print(f"  Total shared across species: {total_shared}")
        
        print(f"\n✅ Gene-based radial analysis complete!")
        print(f"📁 Results saved to: {output_dir}/")
        
        return self.df, uniqueness_data

if __name__ == "__main__":
    analyzer = GeneBasedRadialPlot()
    gene_df, uniqueness_data = analyzer.run_gene_based_radial_analysis()
