#!/usr/bin/env python3
"""
Common Data Processing Module
============================

This module provides standardized data processing for proteomics analysis:
1. Load and filter raw data (no contaminants, no decoys)
2. Extract gene information from Fasta headers
3. Filter to proteins with intensity > 0
4. Filter to protein groups with unique gene mapping
5. Generate standardized sample and species information
6. Create two dataframes: protein-based and gene-based

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import re
import os

class DataProcessor:
    def __init__(self, data_file="raw_data.txt.gz"):
        """
        Initialize the data processor.
        
        Args:
            data_file (str): Path to the raw data file (supports .txt and .gz)
        """
        self.data_file = data_file
        self.df = None
        self.intensity_columns = []
        self.species_mapping = {}
        self.protein_df = None  # All proteins with intensity > 0
        self.gene_df = None     # Only proteins with unique gene mapping
        
    def load_and_process_data(self):
        """Load and process the raw data file with standardized filtering."""
        print("🔄 Loading and processing data...")
        
        # Check if file is gzipped and read accordingly
        if self.data_file.endswith('.gz'):
            import gzip
            print(f"Reading gzipped file: {self.data_file}")
            with gzip.open(self.data_file, 'rt') as f:
                lines = f.readlines()
        else:
            print(f"Reading text file: {self.data_file}")
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
        
        # Parse data lines
        data_rows = []
        for line in lines[1:]:
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
            
            # Extract intensities and Fasta headers
            row_data = {'Protein_IDs': protein_ids}
            
            # Add Fasta headers if available
            if len(parts) > 5:  # Fasta headers is at index 5
                row_data['Fasta headers'] = parts[5].strip()
            
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
        print(f"Loaded {len(self.df)} proteins after filtering decoys/contaminants")
        
        # Filter out undetected proteins (zero intensity across all species)
        print("Filtering out undetected proteins...")
        total_intensity = self.df[self.intensity_columns].sum(axis=1)
        self.df = self.df[total_intensity > 0]
        print(f"Removed undetected proteins. Remaining proteins: {len(self.df)}")
        
        # Extract gene information
        self._extract_gene_information()
        
        # Create standardized dataframes
        self._create_standardized_dataframes()
        
        return self.protein_df, self.gene_df
    
    def _extract_gene_from_fasta_header(self, fasta_header):
        """Extract gene names from Fasta header."""
        genes = []
        if 'GN=' in fasta_header:
            # Extract gene names, stopping at PE= or SV= or end of string
            gene_matches = re.findall(r'GN=([^=]+?)(?:\s+PE=|\s+SV=|$)', fasta_header)
            genes.extend(gene_matches)
        return genes
    
    def _extract_gene_information(self):
        """Extract gene information for each protein."""
        print("🧬 Extracting gene information from Fasta headers...")
        gene_ids = []
        unique_gene_mapping = []  # Track if protein group maps to exactly one unique gene
        
        for idx, row in self.df.iterrows():
            fasta_header = row["Fasta headers"]
            genes = self._extract_gene_from_fasta_header(fasta_header)
            unique_genes = list(set(genes))
            
            # Only assign gene ID if exactly one unique gene
            if len(unique_genes) == 1:
                gene_ids.append(unique_genes[0])
                unique_gene_mapping.append(True)
            else:
                gene_ids.append("Multiple_Genes" if unique_genes else "Unknown")
                unique_gene_mapping.append(False)
        
        # Add gene information to dataframe
        self.df['Gene_ID'] = gene_ids
        self.df['Unique_Gene_Mapping'] = unique_gene_mapping
        
        # Print summary of gene mapping
        total_proteins = len(self.df)
        single_gene_proteins = sum(unique_gene_mapping)
        multiple_gene_proteins = len([x for x in gene_ids if x == "Multiple_Genes"])
        unknown_proteins = len([x for x in gene_ids if x == "Unknown"])
        
        print(f"Gene mapping summary:")
        print(f"  Total protein groups: {total_proteins}")
        print(f"  Protein groups with single gene: {single_gene_proteins} ({single_gene_proteins/total_proteins*100:.1f}%)")
        print(f"  Protein groups with multiple genes: {multiple_gene_proteins} ({multiple_gene_proteins/total_proteins*100:.1f}%)")
        print(f"  Protein groups with unknown genes: {unknown_proteins} ({unknown_proteins/total_proteins*100:.1f}%)")
    
    def _create_standardized_dataframes(self):
        """Create standardized dataframes for protein and gene analysis."""
        print("📊 Creating standardized dataframes...")
        
        # Create protein-based dataframe (all proteins with intensity > 0)
        self.protein_df = self.df.copy()
        
        # Create gene-based dataframe (only proteins with unique gene mapping)
        self.gene_df = self.df[self.df['Unique_Gene_Mapping'] == True].copy()
        
        # Add standardized sample information
        self._add_sample_information()
        
        print(f"✅ Created standardized dataframes:")
        print(f"  Protein-based dataframe: {len(self.protein_df)} proteins")
        print(f"  Gene-based dataframe: {len(self.gene_df)} genes")
    
    def _add_sample_information(self):
        """Add standardized sample and species information."""
        # Create sample information dataframe
        sample_info = []
        for col in self.intensity_columns:
            species = self.species_mapping[col]
            # Remove "Intensity " prefix to get clean sample name
            sample_name = col.replace("Intensity ", "")
            sample_info.append({
                'Original_Column': col,
                'Sample_Name': sample_name,
                'Species': species
            })
        
        self.sample_info_df = pd.DataFrame(sample_info)
        
        # Add sample information to both dataframes
        for df in [self.protein_df, self.gene_df]:
            # Add species columns for each sample
            for _, row in self.sample_info_df.iterrows():
                col = row['Original_Column']
                species = row['Species']
                df[f'Species_{col}'] = species
    
    def get_protein_counts_per_sample(self):
        """Get protein counts per sample (intensity > 0)."""
        results = []
        for col in self.intensity_columns:
            species = self.species_mapping[col]
            sample_name = col.replace("Intensity ", "")
            
            # Count proteins with intensity > 0 in this sample
            detected_proteins = self.protein_df[self.protein_df[col] > 0]
            
            results.append({
                'Sample_Column': col,
                'Sample_Name': sample_name,
                'Species': species,
                'Protein_Groups_Detected': len(detected_proteins)
            })
        
        return pd.DataFrame(results)
    
    def get_gene_counts_per_sample(self):
        """Get gene counts per sample (only unique gene mapping)."""
        results = []
        for col in self.intensity_columns:
            species = self.species_mapping[col]
            sample_name = col.replace("Intensity ", "")
            
            # Count genes with intensity > 0 in this sample
            detected_genes = self.gene_df[self.gene_df[col] > 0]
            unique_genes = detected_genes['Gene_ID'].unique()
            
            # Count unique genes (excluding "Unknown" and "Multiple_Genes")
            unique_genes_clean = [gene for gene in unique_genes if gene not in ["Unknown", "Multiple_Genes"]]
            
            results.append({
                'Sample_Column': col,
                'Sample_Name': sample_name,
                'Species': species,
                'Unique_Genes_Detected': len(unique_genes_clean)
            })
        
        return pd.DataFrame(results)
    
    def get_species_totals(self):
        """Get collapsed totals per species."""
        species_totals = {}
        
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            # Get all samples for this species
            species_cols = [c for c in self.intensity_columns if self.species_mapping[c] == species]
            
            # Protein totals (detected in any sample of this species)
            species_proteins = self.protein_df[self.protein_df[species_cols].sum(axis=1) > 0]
            
            # Gene totals (detected in any sample of this species)
            species_genes = self.gene_df[self.gene_df[species_cols].sum(axis=1) > 0]
            unique_genes = species_genes['Gene_ID'].unique()
            unique_genes_clean = [gene for gene in unique_genes if gene not in ["Unknown", "Multiple_Genes"]]
            
            species_totals[species] = {
                'Protein_Groups_Total': len(species_proteins),
                'Unique_Genes_Total': len(unique_genes_clean)
            }
        
        return species_totals
    
    def get_unique_accessions_per_sample(self):
        """
        Generate a table with truly unique protein and gene accessions for each sample.
        Only includes proteins/genes that are detected in ONLY this sample.
        
        Returns:
            pd.DataFrame: Table with columns:
                - Sample_Name: Clean sample name
                - Species: Species identifier (Lb, Lg, Ln, Lp)
                - Unique_Protein_Accessions: List of truly unique protein accessions
                - Unique_Gene_Accessions: List of truly unique gene accessions
        """
        results = []
        
        for col in self.intensity_columns:
            species = self.species_mapping[col]
            sample_name = col.replace("Intensity ", "")
            
            # Get proteins with intensity > 0 in this sample
            detected_proteins = self.protein_df[self.protein_df[col] > 0]
            
            # Find proteins that are ONLY detected in this sample
            truly_unique_proteins = []
            for _, row in detected_proteins.iterrows():
                protein_id = row['Protein_IDs']
                # Check if this protein is detected in any other sample
                other_samples = [c for c in self.intensity_columns if c != col]
                detected_in_others = False
                
                for other_col in other_samples:
                    if row[other_col] > 0:
                        detected_in_others = True
                        break
                
                # If not detected in any other sample, it's truly unique
                if not detected_in_others:
                    # Format: replace semicolons with pipes within protein groups
                    formatted_protein = protein_id.replace(';', '|')
                    truly_unique_proteins.append(formatted_protein)
            
            # Get genes with intensity > 0 in this sample (only unique gene mapping)
            detected_genes = self.gene_df[self.gene_df[col] > 0]
            
            # Find genes that are ONLY detected in this sample
            truly_unique_genes = []
            for _, row in detected_genes.iterrows():
                gene_id = row['Gene_ID']
                
                # Skip "Unknown" and "Multiple_Genes"
                if gene_id in ["Unknown", "Multiple_Genes"]:
                    continue
                
                # Check if this gene is detected in any other sample
                other_samples = [c for c in self.intensity_columns if c != col]
                detected_in_others = False
                
                for other_col in other_samples:
                    if row[other_col] > 0:
                        detected_in_others = True
                        break
                
                # If not detected in any other sample, it's truly unique
                if not detected_in_others:
                    truly_unique_genes.append(gene_id)
            
            results.append({
                'Sample_Name': sample_name,
                'Species': species,
                'Unique_Protein_Accessions': ';'.join(truly_unique_proteins),
                'Unique_Gene_Accessions': '|'.join(truly_unique_genes)
            })
        
        return pd.DataFrame(results)
    
    def save_processed_data(self, output_dir="processed_data"):
        """Save processed dataframes and sample information."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save dataframes
        self.protein_df.to_csv(f"{output_dir}/protein_dataframe.csv", index=False)
        self.gene_df.to_csv(f"{output_dir}/gene_dataframe.csv", index=False)
        self.sample_info_df.to_csv(f"{output_dir}/sample_information.csv", index=False)
        
        # Generate and save summary counts
        protein_counts = self.get_protein_counts_per_sample()
        gene_counts = self.get_gene_counts_per_sample()
        species_totals = self.get_species_totals()
        
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
        
        combined_results = pd.concat([combined_results, pd.DataFrame([overall_summary])], ignore_index=True)
        combined_results.to_csv(f"{output_dir}/protein_gene_counts_summary.csv", index=False)
        
        # Generate and save unique accessions table
        unique_accessions_df = self.get_unique_accessions_per_sample()
        unique_accessions_df.to_csv(f"{output_dir}/unique_accessions_per_sample.csv", index=False)
        
        print(f"✅ Processed data saved to: {output_dir}/")
        print(f"  - protein_dataframe.csv: {len(self.protein_df)} proteins")
        print(f"  - gene_dataframe.csv: {len(self.gene_df)} genes")
        print(f"  - sample_information.csv: {len(self.sample_info_df)} samples")
        print(f"  - protein_gene_counts_summary.csv: Summary counts")
        print(f"  - unique_accessions_per_sample.csv: Unique protein/gene accessions per sample")

def main():
    """Main function to run the data processor."""
    processor = DataProcessor()
    protein_df, gene_df = processor.load_and_process_data()
    processor.save_processed_data()
    
    return processor

if __name__ == "__main__":
    processor = main()
