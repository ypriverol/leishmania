#!/usr/bin/env python3
"""
Simplified Metabolic Heatmap Generator
======================================

This script generates a metabolic heatmap from general protein data
by extracting functional information from protein descriptions in the data files.

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

class SimplifiedMetabolicHeatmapGenerator:
    def __init__(self):
        # Functional category keywords extracted from protein descriptions
        self.functional_keywords = {
            'Oxidative Phosphorylation': [
                'ATP synthase', 'cytochrome', 'oxidase', 'dehydrogenase', 'NADH', 'FADH', 
                'electron transport', 'respiratory chain', 'complex I', 'complex II', 'complex III', 
                'complex IV', 'complex V', 'ubiquinone', 'coenzyme Q', 'succinate', 'malate', 
                'fumarate', 'isocitrate', 'alpha-ketoglutarate', 'NAD', 'FAD', 'ubiquinol'
            ],
            'Glycolysis': [
                'glucose', 'hexokinase', 'phosphofructokinase', 'pyruvate kinase', 'glyceraldehyde', 
                'phosphoglycerate', 'enolase', 'lactate dehydrogenase', 'aldolase', 'triose phosphate', 
                'fructose', 'glucose-6-phosphate', 'fructose-6-phosphate', 'phosphoglycerate kinase',
                'glyceraldehyde-3-phosphate dehydrogenase', 'pyruvate', 'lactate', 'glucose transporter'
            ],
            'TCA Cycle': [
                'citrate synthase', 'aconitase', 'isocitrate dehydrogenase', 'alpha-ketoglutarate dehydrogenase', 
                'succinyl-CoA', 'succinate dehydrogenase', 'fumarase', 'malate dehydrogenase', 
                'citric acid cycle', 'krebs cycle', 'tricarboxylic acid', 'citrate', 'isocitrate',
                'alpha-ketoglutarate', 'succinate', 'fumarate', 'malate', 'oxaloacetate'
            ],
            'Amino Acid Metabolism': [
                'alanine', 'aspartate', 'glutamate', 'glutamine', 'arginine', 'lysine', 'histidine', 
                'phenylalanine', 'tyrosine', 'tryptophan', 'leucine', 'isoleucine', 'valine', 
                'methionine', 'cysteine', 'serine', 'threonine', 'asparagine', 'proline', 'glycine', 
                'aminotransferase', 'transaminase', 'dehydrogenase', 'synthetase', 'synthase'
            ],
            'Oxidative Stress': [
                'catalase', 'superoxide dismutase', 'peroxiredoxin', 'thioredoxin', 'glutaredoxin', 
                'glutathione', 'peroxidase', 'reductase', 'oxidase', 'reactive oxygen', 'ROS', 
                'antioxidant', 'redox', 'SOD', 'CAT', 'GPX', 'GR', 'GST'
            ],
            'Signal Transduction': [
                'kinase', 'phosphatase', 'receptor', 'G protein', 'cAMP', 'cGMP', 'calcium', 
                'calmodulin', 'protein kinase', 'tyrosine kinase', 'serine kinase', 'threonine kinase', 
                'MAP kinase', 'JNK', 'ERK', 'p38', 'AKT', 'PI3K', 'phospholipase'
            ],
            'Membrane Transport': [
                'transporter', 'channel', 'pump', 'ATPase', 'sodium', 'potassium', 'calcium', 
                'chloride', 'membrane', 'integral membrane', 'transmembrane', 'receptor', 
                'adhesion', 'cell adhesion', 'ABC transporter', 'P-type ATPase', 'V-type ATPase'
            ],
            'Proteasome': [
                'proteasome', 'ubiquitin', 'ubiquitination', 'protease', 'peptidase', '20S', '26S', 
                '19S', 'regulatory particle', 'core particle', 'ubiquitin ligase', 'deubiquitinase'
            ],
            'Ribosomal Proteins': [
                'ribosomal protein', 'ribosome', 'rRNA', 'translation', 'elongation factor', 
                'initiation factor', 'release factor', 'ribosomal RNA', 'ribosomal protein L',
                'ribosomal protein S', 'ribosomal protein P'
            ],
            'DNA Replication': [
                'DNA polymerase', 'helicase', 'primase', 'ligase', 'topoisomerase', 'gyrase', 
                'replication', 'origin', 'replicon', 'replisome', 'DNA primase', 'DNA ligase',
                'DNA helicase', 'DNA topoisomerase'
            ],
            'Transcription': [
                'RNA polymerase', 'transcription factor', 'promoter', 'enhancer', 'silencer', 
                'transcription', 'transcriptional', 'RNA synthesis', 'transcription initiation',
                'transcription elongation', 'transcription termination'
            ],
            'Nuclear Proteins': [
                'nuclear', 'nucleus', 'chromatin', 'histone', 'nucleosome', 'nuclear pore', 
                'nuclear envelope', 'nuclear matrix', 'nuclear import', 'nuclear export'
            ],
            'Cytoskeletal Proteins': [
                'actin', 'tubulin', 'microtubule', 'microfilament', 'intermediate filament', 
                'cytoskeleton', 'myosin', 'dynein', 'kinesin', 'motor protein', 'actin-binding',
                'tubulin-binding', 'microtubule-associated protein'
            ],
            'Heat Shock Proteins': [
                'heat shock', 'HSP', 'chaperone', 'chaperonin', 'GroEL', 'GroES', 'DnaK', 'DnaJ', 
                'GrpE', 'Hsp70', 'Hsp90', 'Hsp60', 'molecular chaperone', 'protein folding'
            ],
            'Histones': [
                'histone', 'H1', 'H2A', 'H2B', 'H3', 'H4', 'nucleosome', 'chromatin', 'histone modification',
                'histone acetyltransferase', 'histone deacetylase', 'histone methyltransferase'
            ],
            'Metabolism': [
                'metabolism', 'metabolic', 'biosynthesis', 'catabolism', 'anabolism', 'synthesis', 
                'degradation', 'breakdown', 'metabolic pathway', 'metabolic enzyme'
            ],
            'Autophagy': [
                'autophagy', 'autophagosome', 'lysosome', 'vacuole', 'ATG', 'autophagic', 
                'macroautophagy', 'microautophagy', 'autophagy-related', 'autophagosome formation'
            ],
            'Cell Cycle': [
                'cyclin', 'CDK', 'cell cycle', 'mitosis', 'meiosis', 'spindle', 'centrosome', 
                'centromere', 'telomere', 'checkpoint', 'cell division', 'mitotic', 'meiotic'
            ],
            'Lipid Metabolism': [
                'lipid', 'fatty acid', 'triglyceride', 'phospholipid', 'sphingolipid', 'cholesterol',
                'lipid biosynthesis', 'fatty acid synthase', 'acyl-CoA', 'lipase', 'phospholipase'
            ],
            'Nucleotide Metabolism': [
                'nucleotide', 'purine', 'pyrimidine', 'adenine', 'guanine', 'cytosine', 'thymine',
                'uracil', 'nucleotide biosynthesis', 'purine biosynthesis', 'pyrimidine biosynthesis'
            ]
        }
    
    def extract_functional_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract functional categories from protein descriptions in the data files.
        """
        print("Extracting functional categories from protein descriptions...")
        
        functional_categories = []
        
        for idx, row in df.iterrows():
            description = str(row['Fasta headers']).upper()
            
            category_found = False
            
            for category, keywords in self.functional_keywords.items():
                for keyword in keywords:
                    if keyword.upper() in description:
                        functional_categories.append(category)
                        category_found = True
                        break
                if category_found:
                    break
            
            if not category_found:
                functional_categories.append('Other')
        
        df['Functional_Category'] = functional_categories
        return df
    
    def calculate_species_expression(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate median expression for each species.
        """
        print("Calculating species-specific expressions...")
        
        # Get intensity columns for each species
        lb_cols = [col for col in df.columns if col.startswith('Intensity Lb_')]
        lg_cols = [col for col in df.columns if col.startswith('Intensity Lg_')]
        ln_cols = [col for col in df.columns if col.startswith('Intensity Ln_')]
        lp_cols = [col for col in df.columns if col.startswith('Intensity Lp_')]
        
        # Calculate median for each species
        df['Lb'] = df[lb_cols].median(axis=1)
        df['Lg'] = df[lg_cols].median(axis=1)
        df['Ln'] = df[ln_cols].median(axis=1)
        df['Lp'] = df[lp_cols].median(axis=1)
        
        # Apply log2 transformation
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            df[species] = np.log2(df[species] + 1)
        
        return df
    
    def filter_unique_gene_proteins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter proteins to only include those that map to unique genes.
        """
        print("Filtering for proteins that map to unique genes...")
        
        # Filter for proteins with unique gene mapping
        unique_gene_proteins = df[df['Unique_Gene_Mapping'] == True].copy()
        
        print(f"Original proteins: {len(df)}")
        print(f"Proteins with unique gene mapping: {len(unique_gene_proteins)}")
        print(f"Proteins removed: {len(df) - len(unique_gene_proteins)}")
        
        return unique_gene_proteins
    
    def create_expression_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a matrix of protein expression across species for heatmap analysis.
        """
        # Select proteins with functional categories (excluding 'Other')
        functional_proteins = df[df['Functional_Category'] != 'Other'].copy()
        
        if functional_proteins.empty:
            print("No proteins with functional categories found")
            return pd.DataFrame()
        
        # Filter for proteins with non-zero intensity in at least one species
        non_zero_proteins = functional_proteins[
            (functional_proteins['Lb'] > 0) | 
            (functional_proteins['Lg'] > 0) | 
            (functional_proteins['Ln'] > 0) | 
            (functional_proteins['Lp'] > 0)
        ].copy()
        
        print(f"Found {len(functional_proteins)} proteins with functional categories")
        print(f"Proteins with non-zero intensity: {len(non_zero_proteins)}")
        
        if non_zero_proteins.empty:
            print("No proteins with non-zero intensity found")
            return pd.DataFrame()
        
        # Group by functional category and calculate median expression
        functional_expression = non_zero_proteins.groupby('Functional_Category')[['Lb', 'Lg', 'Ln', 'Lp']].median()
        
        # Calculate log fold change relative to mean
        mean_across_species = functional_expression.mean(axis=1)
        log_fold_change = functional_expression.sub(mean_across_species, axis=0)
        
        return log_fold_change
    
    def create_enhanced_heatmap(self, expression_matrix: pd.DataFrame, output_file: str = 'refined_metabolic_heatmap.png'):
        """
        Create an enhanced heatmap with statistical annotations.
        """
        print("Creating enhanced functional category heatmap...")
        
        # Include all species that have data
        expression_matrix_filtered = expression_matrix.copy()
        
        # Remove rows with all NaN values
        expression_matrix_filtered = expression_matrix_filtered.dropna(how='all')
        
        # Fill remaining NaN values with 0 for visualization
        expression_matrix_filtered = expression_matrix_filtered.fillna(0)
        
        print(f"Creating heatmap with {len(expression_matrix_filtered)} functional categories")
        print(f"Species included: {list(expression_matrix_filtered.columns)}")
        
        # Create the figure (single panel)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create the main heatmap
        sns.heatmap(expression_matrix_filtered, 
                   cmap='RdBu_r', 
                   center=0,
                   annot=True, 
                   fmt='.2f',
                   cbar_kws={'label': 'Log2 Fold Change', 'shrink': 0.8},
                   ax=ax,
                   linewidths=0.5,
                   linecolor='white')
        
        # Customize the main heatmap
        ax.set_title('Functional Category Expression in Leishmania Species\n' + 
                    'Log2-Transformed Expression Values (Median by Category)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Leishmania Species', fontsize=14, fontweight='bold')
        ax.set_ylabel('Functional Categories', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Enhanced functional category heatmap saved as {output_file}")
        
        return fig

def main():
    """
    Main function to generate the refined metabolic heatmap.
    """
    generator = SimplifiedMetabolicHeatmapGenerator()
    
    # Load the processed protein data
    print("Loading processed protein data...")
    try:
        df = pd.read_csv('refined_processed_protein_data.csv')
        print(f"Loaded {len(df)} proteins from refined_processed_protein_data.csv")
    except FileNotFoundError:
        print("Error: refined_processed_protein_data.csv not found.")
        print("Please ensure the processed data file is available.")
        return
    
    # Filter for unique gene proteins
    print("\nFiltering for unique gene proteins...")
    df = generator.filter_unique_gene_proteins(df)
    
    # Extract functional categories from protein descriptions
    print("\nExtracting functional categories from protein descriptions...")
    df = generator.extract_functional_categories(df)
    
    # Calculate species expressions
    print("\nCalculating species expressions...")
    df = generator.calculate_species_expression(df)
    
    # Create expression matrix
    print("\nCreating expression matrix...")
    expression_matrix = generator.create_expression_matrix(df)
    
    if not expression_matrix.empty:
        # Generate the heatmap
        print("\nGenerating heatmap...")
        generator.create_enhanced_heatmap(expression_matrix, 'refined_metabolic_heatmap.png')
        
        print("\n" + "="*60)
        print("HEATMAP GENERATION COMPLETE!")
        print("="*60)
        print("\nGenerated file:")
        print("- refined_metabolic_heatmap.png")
        
        print(f"\nExpression Matrix:")
        print(expression_matrix)
        
        # Print functional category distribution
        print(f"\nFunctional Category Distribution:")
        category_counts = df['Functional_Category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count} proteins")
    else:
        print("No functional categories found for analysis.")

if __name__ == "__main__":
    main()
