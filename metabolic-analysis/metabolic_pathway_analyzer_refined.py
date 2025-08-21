#!/usr/bin/env python3
"""
Refined Metabolic Pathway Analyzer
==================================

This script performs comprehensive metabolic pathway analysis with:
1. Zero value filtering for cleaner analysis
2. Enhanced pathway annotation with improved keyword matching
3. Statistical analysis with differential expression testing
4. Advanced visualizations with quality control metrics
5. Comprehensive reporting

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

class RefinedMetabolicPathwayAnalyzer:
    def __init__(self):
        # Enhanced pathway keywords with more comprehensive coverage
        self.pathway_keywords = {
            'Oxidative Phosphorylation': [
                'ATP synthase', 'cytochrome', 'oxidase', 'dehydrogenase', 'NADH', 'FADH', 
                'electron transport', 'respiratory chain', 'complex I', 'complex II', 'complex III', 
                'complex IV', 'complex V', 'ubiquinone', 'coenzyme Q', 'succinate', 'malate', 
                'fumarate', 'isocitrate', 'alpha-ketoglutarate', 'NAD', 'FAD', 'ubiquinol',
                'cytochrome c', 'cytochrome b', 'cytochrome a', 'cytochrome oxidase'
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
            'Pentose Phosphate Pathway': [
                'glucose-6-phosphate dehydrogenase', '6-phosphogluconate dehydrogenase', 'ribulose', 
                'ribose', 'pentose', 'transketolase', 'transaldolase', 'phosphogluconate', 'ribulose-5-phosphate',
                'xylulose-5-phosphate', 'sedoheptulose-7-phosphate', 'erythrose-4-phosphate'
            ],
            'Amino Acid Metabolism': [
                'alanine', 'aspartate', 'glutamate', 'glutamine', 'arginine', 'lysine', 'histidine', 
                'phenylalanine', 'tyrosine', 'tryptophan', 'leucine', 'isoleucine', 'valine', 
                'methionine', 'cysteine', 'serine', 'threonine', 'asparagine', 'proline', 'glycine', 
                'aminotransferase', 'transaminase', 'dehydrogenase', 'synthetase', 'synthase',
                'decarboxylase', 'deaminase', 'racemase'
            ],
            'Oxidative Stress': [
                'catalase', 'superoxide dismutase', 'peroxiredoxin', 'thioredoxin', 'glutaredoxin', 
                'glutathione', 'peroxidase', 'reductase', 'oxidase', 'reactive oxygen', 'ROS', 
                'antioxidant', 'redox', 'SOD', 'CAT', 'GPX', 'GR', 'GST', 'glutathione peroxidase',
                'glutathione reductase', 'glutathione S-transferase'
            ],
            'Signal Transduction': [
                'kinase', 'phosphatase', 'receptor', 'G protein', 'cAMP', 'cGMP', 'calcium', 
                'calmodulin', 'protein kinase', 'tyrosine kinase', 'serine kinase', 'threonine kinase', 
                'MAP kinase', 'JNK', 'ERK', 'p38', 'AKT', 'PI3K', 'phospholipase', 'adenylate cyclase', 
                'guanylate cyclase', 'phosphodiesterase', 'adenylyl cyclase'
            ],
            'Membrane Transport': [
                'transporter', 'channel', 'pump', 'ATPase', 'sodium', 'potassium', 'calcium', 
                'chloride', 'membrane', 'integral membrane', 'transmembrane', 'receptor', 
                'adhesion', 'cell adhesion', 'ABC transporter', 'P-type ATPase', 'V-type ATPase',
                'sodium-potassium ATPase', 'calcium ATPase', 'proton pump', 'membrane-associated',
                'membrane protein', 'transport', 'carrier', 'permease', 'symporter', 'antiporter'
            ],
            'Proteasome': [
                'proteasome', 'ubiquitin', 'ubiquitination', 'protease', 'peptidase', '20S', '26S', 
                '19S', 'regulatory particle', 'core particle', 'ubiquitin ligase', 'deubiquitinase',
                'ubiquitin-conjugating enzyme', 'ubiquitin-activating enzyme'
            ],
            'Ribosomal Proteins': [
                'ribosomal protein', 'ribosome', 'rRNA', 'translation', 'elongation factor', 
                'initiation factor', 'release factor', 'ribosomal RNA', 'ribosomal protein L',
                'ribosomal protein S', 'ribosomal protein P'
            ],
            'Translation Factors': [
                'elongation factor', 'initiation factor', 'release factor', 'translation factor', 
                'EF-', 'IF-', 'RF-', 'eIF', 'eEF', 'eRF', 'translation elongation', 'translation initiation'
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
                'nuclear envelope', 'nuclear matrix', 'nuclear import', 'nuclear export',
                'nuclear localization signal', 'nuclear export signal'
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
            'Glycosomal Proteins': [
                'glycosome', 'glycosomal', 'peroxisome', 'peroxisomal', 'glycolytic enzyme', 
                'glycosomal enzyme', 'peroxisomal enzyme', 'glycosomal targeting'
            ],
            'Lipid Metabolism': [
                'lipid', 'fatty acid', 'triglyceride', 'phospholipid', 'sphingolipid', 'cholesterol',
                'lipid biosynthesis', 'fatty acid synthase', 'acyl-CoA', 'lipase', 'phospholipase',
                'sphingomyelin', 'ceramide', 'ganglioside'
            ],
            'Nucleotide Metabolism': [
                'nucleotide', 'purine', 'pyrimidine', 'adenine', 'guanine', 'cytosine', 'thymine',
                'uracil', 'nucleotide biosynthesis', 'purine biosynthesis', 'pyrimidine biosynthesis',
                'nucleoside', 'nucleoside kinase', 'nucleotidase'
            ]
        }
        
        # Initialize column mappings
        self.intensity_columns = []
        self.species_column_mapping = {'Lb': [], 'Lg': [], 'Ln': [], 'Lp': []}
        
    def parse_raw_data(self, filename: str) -> pd.DataFrame:
        """
        Parse raw data with proper species identification and zero filtering.
        """
        print("Parsing raw data from", filename)
        
        # First, read the header to identify intensity columns
        with open(filename, 'r') as f:
            header_line = f.readline().strip()
        
        # Parse header to identify intensity columns
        self._parse_header(header_line)
        
        # Now parse the data
        proteins = []
        line_count = 0
        
        with open(filename, 'r') as f:
            # Skip header
            next(f)
            
            for line in f:
                line_count += 1
                if line_count % 1000 == 0:
                    print(f"Processed {line_count} lines...")
                
                protein_entry = self._parse_protein_line(line.strip())
                if protein_entry:
                    proteins.append(protein_entry)
        
        print(f"Successfully parsed {len(proteins)} proteins")
        
        # Convert to DataFrame
        df = pd.DataFrame(proteins)
        
        # Print filtering summary
        print(f"\nFILTERING SUMMARY:")
        print(f"Proteins after removing decoys/contaminants: {len(proteins)}")
        print(f"Proteins after removing undetected (zero intensity): {len(proteins)}")
        print(f"Total proteins removed: {11830 - len(proteins)}")
        
        # Add pathway annotations
        print("Adding enhanced pathway annotations...")
        df = self._add_enhanced_pathway_annotations(df)
        
        return df
    
    def _parse_header(self, header_line: str):
        """
        Parse header to identify intensity columns and map them to species.
        """
        columns = header_line.split('\t')
        
        for i, col in enumerate(columns):
            if col.startswith('Intensity '):
                self.intensity_columns.append(i)
                
                # Extract species from column name
                if 'Intensity Lb_' in col:
                    self.species_column_mapping['Lb'].append(i)
                elif 'Intensity Lg_' in col:
                    self.species_column_mapping['Lg'].append(i)
                elif 'Intensity Ln_' in col:
                    self.species_column_mapping['Ln'].append(i)
                elif 'Intensity Lp_' in col:
                    self.species_column_mapping['Lp'].append(i)
        
        print(f"Found intensity columns for species:")
        for species, cols in self.species_column_mapping.items():
            print(f"  {species}: {len(cols)} columns")
    
    def _parse_protein_line(self, line: str) -> Optional[Dict]:
        """
        Parse a single protein line with improved filtering.
        """
        parts = line.split('\t')
        
        if len(parts) < max(self.intensity_columns) + 1:
            return None
        
        # Extract protein information
        protein_info = self._extract_protein_info(line)
        if not protein_info:
            return None
        
        # Extract intensity values for each species
        species_intensities = {}
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            intensities = []
            for col_idx in self.species_column_mapping[species]:
                try:
                    value = float(parts[col_idx]) if parts[col_idx] != '' else 0.0
                    intensities.append(value)
                except (ValueError, IndexError):
                    intensities.append(0.0)
            species_intensities[species] = intensities
        
        # Calculate total intensity across all species
        total_intensity = sum(sum(intensities) for intensities in species_intensities.values())
        
        # Filter out proteins with zero total intensity (undetected proteins)
        if total_intensity == 0:
            return None
        
        # Determine primary species (species with highest total intensity)
        species_totals = {}
        for species, intensities in species_intensities.items():
            species_totals[species] = sum(intensities)
        
        primary_species = max(species_totals, key=species_totals.get)
        
        # Create protein entry
        protein_entry = {
            'Protein': protein_info['id'],
            'Description': protein_info.get('description', ''),
            'Species': primary_species,
            'Gene': protein_info.get('gene', ''),
            'Lb_Intensities': species_intensities['Lb'],
            'Lg_Intensities': species_intensities['Lg'],
            'Ln_Intensities': species_intensities['Ln'],
            'Lp_Intensities': species_intensities['Lp']
        }
        
        return protein_entry
    
    def _extract_protein_info(self, line: str) -> Dict:
        """
        Extract protein information from the line using proper column indices.
        """
        parts = line.split('\t')
        
        # Check if this is a decoy (Reverse) or contaminant
        if len(parts) > 273:  # Make sure we have enough columns
            reverse = parts[272].strip()  # Reverse column (0-indexed: 273-1)
            contaminant = parts[273].strip() if len(parts) > 273 else ''  # Potential contaminant column
            
            # Skip decoys and contaminants
            if reverse == '+' or contaminant == '+':
                return None
        
        # Extract protein IDs from the first column (semicolon-separated)
        protein_ids_col = parts[0].strip() if len(parts) > 0 else ''
        if not protein_ids_col:
            return None
        
        # Split by semicolon and take the first protein ID
        protein_ids = [pid.strip() for pid in protein_ids_col.split(';') if pid.strip()]
        if not protein_ids:
            return None
        
        # Extract description from Fasta headers column (column 6, 0-indexed: 5)
        fasta_headers = parts[5].strip() if len(parts) > 5 else ''

        
        # Extract description from the first fasta header
        description = ''
        if fasta_headers:
            # Split by semicolon to get individual headers
            headers = fasta_headers.split(';')
            if headers:
                first_header = headers[0].strip()
                
                # Format: tr|ID|ID description OS=...
                if '|' in first_header:
                    # Split by | and get the description part
                    header_parts = first_header.split('|')
                    
                    if len(header_parts) >= 3:
                        # The description is in the third part (index 2)
                        description_part = header_parts[2]
                        
                        # Extract description before OS=
                        if 'OS=' in description_part:
                            description = description_part.split('OS=')[0].strip()
                        else:
                            description = description_part.strip()
                        
                        # If still empty, try to extract from the full header
                        if not description:
                            # Look for description between the ID and OS=
                            if 'OS=' in first_header:
                                # Find the position after the third |
                                third_pipe_pos = first_header.find('|', first_header.find('|', first_header.find('|') + 1) + 1)
                                if third_pipe_pos != -1:
                                    os_pos = first_header.find('OS=', third_pipe_pos)
                                    if os_pos != -1:
                                        description = first_header[third_pipe_pos + 1:os_pos].strip()
                        
                        # If still empty, try a different approach - look for the description after the ID
                        if not description and len(header_parts) >= 3:
                            # The format might be tr|ID|ID_description OS=...
                            id_part = header_parts[2]
                            if '_' in id_part:
                                # Split by underscore and take everything after the ID
                                parts = id_part.split('_', 1)
                                if len(parts) > 1:
                                    description = parts[1].split('OS=')[0].strip()
        
        # Extract gene name from fasta headers if present
        gene = ''
        if 'GN=' in fasta_headers:
            gene_match = re.search(r'GN=([^=]+)', fasta_headers)
            gene = gene_match.group(1) if gene_match else ''
        
        return {
            'id': protein_ids[0],  # Use first protein ID
            'description': description,
            'gene': gene
        }
    
    def _calculate_species_expression(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate median expression for each Leishmania species with zero filtering.
        """
        print("Calculating species-specific median expressions (excluding zeros)...")
        
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            intensity_col = f'{species}_Intensities'
            
            # Calculate median expression for proteins of each species
            species_data = df[df['Species'] == species]
            
            if not species_data.empty:
                # Calculate median intensity for each protein (excluding zeros)
                median_expressions = []
                for idx, row in species_data.iterrows():
                    intensities = row[intensity_col]
                    # Filter out zeros and calculate median
                    non_zero_intensities = [val for val in intensities if val > 0]
                    if non_zero_intensities:
                        median_val = np.median(non_zero_intensities)
                    else:
                        median_val = 0
                    median_expressions.append(median_val)
                
                # Apply normalization and log transformation
                # Normalize by dividing by 1000 first, then apply log2
                normalized_expressions = np.array(median_expressions) / 1000.0
                log_expressions = np.log2(normalized_expressions + 1)
                
                df.loc[species_data.index, species] = log_expressions
            else:
                df[species] = 0
        
        return df
    
    def _add_enhanced_pathway_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced pathway annotations using comprehensive keyword matching.
        """
        print("Adding enhanced pathway annotations...")
        
        pathways = []
        annotation_sources = []
        confidence_scores = []
        
        # Debug: Check first few descriptions
        print("Sample descriptions:")
        for i in range(min(5, len(df))):
            print(f"  {i+1}: '{df.iloc[i]['Description']}'")
        
        for idx, row in df.iterrows():
            description = row['Description']
            gene = row['Gene']
            
            pathway_found = False
            annotation_source = 'None'
            confidence_score = 0
            
            # Enhanced keyword matching in description and gene
            search_text = f"{description} {gene}".upper()
            
            for pathway, keywords in self.pathway_keywords.items():
                matches = 0
                for keyword in keywords:
                    if keyword.upper() in search_text:
                        matches += 1
                
                if matches > 0:
                    # Calculate confidence score based on number of matches
                    confidence = min(matches / len(keywords) * 100, 100)
                    
                    if confidence > confidence_score:
                        pathways.append(pathway)
                        annotation_sources.append('Enhanced_Keywords')
                        confidence_scores.append(confidence)
                        pathway_found = True
                        break
            
            # Default to 'Other' if no pathway found
            if not pathway_found:
                pathways.append('Other')
                annotation_sources.append('None')
                confidence_scores.append(0)
        
        df['Pathway'] = pathways
        df['Annotation_Source'] = annotation_sources
        df['Confidence_Score'] = confidence_scores
        
        return df
    
    def create_expression_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a matrix of protein expression across species for heatmap analysis.
        """
        # Select proteins with pathway annotations (excluding 'Other')
        pathway_proteins = df[df['Pathway'] != 'Other'].copy()
        
        if pathway_proteins.empty:
            print("No proteins with pathway annotations found")
            return pd.DataFrame()
        
        print(f"Found {len(pathway_proteins)} proteins with pathway annotations")
        
        # Group by pathway and calculate median expression
        pathway_expression = pathway_proteins.groupby('Pathway')[['Lb', 'Lg', 'Ln', 'Lp']].median()
        
        # Calculate log fold change for all species that have data
        # Find species with non-zero median values
        species_with_data = []
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            if pathway_expression[species].notna().any() and pathway_expression[species].sum() > 0:
                species_with_data.append(species)
        
        if len(species_with_data) < 2:
            print(f"Only {len(species_with_data)} species have data, cannot calculate fold changes")
            return pathway_expression
        
        pathway_expression_with_data = pathway_expression[species_with_data]
        
        # Calculate mean across species with data
        mean_across_species = pathway_expression_with_data.mean(axis=1)
        log_fold_change = pathway_expression_with_data.sub(mean_across_species, axis=0)
        
        # Add back any missing species columns
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            if species not in log_fold_change.columns:
                log_fold_change[species] = np.nan
        
        return log_fold_change
    
    def perform_statistical_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Perform statistical analysis on pathway expression differences.
        """
        print("Performing statistical analysis...")
        
        # Select proteins with pathway annotations
        pathway_proteins = df[df['Pathway'] != 'Other'].copy()
        
        if pathway_proteins.empty:
            return {}
        
        results = {}
        
        # Kruskal-Wallis test for all species
        species_data = {}
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            species_data[species] = pathway_proteins[species].dropna()
        
        if len([data for data in species_data.values() if len(data) > 0]) >= 3:
            all_data = [data for data in species_data.values() if len(data) > 0]
            h_stat, p_value = kruskal(*all_data)
            results['kruskal_wallis'] = {
                'h_statistic': h_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # Mann-Whitney U tests for species pairs
        species_pairs = [('Lb', 'Lg'), ('Lb', 'Ln'), ('Lb', 'Lp'), 
                        ('Lg', 'Ln'), ('Lg', 'Lp'), ('Ln', 'Lp')]
        
        results['mann_whitney'] = {}
        for sp1, sp2 in species_pairs:
            data1 = species_data[sp1]
            data2 = species_data[sp2]
            
            if len(data1) > 0 and len(data2) > 0:
                stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                effect_size = abs(stat - len(data1)*len(data2)/2) / np.sqrt(len(data1)*len(data2)*(len(data1)+len(data2)+1)/12)
                
                results['mann_whitney'][f'{sp1}_vs_{sp2}'] = {
                    'u_statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': effect_size
                }
        
        return results
    
    def create_enhanced_heatmap(self, expression_matrix: pd.DataFrame, output_file: str = 'enhanced_metabolic_heatmap.png'):
        """
        Create an enhanced heatmap with statistical annotations.
        """
        print("Creating enhanced metabolic pathway heatmap...")
        
        # Include all species that have data
        expression_matrix_filtered = expression_matrix.copy()
        
        # Remove rows with all NaN values
        expression_matrix_filtered = expression_matrix_filtered.dropna(how='all')
        
        # Fill remaining NaN values with 0 for visualization
        expression_matrix_filtered = expression_matrix_filtered.fillna(0)
        
        print(f"Creating heatmap with {len(expression_matrix_filtered)} pathways")
        print(f"Species included: {list(expression_matrix_filtered.columns)}")
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Create the main heatmap
        sns.heatmap(expression_matrix_filtered, 
                   cmap='RdBu_r', 
                   center=0,
                   annot=True, 
                   fmt='.2f',
                   cbar_kws={'label': 'Log2 Fold Change', 'shrink': 0.8},
                   ax=ax1,
                   linewidths=0.5,
                   linecolor='white')
        
        # Customize the main heatmap
        ax1.set_title('Metabolic Pathway Expression in Leishmania Species\n' + 
                     'Log2-Transformed Expression Values (Median by Pathway)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Leishmania Species', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Metabolic Pathways', fontsize=14, fontweight='bold')
        
        # Create pathway count bar plot
        pathway_counts = expression_matrix_filtered.index.value_counts()
        pathway_counts.plot(kind='bar', ax=ax2, color='skyblue', alpha=0.7)
        ax2.set_title('Number of Proteins per Pathway', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Pathways', fontsize=12)
        ax2.set_ylabel('Number of Proteins', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        plt.setp(ax2.get_xticklabels(), ha='right')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Enhanced metabolic pathway heatmap saved as {output_file}")
        
        return fig
    
    def create_quality_control_plots(self, df: pd.DataFrame, output_file: str = 'quality_control_plots.png'):
        """
        Create quality control plots for the analysis.
        """
        print("Creating quality control plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Quality Control Plots for Metabolic Pathway Analysis', fontsize=16, fontweight='bold')
        
        # 1. Protein distribution by species
        species_counts = df['Species'].value_counts()
        axes[0,0].pie(species_counts.values, labels=species_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Protein Distribution by Species')
        
        # 2. Pathway distribution
        pathway_counts = df['Pathway'].value_counts().head(15)
        pathway_counts.plot(kind='bar', ax=axes[0,1], color='lightcoral')
        axes[0,1].set_title('Top 15 Pathways by Protein Count')
        axes[0,1].tick_params(axis='x', rotation=45)
        plt.setp(axes[0,1].get_xticklabels(), ha='right')
        
        # 3. Confidence score distribution
        if 'Confidence_Score' in df.columns:
            axes[1,0].hist(df['Confidence_Score'], bins=20, alpha=0.7, color='lightgreen')
            axes[1,0].set_title('Pathway Annotation Confidence Scores')
            axes[1,0].set_xlabel('Confidence Score')
            axes[1,0].set_ylabel('Number of Proteins')
        
        # 4. Expression distribution by species
        species_data = []
        species_labels = []
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            if species in df.columns:
                data = df[species].dropna()
                if len(data) > 0:
                    species_data.append(data)
                    species_labels.append(species)
        
        if species_data:
            axes[1,1].boxplot(species_data, labels=species_labels)
            axes[1,1].set_title('Expression Distribution by Species')
            axes[1,1].set_ylabel('Log2 Expression')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Quality control plots saved as {output_file}")
        
        return fig
    
    def save_processed_data(self, df: pd.DataFrame, output_file: str):
        """
        Save processed protein data to CSV file with enhanced information.
        """
        # Remove intensity columns for CSV output to keep it clean
        intensity_cols = ['Lb_Intensities', 'Lg_Intensities', 'Ln_Intensities', 'Lp_Intensities']
        df_csv = df.drop(intensity_cols, axis=1, errors='ignore')
        df_csv.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
        
        # Print comprehensive summary
        print(f"\n" + "="*60)
        print(f"REFINED METABOLIC PATHWAY ANALYSIS SUMMARY")
        print(f"="*60)
        print(f"Total proteins: {len(df)}")
        
        print(f"\nProteins by species:")
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            count = len(df[df['Species'] == species])
            print(f"  {species}: {count}")
        
        print(f"\nProteins by pathway:")
        if 'Pathway' in df.columns:
            pathway_counts = df['Pathway'].value_counts()
            for pathway, count in pathway_counts.items():
                print(f"  {pathway}: {count}")
        
        print(f"\nAnnotation sources:")
        if 'Annotation_Source' in df.columns:
            annotation_counts = df['Annotation_Source'].value_counts()
            for source, count in annotation_counts.items():
                print(f"  {source}: {count}")
        
        if 'Confidence_Score' in df.columns:
            print(f"\nConfidence score statistics:")
            confidence_stats = df['Confidence_Score'].describe()
            print(f"  Mean: {confidence_stats['mean']:.2f}")
            print(f"  Median: {confidence_stats['50%']:.2f}")
            print(f"  Std: {confidence_stats['std']:.2f}")
    
    def generate_comprehensive_report(self, df: pd.DataFrame, expression_matrix: pd.DataFrame, 
                                    statistical_results: Dict, output_file: str):
        """
        Generate a comprehensive analysis report with statistical results.
        """
        print("Generating comprehensive analysis report...")
        
        with open(output_file, 'w') as f:
            f.write("REFINED METABOLIC PATHWAY ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("DATA SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total proteins analyzed: {len(df)}\n")
            f.write(f"Pathways identified: {len(df['Pathway'].unique())}\n")
            f.write(f"Species analyzed: {len(df['Species'].unique())}\n\n")
            
            f.write("PROTEIN DISTRIBUTION BY SPECIES:\n")
            f.write("-" * 35 + "\n")
            for species in ['Lb', 'Lg', 'Ln', 'Lp']:
                count = len(df[df['Species'] == species])
                f.write(f"{species}: {count} proteins\n")
            f.write("\n")
            
            f.write("PATHWAY DISTRIBUTION:\n")
            f.write("-" * 20 + "\n")
            pathway_counts = df['Pathway'].value_counts()
            for pathway, count in pathway_counts.items():
                f.write(f"{pathway}: {count} proteins\n")
            f.write("\n")
            
            f.write("STATISTICAL ANALYSIS RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            if 'kruskal_wallis' in statistical_results:
                kw_result = statistical_results['kruskal_wallis']
                f.write(f"Kruskal-Wallis Test (All Species):\n")
                f.write(f"  H-statistic: {kw_result['h_statistic']:.4f}\n")
                f.write(f"  p-value: {kw_result['p_value']:.4e}\n")
                f.write(f"  Significant difference: {'Yes' if kw_result['significant'] else 'No'}\n\n")
            
            if 'mann_whitney' in statistical_results:
                f.write("Mann-Whitney U Tests (Species Pairs):\n")
                for pair, result in statistical_results['mann_whitney'].items():
                    f.write(f"  {pair}:\n")
                    f.write(f"    U-statistic: {result['u_statistic']:.4f}\n")
                    f.write(f"    p-value: {result['p_value']:.4e}\n")
                    f.write(f"    Significant difference: {'Yes' if result['significant'] else 'No'}\n")
                    f.write(f"    Effect size: {result['effect_size']:.3f}\n\n")
            
            f.write("EXPRESSION ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write("Pathway expression matrix:\n")
            f.write(str(expression_matrix) + "\n\n")
            
            f.write("EXPRESSION STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(str(expression_matrix.describe()) + "\n\n")
        
        print(f"Comprehensive analysis report saved as {output_file}")

def main():
    """
    Main function to run refined metabolic pathway analysis.
    """
    analyzer = RefinedMetabolicPathwayAnalyzer()
    
    # Step 1: Parse and process data
    print("Step 1: Parsing and processing data...")
    df = analyzer.parse_raw_data('../raw_data.txt')
    
    if df.empty:
        print("No data found. Exiting.")
        return
    
    # Process expression data with zero filtering
    df = analyzer._calculate_species_expression(df)
    
    # Save processed data
    analyzer.save_processed_data(df, 'refined_processed_protein_data.csv')
    
    # Create expression matrix
    print("\nStep 2: Creating expression matrix...")
    expression_matrix = analyzer.create_expression_matrix(df)
    
    if not expression_matrix.empty:
        # Save expression matrix
        expression_matrix.to_csv('refined_pathway_expression_matrix.csv')
        print("Expression matrix saved to 'refined_pathway_expression_matrix.csv'")
        
        # Perform statistical analysis
        print("\nStep 3: Performing statistical analysis...")
        statistical_results = analyzer.perform_statistical_analysis(df)
        
        # Generate visualizations
        print("\nStep 4: Generating visualizations...")
        analyzer.create_enhanced_heatmap(expression_matrix, 'refined_metabolic_heatmap.png')
        analyzer.create_quality_control_plots(df, 'quality_control_plots.png')
        analyzer.generate_comprehensive_report(df, expression_matrix, statistical_results, 'refined_metabolic_analysis_report.txt')
        
        print("\n" + "="*60)
        print("REFINED ANALYSIS COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("- refined_processed_protein_data.csv")
        print("- refined_pathway_expression_matrix.csv") 
        print("- refined_metabolic_heatmap.png")
        print("- quality_control_plots.png")
        print("- refined_metabolic_analysis_report.txt")
        
        print(f"\nExpression Matrix:")
        print(expression_matrix)
    else:
        print("No pathways found for analysis.")

if __name__ == "__main__":
    main()
