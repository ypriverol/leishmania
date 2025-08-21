import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class MetabolicPathwayAnalyzer:
    def __init__(self):
        # Enhanced pathway keywords for annotation
        self.pathway_keywords = {
            'Oxidative Phosphorylation': ['ATP', 'cytochrome', 'oxidase', 'dehydrogenase', 'NADH', 'FADH', 'electron transport', 'respiratory chain', 'ATP synthase', 'complex I', 'complex II', 'complex III', 'complex IV', 'complex V', 'ubiquinone', 'coenzyme Q', 'succinate', 'malate', 'fumarate', 'isocitrate', 'alpha-ketoglutarate'],
            'Glycolysis': ['glucose', 'hexokinase', 'phosphofructokinase', 'pyruvate kinase', 'glyceraldehyde', 'phosphoglycerate', 'enolase', 'lactate dehydrogenase', 'aldolase', 'triose phosphate', 'fructose', 'glucose-6-phosphate', 'fructose-6-phosphate'],
            'TCA Cycle': ['citrate synthase', 'aconitase', 'isocitrate dehydrogenase', 'alpha-ketoglutarate dehydrogenase', 'succinyl-CoA', 'succinate dehydrogenase', 'fumarase', 'malate dehydrogenase', 'citric acid cycle', 'krebs cycle', 'tricarboxylic acid'],
            'Pentose Phosphate Pathway': ['glucose-6-phosphate dehydrogenase', '6-phosphogluconate dehydrogenase', 'ribulose', 'ribose', 'pentose', 'transketolase', 'transaldolase', 'phosphogluconate'],
            'Amino Acid Metabolism': ['alanine', 'aspartate', 'glutamate', 'glutamine', 'arginine', 'lysine', 'histidine', 'phenylalanine', 'tyrosine', 'tryptophan', 'leucine', 'isoleucine', 'valine', 'methionine', 'cysteine', 'serine', 'threonine', 'asparagine', 'proline', 'glycine', 'aminotransferase', 'transaminase', 'dehydrogenase'],
            'Oxidative Stress': ['catalase', 'superoxide dismutase', 'peroxiredoxin', 'thioredoxin', 'glutaredoxin', 'glutathione', 'peroxidase', 'reductase', 'oxidase', 'reactive oxygen', 'ROS', 'antioxidant', 'redox'],
            'Signal Transduction': ['kinase', 'phosphatase', 'receptor', 'G protein', 'cAMP', 'cGMP', 'calcium', 'calmodulin', 'protein kinase', 'tyrosine kinase', 'serine kinase', 'threonine kinase', 'MAP kinase', 'JNK', 'ERK', 'p38', 'AKT', 'PI3K', 'phospholipase', 'adenylate cyclase', 'guanylate cyclase'],
            'Membrane Proteins': ['transporter', 'channel', 'pump', 'ATPase', 'sodium', 'potassium', 'calcium', 'chloride', 'membrane', 'integral membrane', 'transmembrane', 'receptor', 'adhesion', 'cell adhesion'],
            'Proteasome': ['proteasome', 'ubiquitin', 'ubiquitination', 'protease', 'peptidase', '20S', '26S', '19S', 'regulatory particle', 'core particle'],
            'Ribosomal Proteins': ['ribosomal protein', 'ribosome', 'rRNA', 'translation', 'elongation factor', 'initiation factor', 'release factor', 'ribosomal RNA'],
            'Translation Factors': ['elongation factor', 'initiation factor', 'release factor', 'translation factor', 'EF-', 'IF-', 'RF-', 'eIF', 'eEF', 'eRF'],
            'DNA Replication': ['DNA polymerase', 'helicase', 'primase', 'ligase', 'topoisomerase', 'gyrase', 'replication', 'origin', 'replicon', 'replisome'],
            'Transcription': ['RNA polymerase', 'transcription factor', 'promoter', 'enhancer', 'silencer', 'transcription', 'transcriptional', 'RNA synthesis'],
            'Nuclear Proteins': ['nuclear', 'nucleus', 'chromatin', 'histone', 'nucleosome', 'nuclear pore', 'nuclear envelope', 'nuclear matrix'],
            'Cytoskeletal Proteins': ['actin', 'tubulin', 'microtubule', 'microfilament', 'intermediate filament', 'cytoskeleton', 'myosin', 'dynein', 'kinesin', 'motor protein'],
            'Heat Shock Proteins': ['heat shock', 'HSP', 'chaperone', 'chaperonin', 'GroEL', 'GroES', 'DnaK', 'DnaJ', 'GrpE', 'Hsp70', 'Hsp90', 'Hsp60'],
            'Histones': ['histone', 'H1', 'H2A', 'H2B', 'H3', 'H4', 'nucleosome', 'chromatin'],
            'Metabolism': ['metabolism', 'metabolic', 'biosynthesis', 'catabolism', 'anabolism', 'synthesis', 'degradation', 'breakdown'],
            'Autophagy': ['autophagy', 'autophagosome', 'lysosome', 'vacuole', 'ATG', 'autophagic', 'macroautophagy', 'microautophagy'],
            'Cell Cycle': ['cyclin', 'CDK', 'cell cycle', 'mitosis', 'meiosis', 'spindle', 'centrosome', 'centromere', 'telomere', 'checkpoint'],
            'Glycosomal Proteins': ['glycosome', 'glycosomal', 'peroxisome', 'peroxisomal', 'glycolytic enzyme', 'glycosomal enzyme']
        }
        
        # Initialize column mappings
        self.intensity_columns = []
        self.species_column_mapping = {'Lb': [], 'Lg': [], 'Ln': [], 'Lp': []}
        
    def parse_raw_data(self, filename: str) -> pd.DataFrame:
        """
        Parse raw data with proper species identification based on intensity columns.
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
        Parse a single protein line.
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
        
        # Determine primary species (species with highest total intensity)
        species_totals = {}
        for species, intensities in species_intensities.items():
            species_totals[species] = sum(intensities)
        
        primary_species = max(species_totals, key=species_totals.get) if max(species_totals.values()) > 0 else 'Unknown'
        
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
        
        # Extract description from Fasta headers column (column 6, 0-indexed: 6)
        fasta_headers = parts[6].strip() if len(parts) > 6 else ''
        
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
                        # The description is after the third |
                        description_part = header_parts[2]
                        # Extract description before OS=
                        if 'OS=' in description_part:
                            description = description_part.split('OS=')[0].strip()
                        else:
                            description = description_part.strip()
                        
                        # If description is empty, try to extract from the full header
                        if not description:
                            # Look for description between the ID and OS=
                            if 'OS=' in first_header:
                                # Find the position after the third |
                                third_pipe_pos = first_header.find('|', first_header.find('|', first_header.find('|') + 1) + 1)
                                if third_pipe_pos != -1:
                                    os_pos = first_header.find('OS=', third_pipe_pos)
                                    if os_pos != -1:
                                        description = first_header[third_pipe_pos + 1:os_pos].strip()
        
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
    
    def _annotate_pathway(self, description: str) -> str:
        """
        Annotate pathway based on protein description using enhanced keywords.
        """
        if not description:
            return 'Other'
        
        description_lower = description.lower()
        
        for pathway, keywords in self.pathway_keywords.items():
            for keyword in keywords:
                if keyword.lower() in description_lower:
                    return pathway
        
        return 'Other'
    
    def _calculate_species_expression(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate average expression for each Leishmania species with improved normalization.
        """
        for species in ['Lb', 'Lg', 'Ln', 'Lp']:
            intensity_col = f'{species}_Intensities'
            
            # Calculate mean expression for proteins of each species
            species_data = df[df['Species'] == species]
            
            if not species_data.empty:
                # Calculate mean intensity for each protein
                mean_expressions = species_data[intensity_col].apply(lambda x: np.mean(x) if x and any(val > 0 for val in x) else 0)
                
                # Apply normalization and log transformation
                # Normalize by dividing by 1000 first, then apply log2
                normalized_expressions = mean_expressions / 1000.0
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
        
        for idx, row in df.iterrows():
            description = row['Description']
            gene = row['Gene']
            
            pathway_found = False
            annotation_source = 'None'
            
            # Enhanced keyword matching in description and gene
            search_text = f"{description} {gene}".upper()
            
            for pathway, keywords in self.pathway_keywords.items():
                for keyword in keywords:
                    if keyword.upper() in search_text:
                        pathways.append(pathway)
                        annotation_sources.append('Enhanced_Keywords')
                        pathway_found = True
                        break
                if pathway_found:
                    break
            
            # Default to 'Other' if no pathway found
            if not pathway_found:
                pathways.append('Other')
                annotation_sources.append('None')
        
        df['Pathway'] = pathways
        df['Annotation_Source'] = annotation_sources
        
        return df
    
    def create_expression_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a matrix of protein expression across species for heatmap analysis.
        """
        # Select proteins with pathway annotations
        pathway_proteins = df[df['Pathway'] != 'Other'].copy()
        
        if pathway_proteins.empty:
            print("No proteins with pathway annotations found")
            return pd.DataFrame()
        
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
    
    def save_processed_data(self, df: pd.DataFrame, output_file: str):
        """
        Save processed protein data to CSV file.
        """
        # Remove intensity columns for CSV output to keep it clean
        intensity_cols = ['Lb_Intensities', 'Lg_Intensities', 'Ln_Intensities', 'Lp_Intensities']
        df_csv = df.drop(intensity_cols, axis=1, errors='ignore')
        df_csv.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
        
        # Print summary
        print(f"\nData Summary:")
        print(f"Total proteins: {len(df)}")
        print(f"Proteins by species:")
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
    
    def create_metabolic_heatmap(self, expression_matrix, output_file='metabolic_heatmap.png'):
        """
        Create a clean heatmap of metabolic pathways.
        """
        print("Creating metabolic pathway heatmap...")
        
        # Include all species that have data
        expression_matrix_filtered = expression_matrix.copy()
        
        # Remove rows with all NaN values
        expression_matrix_filtered = expression_matrix_filtered.dropna(how='all')
        
        # Fill remaining NaN values with 0 for visualization
        expression_matrix_filtered = expression_matrix_filtered.fillna(0)
        
        print(f"Creating heatmap with {len(expression_matrix_filtered)} pathways")
        print(f"Species included: {list(expression_matrix_filtered.columns)}")
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create the heatmap
        sns.heatmap(expression_matrix_filtered, 
                   cmap='RdBu_r', 
                   center=0,
                   annot=True, 
                   fmt='.2f',
                   cbar_kws={'label': 'Log2 Fold Change', 'shrink': 0.8},
                   ax=ax,
                   linewidths=0.5,
                   linecolor='white')
        
        # Customize the plot
        ax.set_title('Metabolic Pathway Expression in Leishmania Species\n' + 
                    'Log2-Transformed Expression Values (Median by Pathway)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Leishmania Species', fontsize=14, fontweight='bold')
        ax.set_ylabel('Metabolic Pathways', fontsize=14, fontweight='bold')
        
        # Customize tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Metabolic pathway heatmap saved as {output_file}")
        
        return fig
    
    def create_expression_bar_plot(self, expression_matrix, output_file='expression_bar_plot.png'):
        """
        Create a bar plot showing expression levels for each pathway and species.
        """
        print("Creating expression bar plot...")
        
        # Prepare data for bar plot
        data = expression_matrix.reset_index()
        data_melted = data.melt(id_vars=['Pathway'], 
                               var_name='Species', 
                               value_name='Expression')
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create grouped bar plot using seaborn
        sns.barplot(data=data_melted, x='Pathway', y='Expression', hue='Species', ax=ax, alpha=0.8)
        
        # Customize the plot
        ax.set_title('Expression Levels by Pathway and Species\n' + 
                    'Log2-Transformed Values', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Metabolic Pathways', fontsize=14, fontweight='bold')
        ax.set_ylabel('Log2 Expression', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        ax.legend(title='Species', title_fontsize=12, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Expression bar plot saved as {output_file}")
        
        return fig
    
    def generate_analysis_report(self, df: pd.DataFrame, expression_matrix: pd.DataFrame, output_file: str):
        """
        Generate a comprehensive analysis report.
        """
        print("Generating analysis report...")
        
        with open(output_file, 'w') as f:
            f.write("METABOLIC PATHWAY ANALYSIS REPORT\n")
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
            
            f.write("EXPRESSION ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write("Pathway expression matrix:\n")
            f.write(str(expression_matrix) + "\n\n")
            
            f.write("EXPRESSION STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(str(expression_matrix.describe()) + "\n\n")
        
        print(f"Analysis report saved as {output_file}")

def main():
    """
    Main function to run metabolic pathway analysis.
    """
    analyzer = MetabolicPathwayAnalyzer()
    
    # Step 1: Parse and process data
    print("Step 1: Parsing and processing data...")
    df = analyzer.parse_raw_data('../raw_data.txt')
    
    if df.empty:
        print("No data found. Exiting.")
        return
    
    # Process expression data
    df = analyzer._calculate_species_expression(df)
    
    # Save processed data
    analyzer.save_processed_data(df, 'processed_protein_data.csv')
    
    # Create expression matrix
    print("\nStep 2: Creating expression matrix...")
    expression_matrix = analyzer.create_expression_matrix(df)
    
    if not expression_matrix.empty:
        # Save expression matrix
        expression_matrix.to_csv('pathway_expression_matrix.csv')
        print("Expression matrix saved to 'pathway_expression_matrix.csv'")
        
        # Generate visualizations
        print("\nStep 3: Generating visualizations...")
        analyzer.create_metabolic_heatmap(expression_matrix, 'metabolic_heatmap.png')
        analyzer.create_expression_bar_plot(expression_matrix, 'expression_bar_plot.png')
        analyzer.generate_analysis_report(df, expression_matrix, 'metabolic_analysis_report.txt')
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("- processed_protein_data.csv")
        print("- pathway_expression_matrix.csv") 
        print("- metabolic_heatmap.png")
        print("- expression_bar_plot.png")
        print("- metabolic_analysis_report.txt")
        
        print(f"\nExpression Matrix:")
        print(expression_matrix)
    else:
        print("No pathways found for analysis.")

if __name__ == "__main__":
    main()
