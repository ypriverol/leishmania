#!/usr/bin/env python3
"""
Protein Group to Gene Mapping Analysis
======================================

This script analyzes how many protein groups (multi-ID proteins) map to unique genes.

Author: Assistant
Date: 2024
"""

import pandas as pd
import re
from collections import defaultdict, Counter

def extract_gene_from_fasta_header(fasta_header):
    """Extract gene names from Fasta header."""
    genes = []
    if 'GN=' in fasta_header:
        # Find all GN= patterns in the header
        gene_matches = re.findall(r'GN=([^=]+)', fasta_header)
        genes.extend(gene_matches)
    return genes

def analyze_protein_group_gene_mapping():
    """Analyze protein groups and their gene mappings."""
    
    print("🔍 Analyzing Protein Groups to Gene Mapping")
    print("=" * 60)
    
    # Read the raw data
    print("Loading raw data...")
    df = pd.read_csv('raw_data.txt', sep='\t', low_memory=False)
    
    # Filter out decoys and contaminants
    print("Filtering out decoys and contaminants...")
    if "Reverse" in df.columns and "Potential contaminant" in df.columns:
        df = df[(df["Reverse"] != '+') & (df["Potential contaminant"] != '+')]
        print(f"Proteins after filtering decoys/contaminants: {len(df)}")
    
    # Identify multi-ID proteins (protein groups)
    multi_id_mask = df["Protein IDs"].str.contains(";", na=False)
    protein_groups = df[multi_id_mask].copy()
    
    print(f"\n📊 PROTEIN GROUPS ANALYSIS:")
    print(f"Total protein groups (multi-ID): {len(protein_groups)}")
    
    # Analyze gene mapping for each protein group
    print(f"\n🧬 GENE MAPPING ANALYSIS:")
    
    gene_mapping_results = []
    unique_genes_per_group = []
    all_unique_genes = set()
    
    for idx, row in protein_groups.iterrows():
        protein_ids = row["Protein IDs"]
        fasta_header = row["Fasta headers"]
        
        # Count protein IDs in this group
        protein_id_count = len(protein_ids.split(';'))
        
        # Extract genes
        genes = extract_gene_from_fasta_header(fasta_header)
        unique_genes = list(set(genes))  # Remove duplicates
        unique_gene_count = len(unique_genes)
        
        # Store results
        gene_mapping_results.append({
            'Protein_Group': protein_ids,
            'Protein_ID_Count': protein_id_count,
            'Gene_Count': len(genes),
            'Unique_Gene_Count': unique_gene_count,
            'Genes': ';'.join(unique_genes) if unique_genes else 'None',
            'Maps_To_Single_Gene': unique_gene_count == 1,
            'Maps_To_Multiple_Genes': unique_gene_count > 1,
            'No_Gene_Info': unique_gene_count == 0
        })
        
        unique_genes_per_group.append(unique_gene_count)
        all_unique_genes.update(unique_genes)
    
    # Create results dataframe
    results_df = pd.DataFrame(gene_mapping_results)
    
    # Calculate statistics
    total_protein_groups = len(results_df)
    single_gene_groups = results_df['Maps_To_Single_Gene'].sum()
    multiple_gene_groups = results_df['Maps_To_Multiple_Genes'].sum()
    no_gene_groups = results_df['No_Gene_Info'].sum()
    
    print(f"📈 MAPPING STATISTICS:")
    print(f"Total protein groups: {total_protein_groups}")
    print(f"Groups mapping to single gene: {single_gene_groups} ({single_gene_groups/total_protein_groups*100:.1f}%)")
    print(f"Groups mapping to multiple genes: {multiple_gene_groups} ({multiple_gene_groups/total_protein_groups*100:.1f}%)")
    print(f"Groups with no gene info: {no_gene_groups} ({no_gene_groups/total_protein_groups*100:.1f}%)")
    print(f"Total unique genes across all groups: {len(all_unique_genes)}")
    
    # Analyze protein ID distribution
    protein_id_counts = results_df['Protein_ID_Count'].value_counts().sort_index()
    print(f"\n📋 PROTEIN ID DISTRIBUTION PER GROUP:")
    for count, freq in protein_id_counts.items():
        print(f"  {count} protein IDs: {freq} groups ({freq/total_protein_groups*100:.1f}%)")
    
    # Analyze gene count distribution
    gene_count_distribution = results_df['Unique_Gene_Count'].value_counts().sort_index()
    print(f"\n🧬 GENE COUNT DISTRIBUTION PER GROUP:")
    for count, freq in gene_count_distribution.items():
        print(f"  {count} unique genes: {freq} groups ({freq/total_protein_groups*100:.1f}%)")
    
    # Show examples of different mapping types
    print(f"\n📋 EXAMPLES:")
    
    # Single gene examples
    single_gene_examples = results_df[results_df['Maps_To_Single_Gene']].head(3)
    print(f"🔸 GROUPS MAPPING TO SINGLE GENE:")
    for _, row in single_gene_examples.iterrows():
        print(f"  Proteins: {row['Protein_Group']}")
        print(f"  Gene: {row['Genes']}")
        print()
    
    # Multiple gene examples
    multiple_gene_examples = results_df[results_df['Maps_To_Multiple_Genes']].head(3)
    print(f"🔸 GROUPS MAPPING TO MULTIPLE GENES:")
    for _, row in multiple_gene_examples.iterrows():
        print(f"  Proteins: {row['Protein_Group']}")
        print(f"  Genes: {row['Genes']}")
        print()
    
    # No gene info examples
    no_gene_examples = results_df[results_df['No_Gene_Info']].head(3)
    if len(no_gene_examples) > 0:
        print(f"🔸 GROUPS WITH NO GENE INFO:")
        for _, row in no_gene_examples.iterrows():
            print(f"  Proteins: {row['Protein_Group']}")
            print()
    
    # Calculate potential aggregation benefits
    print(f"\n💡 AGGREGATION ANALYSIS:")
    
    # For single gene groups, we can aggregate protein intensities
    single_gene_aggregatable = single_gene_groups
    print(f"Groups that can be aggregated by gene: {single_gene_aggregatable}")
    
    # For multiple gene groups, we need to keep separate or choose representative
    multiple_gene_keep_separate = multiple_gene_groups
    print(f"Groups that need separate handling: {multiple_gene_keep_separate}")
    
    # Calculate potential reduction
    potential_reduction = single_gene_aggregatable
    print(f"Potential reduction through gene aggregation: {potential_reduction} groups")
    
    # Save detailed results
    print(f"\n💾 SAVING DETAILED RESULTS...")
    results_df.to_csv('protein_group_gene_mapping.csv', index=False)
    print(f"Detailed results saved to: protein_group_gene_mapping.csv")
    
    # Create summary statistics
    summary_stats = {
        'Total_Protein_Groups': total_protein_groups,
        'Single_Gene_Groups': single_gene_groups,
        'Multiple_Gene_Groups': multiple_gene_groups,
        'No_Gene_Groups': no_gene_groups,
        'Single_Gene_Percentage': single_gene_groups/total_protein_groups*100,
        'Multiple_Gene_Percentage': multiple_gene_groups/total_protein_groups*100,
        'No_Gene_Percentage': no_gene_groups/total_protein_groups*100,
        'Total_Unique_Genes': len(all_unique_genes),
        'Aggregatable_Groups': single_gene_aggregatable,
        'Potential_Reduction': potential_reduction
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv('protein_group_gene_mapping_summary.csv', index=False)
    print(f"Summary statistics saved to: protein_group_gene_mapping_summary.csv")
    
    return results_df, summary_df

if __name__ == "__main__":
    results, summary = analyze_protein_group_gene_mapping()
