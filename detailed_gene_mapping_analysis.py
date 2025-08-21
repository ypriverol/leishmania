#!/usr/bin/env python3
"""
Detailed Gene Mapping Analysis
==============================

This script shows exactly how protein groups are mapped to genes and what happens
when multiple protein IDs map to the same gene.

Author: Assistant
Date: 2024
"""

import pandas as pd
import re
from collections import defaultdict

def extract_gene_from_fasta_header(fasta_header):
    """Extract gene names from Fasta header."""
    genes = []
    if 'GN=' in fasta_header:
        gene_matches = re.findall(r'GN=([^=]+)', fasta_header)
        genes.extend(gene_matches)
    return genes

def analyze_detailed_gene_mapping():
    """Analyze detailed gene mapping for protein groups."""
    
    print("🔍 Detailed Gene Mapping Analysis")
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
    multi_id_proteins = df[multi_id_mask].copy()
    
    print(f"\n📊 ANALYZING {len(multi_id_proteins)} PROTEIN GROUPS")
    print("=" * 60)
    
    # Detailed analysis of gene mapping
    mapping_examples = []
    gene_to_protein_groups = defaultdict(list)
    
    for idx, row in multi_id_proteins.iterrows():
        protein_ids = row["Protein IDs"]
        fasta_header = row["Fasta headers"]
        
        # Split protein IDs
        protein_id_list = protein_ids.split(';')
        
        # Extract genes from fasta header
        genes = extract_gene_from_fasta_header(fasta_header)
        unique_genes = list(set(genes))  # Remove duplicates
        
        # Store mapping information
        mapping_info = {
            'Protein_Group': protein_ids,
            'Protein_ID_Count': len(protein_id_list),
            'Protein_IDs': protein_id_list,
            'All_Genes': genes,
            'Unique_Genes': unique_genes,
            'Unique_Gene_Count': len(unique_genes),
            'Maps_To_Single_Gene': len(unique_genes) == 1,
            'Maps_To_Multiple_Genes': len(unique_genes) > 1,
            'No_Gene_Info': len(unique_genes) == 0
        }
        
        mapping_examples.append(mapping_info)
        
        # Group by unique genes
        for gene in unique_genes:
            gene_to_protein_groups[gene].append(protein_ids)
    
    # Create results dataframe
    results_df = pd.DataFrame(mapping_examples)
    
    # Show detailed examples
    print(f"\n📋 DETAILED MAPPING EXAMPLES:")
    print("=" * 60)
    
    # Example 1: Single gene mapping
    single_gene_examples = results_df[results_df['Maps_To_Single_Gene']].head(3)
    print(f"\n🔸 SINGLE GENE MAPPING (Included in analysis):")
    for _, row in single_gene_examples.iterrows():
        print(f"Protein Group: {row['Protein_Group']}")
        print(f"  Protein IDs: {row['Protein_IDs']}")
        print(f"  All Genes: {row['All_Genes']}")
        print(f"  Unique Gene: {row['Unique_Genes'][0]}")
        print(f"  → INCLUDED (single gene)")
        print()
    
    # Example 2: Multiple gene mapping
    multiple_gene_examples = results_df[results_df['Maps_To_Multiple_Genes']].head(3)
    print(f"\n🔸 MULTIPLE GENE MAPPING (Excluded from analysis):")
    for _, row in multiple_gene_examples.iterrows():
        print(f"Protein Group: {row['Protein_Group']}")
        print(f"  Protein IDs: {row['Protein_IDs']}")
        print(f"  All Genes: {row['All_Genes']}")
        print(f"  Unique Genes: {row['Unique_Genes']}")
        print(f"  → EXCLUDED (multiple genes)")
        print()
    
    # Example 3: No gene info
    no_gene_examples = results_df[results_df['No_Gene_Info']].head(2)
    if len(no_gene_examples) > 0:
        print(f"\n🔸 NO GENE INFO (Excluded from analysis):")
        for _, row in no_gene_examples.iterrows():
            print(f"Protein Group: {row['Protein_Group']}")
            print(f"  Protein IDs: {row['Protein_IDs']}")
            print(f"  All Genes: {row['All_Genes']}")
            print(f"  → EXCLUDED (no gene info)")
            print()
    
    # Show gene sharing patterns
    print(f"\n🧬 GENE SHARING PATTERNS:")
    print("=" * 60)
    
    # Find genes that appear in multiple protein groups
    shared_genes = {gene: protein_groups for gene, protein_groups in gene_to_protein_groups.items() 
                   if len(protein_groups) > 1}
    
    print(f"Genes appearing in multiple protein groups: {len(shared_genes)}")
    
    # Show top shared genes
    print(f"\n🏆 TOP SHARED GENES:")
    sorted_shared_genes = sorted(shared_genes.items(), key=lambda x: len(x[1]), reverse=True)
    for gene, protein_groups in sorted_shared_genes[:5]:
        print(f"Gene: {gene}")
        print(f"  Appears in {len(protein_groups)} protein groups:")
        for group in protein_groups[:3]:  # Show first 3
            print(f"    - {group}")
        if len(protein_groups) > 3:
            print(f"    ... and {len(protein_groups)-3} more")
        print()
    
    # Current logic explanation
    print(f"\n🔧 CURRENT MAPPING LOGIC:")
    print("=" * 60)
    print("1. Extract ALL genes from Fasta header (GN= patterns)")
    print("2. Remove duplicate genes (same gene appears multiple times)")
    print("3. Count unique genes for the protein group")
    print("4. DECISION:")
    print("   - If unique_gene_count == 1: INCLUDE in analysis")
    print("   - If unique_gene_count > 1: EXCLUDE from analysis")
    print("   - If unique_gene_count == 0: EXCLUDE from analysis")
    print()
    
    # Show what happens with intensity data
    print(f"\n📈 INTENSITY HANDLING:")
    print("=" * 60)
    print("When a protein group is INCLUDED:")
    print("- The ENTIRE protein group is kept as one unit")
    print("- Intensity values are used as-is (no aggregation)")
    print("- Each protein ID in the group contributes to the same gene")
    print()
    print("When a protein group is EXCLUDED:")
    print("- The ENTIRE protein group is removed")
    print("- No intensity data is used from any protein ID in the group")
    print()
    
    # Statistics
    print(f"\n📊 FINAL STATISTICS:")
    print("=" * 60)
    single_gene_count = results_df['Maps_To_Single_Gene'].sum()
    multiple_gene_count = results_df['Maps_To_Multiple_Genes'].sum()
    no_gene_count = results_df['No_Gene_Info'].sum()
    
    print(f"Total protein groups: {len(results_df)}")
    print(f"Included (single gene): {single_gene_count} ({single_gene_count/len(results_df)*100:.1f}%)")
    print(f"Excluded (multiple genes): {multiple_gene_count} ({multiple_gene_count/len(results_df)*100:.1f}%)")
    print(f"Excluded (no gene info): {no_gene_count} ({no_gene_count/len(results_df)*100:.1f}%)")
    
    # Save detailed results
    print(f"\n💾 SAVING DETAILED RESULTS...")
    results_df.to_csv('detailed_gene_mapping_analysis.csv', index=False)
    print(f"Detailed analysis saved to: detailed_gene_mapping_analysis.csv")
    
    return results_df

if __name__ == "__main__":
    results = analyze_detailed_gene_mapping()
