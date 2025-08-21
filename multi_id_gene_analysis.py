#!/usr/bin/env python3
"""
Multi-ID Protein Gene Analysis
==============================

This script analyzes proteins with multiple IDs (semicolon-separated) to determine:
1. How many unique genes they represent
2. Whether they should be included in phylogenetic analysis
3. The biological significance of multi-ID proteins

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

def analyze_multi_id_proteins():
    """Analyze multi-ID proteins and their gene mappings."""
    
    print("🔍 Analyzing Multi-ID Proteins and Gene Mappings")
    print("=" * 60)
    
    # Read the raw data
    print("Loading raw data...")
    df = pd.read_csv('raw_data.txt', sep='\t', low_memory=False)
    
    # Filter out decoys and contaminants
    print("Filtering out decoys and contaminants...")
    if "Reverse" in df.columns and "Potential contaminant" in df.columns:
        df = df[(df["Reverse"] != '+') & (df["Potential contaminant"] != '+')]
        print(f"Proteins after filtering decoys/contaminants: {len(df)}")
    
    # Identify multi-ID proteins
    multi_id_mask = df["Protein IDs"].str.contains(";", na=False)
    multi_id_proteins = df[multi_id_mask].copy()
    single_id_proteins = df[~multi_id_mask].copy()
    
    print(f"\n📊 PROTEIN DISTRIBUTION:")
    print(f"Total proteins: {len(df)}")
    print(f"Single-ID proteins: {len(single_id_proteins)}")
    print(f"Multi-ID proteins: {len(multi_id_proteins)}")
    
    # Analyze gene information in multi-ID proteins
    print(f"\n🧬 GENE ANALYSIS FOR MULTI-ID PROTEINS:")
    
    gene_counts = []
    unique_genes_per_protein = []
    all_genes = []
    
    for idx, row in multi_id_proteins.iterrows():
        fasta_header = row["Fasta headers"]
        genes = extract_gene_from_fasta_header(fasta_header)
        
        if genes:
            gene_counts.append(len(genes))
            unique_genes = list(set(genes))  # Remove duplicates
            unique_genes_per_protein.append(len(unique_genes))
            all_genes.extend(unique_genes)
        else:
            gene_counts.append(0)
            unique_genes_per_protein.append(0)
    
    # Calculate statistics
    total_genes = sum(gene_counts)
    total_unique_genes = sum(unique_genes_per_protein)
    overall_unique_genes = len(set(all_genes))
    
    print(f"Total gene annotations found: {total_genes}")
    print(f"Unique genes per protein: {total_unique_genes}")
    print(f"Overall unique genes: {overall_unique_genes}")
    
    if gene_counts:
        print(f"Average genes per multi-ID protein: {sum(gene_counts)/len(gene_counts):.2f}")
        print(f"Average unique genes per multi-ID protein: {sum(unique_genes_per_protein)/len(unique_genes_per_protein):.2f}")
    
    # Analyze gene distribution
    gene_counter = Counter(all_genes)
    most_common_genes = gene_counter.most_common(10)
    
    print(f"\n🏆 MOST COMMON GENES IN MULTI-ID PROTEINS:")
    for gene, count in most_common_genes:
        print(f"  {gene}: {count} occurrences")
    
    # Check for proteins with same genes
    print(f"\n🔍 PROTEINS WITH SAME GENES:")
    gene_to_proteins = defaultdict(list)
    
    for idx, row in multi_id_proteins.iterrows():
        fasta_header = row["Fasta headers"]
        genes = extract_gene_from_fasta_header(fasta_header)
        unique_genes = list(set(genes))
        
        for gene in unique_genes:
            gene_to_proteins[gene].append(row["Protein IDs"])
    
    same_gene_count = sum(1 for gene, proteins in gene_to_proteins.items() if len(proteins) > 1)
    print(f"Genes appearing in multiple multi-ID proteins: {same_gene_count}")
    
    # Show examples
    print(f"\n📋 EXAMPLES OF GENES IN MULTIPLE PROTEINS:")
    examples_shown = 0
    for gene, proteins in gene_to_proteins.items():
        if len(proteins) > 1 and examples_shown < 5:
            print(f"  Gene: {gene}")
            print(f"    Proteins: {len(proteins)}")
            for protein in proteins[:3]:  # Show first 3
                print(f"      - {protein}")
            if len(proteins) > 3:
                print(f"      ... and {len(proteins)-3} more")
            print()
            examples_shown += 1
    
    # Analyze intensity patterns
    print(f"\n📈 INTENSITY ANALYSIS:")
    intensity_cols = [col for col in df.columns if col.startswith("Intensity ")]
    
    if intensity_cols:
        # Calculate total intensity for multi-ID proteins
        multi_id_intensities = multi_id_proteins[intensity_cols].sum(axis=1)
        detected_multi_id = (multi_id_intensities > 0).sum()
        
        print(f"Multi-ID proteins with detected intensity: {detected_multi_id}/{len(multi_id_proteins)} ({detected_multi_id/len(multi_id_proteins)*100:.1f}%)")
        
        # Compare with single-ID proteins
        single_id_intensities = single_id_proteins[intensity_cols].sum(axis=1)
        detected_single_id = (single_id_intensities > 0).sum()
        
        print(f"Single-ID proteins with detected intensity: {detected_single_id}/{len(single_id_proteins)} ({detected_single_id/len(single_id_proteins)*100:.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if overall_unique_genes < len(multi_id_proteins):
        print(f"✅ Multi-ID proteins represent {overall_unique_genes} unique genes out of {len(multi_id_proteins)} proteins")
        print(f"   This suggests gene-level aggregation could be beneficial")
        
        # Calculate potential reduction
        potential_reduction = len(multi_id_proteins) - overall_unique_genes
        print(f"   Potential reduction in protein count: {potential_reduction} proteins")
        
        if detected_multi_id > 0:
            print(f"✅ {detected_multi_id} multi-ID proteins have detected intensity")
            print(f"   These should be considered for inclusion in analysis")
    else:
        print(f"⚠️  Multi-ID proteins represent {overall_unique_genes} unique genes")
        print(f"   No reduction possible through gene aggregation")
    
    # Save detailed results
    print(f"\n💾 SAVING DETAILED ANALYSIS...")
    
    # Create summary dataframe
    summary_data = []
    for idx, row in multi_id_proteins.iterrows():
        fasta_header = row["Fasta headers"]
        genes = extract_gene_from_fasta_header(fasta_header)
        unique_genes = list(set(genes))
        
        # Calculate total intensity
        total_intensity = 0
        if intensity_cols:
            total_intensity = row[intensity_cols].sum()
        
        summary_data.append({
            'Protein_IDs': row["Protein IDs"],
            'Gene_Count': len(genes),
            'Unique_Gene_Count': len(unique_genes),
            'Genes': ';'.join(unique_genes) if unique_genes else 'None',
            'Total_Intensity': total_intensity,
            'Detected': total_intensity > 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('multi_id_gene_analysis.csv', index=False)
    print(f"Detailed analysis saved to: multi_id_gene_analysis.csv")
    
    return summary_df

if __name__ == "__main__":
    results = analyze_multi_id_proteins()
