# Phyloproteomics Analysis Suite

This project provides comprehensive analysis tools for protein expression data across four Leishmania species, including gene-based analysis, metabolic pathway analysis, phylogenetic analysis, and protein intensity analysis.

## Overview

The analysis processes protein expression data from four Leishmania species:
- **Lb** (Leishmania braziliensis)
- **Lg** (Leishmania guyanensis) 
- **Ln** (Leishmania naiffi)
- **Lp** (Leishmania panamensis)

## Requirements

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## Input Data

- `raw_data.txt.gz` - Main protein expression data file (gzipped for GitHub compatibility)
  - Contains intensity columns for each species sample
  - Includes protein IDs, Fasta headers, and annotation columns
  - Automatically handles gzipped files

## Analysis Scripts

### 🧬 Gene-Based Analysis (Recommended)

#### 1. **Gene-Based Analysis** (`gene_based_analysis.py`)
Comprehensive gene-level analysis using only protein groups that map to unique genes.

```bash
python gene_based_analysis.py
```

**Output:** `gene_analysis/`
- `gene_statistics.csv` - Gene-level statistics
- `gene_intensity_distributions.png` - Intensity distribution plots
- `gene_sharing_heatmap.png` - Gene sharing patterns
- `gene_uniqueness_summary.png` - Gene uniqueness analysis

#### 2. **Gene-Based Sample Analysis** (`gene_based_sample_analysis.py`)
Sample-level phylogenetic analysis using gene-filtered data.

```bash
python gene_based_sample_analysis.py
```

**Output:** `gene_sample_analysis/`
- `gene_sample_radial_phylogenetic_tree.png` - Radial phylogenetic tree
- `gene_sample_traditional_dendrogram.png` - Traditional dendrogram
- `gene_sample_uniqueness_summary.csv` - Sample uniqueness data

#### 3. **Gene-Based Radial Plot** (`gene_based_radial_plot.py`)
Specialized radial phylogenetic visualization with gene-filtered data.

```bash
python gene_based_radial_plot.py
```

**Output:** `gene_radial_analysis/`
- `gene_radial_phylogenetic_tree.png` - Radial plot with sample labels
- `gene_radial_uniqueness_summary.csv` - Radial plot statistics

### 🔬 Protein Intensity Analysis

#### 4. **Gene-Filtered Protein Intensity Analysis** (`analysis/gene_filtered_protein_intensity_analysis.py`)
Protein intensity analysis using only proteins mapping to unique genes.

```bash
cd analysis
python gene_filtered_protein_intensity_analysis.py
```

**Output:** `analysis/gene_filtered_analysis/`
- `gene_filtered_intensity_violin_plots.png` - Violin plots by species
- `gene_filtered_intensity_histograms.png` - Intensity histograms
- `gene_filtered_ma_plots.png` - MA plots for differential expression
- `gene_filtered_comprehensive_analysis_report.png` - Comprehensive report
- `gene_filtered_statistical_tests.txt` - Statistical test results
- `gene_filtered_processed_intensity_data.csv` - Processed data

#### 5. **Original Protein Intensity Analysis** (`analysis/protein_intensity_analysis.py`)
Protein intensity analysis using all proteins (including multi-gene protein groups).

```bash
cd analysis
python protein_intensity_analysis.py
```

**Output:** `analysis/`
- `intensity_violin_plots.png` - Violin plots by species
- `intensity_histograms.png` - Intensity histograms
- `ma_plots.png` - MA plots for differential expression
- `comprehensive_analysis_report.png` - Comprehensive report
- `statistical_tests.txt` - Statistical test results
- `processed_intensity_data.csv` - Processed data

### 🧪 Metabolic Pathway Analysis

#### 6. **Metabolic Pathway Analyzer** (`metabolic-analysis/metabolic_pathway_analyzer.py`)
Metabolic pathway analysis using UniProt keyword annotations.

```bash
cd metabolic-analysis
python metabolic_pathway_analyzer.py
```

**Output:** `metabolic-analysis/`
- `processed_protein_data.csv` - Processed protein data
- `pathway_expression_matrix.csv` - Expression matrix by pathway
- `metabolic_heatmap.png` - Metabolic pathway heatmap
- `metabolic_analysis_report.txt` - Analysis report

#### 7. **Refined Metabolic Pathway Analyzer** (`metabolic-analysis/metabolic_pathway_analyzer_refined.py`)
Enhanced metabolic pathway analysis with additional filtering and quality control.

```bash
cd metabolic-analysis
python metabolic_pathway_analyzer_refined.py
```

**Output:** `metabolic-analysis/`
- `refined_processed_protein_data.csv` - Refined processed data
- `refined_pathway_expression_matrix.csv` - Refined expression matrix
- `refined_metabolic_heatmap.png` - Refined heatmap
- `quality_control_plots.png` - Quality control visualizations
- `refined_metabolic_analysis_report.txt` - Refined analysis report

### 🌳 Phylogenetic Analysis

#### 8. **Radial Phylogenetic Analyzer** (`phylogenic-analysis/radial_phylo_analyzer.py`)
Original radial phylogenetic analysis using all proteins.

```bash
cd phylogenic-analysis
python radial_phylo_analyzer.py
```

**Output:** `phylogenic-analysis/phylo/`
- `radial_phylogenetic_tree.png` - Radial phylogenetic tree
- `traditional_dendrogram.png` - Traditional dendrogram
- `protein_uniqueness_summary.csv` - Protein uniqueness data
- `detailed_protein_breakdown.csv` - Detailed protein breakdown

## Analysis Features

### 🧬 Gene-Based Analysis Features
- **Gene filtering**: Only includes protein groups mapping to unique genes
- **Sample-level clustering**: Phylogenetic analysis at sample level
- **Gene aggregation**: Combines multiple protein groups mapping to same gene
- **Consistent filtering**: Same criteria across all gene-based analyses

### 🔬 Protein Intensity Analysis Features
- **Median calculation**: Per protein per species (excluding zeros)
- **Log2 transformation**: For normalization and visualization
- **Statistical testing**: Mann-Whitney U and Kruskal-Wallis tests
- **MA plots**: Differential expression analysis
- **Violin plots**: Intensity distribution visualization

### 🧪 Metabolic Pathway Analysis Features
- **21 pathway categories**: Comprehensive metabolic pathway annotation
- **UniProt keyword matching**: Based on protein descriptions
- **Log fold change**: Analysis across species
- **Quality control**: Multiple filtering steps

### 🌳 Phylogenetic Analysis Features
- **Hierarchical clustering**: Bray-Curtis and Spearman distance
- **UPGMA algorithm**: Unweighted Pair Group Method with Arithmetic Mean
- **Uniqueness analysis**: Protein/gene sharing patterns
- **Radial visualization**: Specialized circular dendrograms

## Key Results

The analysis suite provides multiple perspectives on protein expression patterns:

1. **Gene-based analysis** reveals consistent patterns using unique gene mappings
2. **Protein intensity analysis** shows expression differences across species
3. **Metabolic pathway analysis** identifies pathway-specific expression patterns
4. **Phylogenetic analysis** reveals evolutionary relationships between species

## Data Processing

### Filtering Criteria
- **Decoys**: Removes reverse proteins (REV_)
- **Contaminants**: Removes potential contaminants
- **Undetected**: Removes proteins with zero intensity across all species
- **Gene filtering**: Only includes proteins mapping to unique genes (gene-based analysis)

### Data Transformation
- **Log2 transformation**: `log2(intensity + 1)` for normalization
- **Median calculation**: Per species, excluding zeros
- **Missing value imputation**: Simple imputer for clustering

## File Structure

```
phyloproteomics/
├── raw_data.txt.gz                    # Input data (gzipped)
├── gene_based_analysis.py             # Gene-based analysis
├── gene_based_sample_analysis.py      # Gene-based sample analysis
├── gene_based_radial_plot.py          # Gene-based radial plot
├── analysis/
│   ├── gene_filtered_protein_intensity_analysis.py  # Gene-filtered intensity analysis
│   └── protein_intensity_analysis.py                # Original intensity analysis
├── metabolic-analysis/
│   ├── metabolic_pathway_analyzer.py                # Metabolic pathway analysis
│   └── metabolic_pathway_analyzer_refined.py        # Refined metabolic analysis
├── phylogenic-analysis/
│   └── radial_phylo_analyzer.py                     # Original phylogenetic analysis
└── README.md                          # This file
```

## Recommendations

1. **For consistent analysis**: Use gene-based analysis scripts (scripts 1-3)
2. **For comprehensive protein analysis**: Use gene-filtered intensity analysis (script 4)
3. **For metabolic insights**: Use refined metabolic pathway analysis (script 7)
4. **For phylogenetic relationships**: Use gene-based sample analysis (script 2)

All scripts automatically handle gzipped input files and provide comprehensive output with visualizations and statistical analyses.
