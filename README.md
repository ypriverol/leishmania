# Metabolic Pathway Heatmap Analysis

This project analyzes protein expression data across four Leishmania species and generates metabolic pathway heatmaps.

## Overview

The analysis processes protein expression data from four Leishmania species:
- **Lb** (Leishmania braziliensis)
- **Lg** (Leishmania guyanensis) 
- **Ln** (Leishmania naiffi)
- **Lp** (Leishmania panamensis)

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the analysis:

```bash
python metabolic_pathway_heatmap_analyzer.py
```

## Input

- `raw_data.txt` - Main protein expression data file with intensity columns for each species

## Output Files

The analysis generates the following files:

### Data Files
- `improved_log_processed_protein_data.csv` - Processed protein data with pathway annotations
- `improved_log_pathway_expression_matrix.csv` - Expression matrix by pathway and species
- `improved_log_metabolic_analysis_report.txt` - Detailed analysis report

### Visualizations
- `improved_log_metabolic_heatmap.png` - Main heatmap showing pathway expression across species
- `improved_log_expression_bar_plot.png` - Bar plot of expression levels
- `improved_log_summary_plots.png` - Summary plots including distributions and correlations
- `improved_log_species_comparison.png` - Species comparison visualization

## Analysis Features

- **Species identification** based on intensity column patterns (`Intensity_{species}_{sample_id}`)
- **Pathway annotation** using comprehensive keyword matching
- **Log2 transformation** with normalization for proper visualization
- **Median expression** calculation per pathway
- **Log fold change** analysis across species
- **21 pathway categories** including:
  - Oxidative Phosphorylation
  - Signal Transduction
  - Amino Acid Metabolism
  - Membrane Proteins
  - DNA Replication
  - Proteasome
  - Oxidative Stress
  - And more...

## Key Results

The analysis successfully identifies protein expression patterns across all four Leishmania species, revealing species-specific metabolic pathway preferences and differences in protein expression levels.
