import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
import math
import sys

# Add parent directory to path to import data_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processor import DataProcessor

# -----------------------
# Paths (edit if needed)
# -----------------------
file_path = r"raw_data.txt.gz"
base_out   = r"./"

# -----------------------
# Helpers
# -----------------------
def _to_newick(node, leaf_names, parent_dist, buffer):
    if node.is_leaf():
        name = leaf_names[node.id]
        length = parent_dist - node.dist
        buffer.append(f"{name}:{max(length,0):.6f}")
    else:
        buffer.append("(")
        _to_newick(node.get_left(),  leaf_names, node.dist, buffer)
        buffer.append(",")
        _to_newick(node.get_right(), leaf_names, node.dist, buffer)
        buffer.append(")")
        if parent_dist is None:
            buffer.append(";")
        else:
            length = parent_dist - node.dist
            buffer.append(f":{max(length,0):.6f}")

def write_newick(Z, labels, out_path):
    tree = to_tree(Z, rd=False)
    buf = []
    _to_newick(tree, labels, parent_dist=None, buffer=buf)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(buf))

def calculate_protein_uniqueness(df, strain_cols):
    """Calculate exclusive sharing categories for each strain:
    - unique_sample: only in this strain
    - species_only: shared with same-species strains, but in no other species
    - <Sp>_only: shared with that species only (no other species, no same-species)
    - others: everything else (multi-species mixes, combinations incl. same+other species)
    """
    strain_data = {}
    
    # Get proteins detected in each strain
    for strain in strain_cols:
        detected_proteins = df[df[strain] > 0].index.tolist()
        strain_data[strain] = set(detected_proteins)
    
    # Group strains by species
    species_groups = {
        "Lb": [s for s in strain_cols if s.startswith("Intensity Lb")],
        "Lg": [s for s in strain_cols if s.startswith("Intensity Lg")],
        "Ln": [s for s in strain_cols if s.startswith("Intensity Ln")],
        "Lp": [s for s in strain_cols if s.startswith("Intensity Lp")]
    }
    
    # Calculate exclusive breakdown for each strain
    uniqueness_data = {}
    for strain in strain_cols:
        strain_proteins = strain_data[strain]
        strain_species = None
        for sp, strains in species_groups.items():
            if strain in strains:
                strain_species = sp
                break
        
        if strain_species is None:
            continue

        # Universe of other strains' proteins
        union_all_others = set()
        for other_s, prots in strain_data.items():
            if other_s != strain:
                union_all_others |= prots

        # Unique to this sample
        unique_sample = strain_proteins - union_all_others

        # Proteins from same species (excluding this strain)
        union_same_species = set()
        for other in species_groups[strain_species]:
            if other != strain:
                union_same_species |= strain_data[other]

        # Proteins from each other species
        union_by_species = {}
        for sp2, members in species_groups.items():
            if sp2 == strain_species:
                continue
            u = set()
            for m in members:
                u |= strain_data[m]
            union_by_species[sp2] = u

        # Exclusive species-only: shared with same species, not present in any other species
        species_only = (strain_proteins & union_same_species)
        for sp2, u in union_by_species.items():
            species_only -= u

        # Exclusive sharing with a single other species (not same species, not any other species)
        sp_only_counts = {"Lb": 0, "Lg": 0, "Ln": 0, "Lp": 0}
        for sp2 in ["Lb", "Lg", "Ln", "Lp"]:
            if sp2 == strain_species:
                continue
            only_set = (strain_proteins & union_by_species[sp2])
            # exclude same-species and any other species unions
            only_set -= union_same_species
            for sp3, u3 in union_by_species.items():
                if sp3 != sp2:
                    only_set -= u3
            sp_only_counts[sp2] = len(only_set)

        # Others: everything else
        accounted = unique_sample | species_only
        for sp2, members in species_groups.items():
            if sp2 == strain_species:
                continue
            # reconstruct the set for sp2-only to subtract
            # Note: recompute to avoid rounding issues
            only_set = (strain_proteins & union_by_species[sp2])
            only_set -= union_same_species
            for sp3, u3 in union_by_species.items():
                if sp3 != sp2:
                    only_set -= u3
            accounted |= only_set
        others = strain_proteins - accounted

        uniqueness_data[strain] = {
            'unique_sample': len(unique_sample),
            'species_only': len(species_only),
            'Lb_only': sp_only_counts["Lb"],
            'Lg_only': sp_only_counts["Lg"],
            'Ln_only': sp_only_counts["Ln"],
            'Lp_only': sp_only_counts["Lp"],
            'others': len(others),
            'total': len(strain_proteins)
        }
    
    return uniqueness_data

def plot_radial_phtic_tree(Z, labels, uniqueness_data, out_png, title_suffix):
    """Create a true radial dendrogram using the clustering structure in Z,
    split into four species sections, and overlay per-strain pie charts.
    """
    
    # Species colors and mapping
    species_colors = {
        "Lb": "#1f77b4",  # blue
        "Lg": "#ff7f0e",  # orange  
        "Ln": "#d62728",  # red
        "Lp": "#2ca02c"   # green
    }
    
    def strain_to_species(label):
        if label.startswith("Intensity Lb"): return "Lb"
        if label.startswith("Intensity Lg"): return "Lg"
        if label.startswith("Intensity Ln"): return "Ln"
        if label.startswith("Intensity Lp"): return "Lp"
        return "Unknown"
    
    # Clean labels (remove "Intensity" and species prefix)
    clean_labels = []
    for label in labels:
        clean_label = label.replace("Intensity ", "")
        # Remove species prefix (Lb_, Lg_, Ln_, Lp_)
        for prefix in ["Lb_", "Lg_", "Ln_", "Lp_"]:
            if clean_label.startswith(prefix):
                clean_label = clean_label[len(prefix):]
                break
        clean_labels.append(clean_label)
    
    # Group strains by species
    species_groups = {"Lb": [], "Lg": [], "Ln": [], "Lp": []}
    for i, label in enumerate(labels):
        species = strain_to_species(label)
        if species in species_groups:
            species_groups[species].append((i, clean_labels[i]))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 24))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    
    # Calculate proportional section angles based on number of samples per species
    species_counts = {sp: len(indices) for sp, indices in species_groups.items()}
    total_samples = sum(species_counts.values())
    
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

        # label will be drawn after computing unique % (to use smaller font for the %)

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
        ax.text(x, y, clean_labels[leaf_idx], fontsize=8, ha='center', va='center',
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
                
                # OUTER CIRCLE: Proteins shared with other species
                outer_radius_pie = 0.045
                if shared_other_species > 0:
                    # Create outer circle showing proteins shared with other species
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
    
    # Add title
    ax.set_title(f'Leishmania Phylogenic Tree {title_suffix}', 
                fontsize=16, pad=20)
    
    # Add legend below the radial plot
    legend_elements = [
        mpatches.Patch(color='lightgray', alpha=0.7, label='Outer circle: Proteins shared with other species'),
        mpatches.Patch(color='blue', alpha=0.4, label='Inner circle: Species-only shared'),
        mpatches.Patch(color='blue', alpha=0.2, label='Inner circle: Unique to sample'),
    ]
    
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.005))
    
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

# -----------------------
# Distances
# -----------------------
def braycurtis_upgma(matrix, labels, tag):
    D = pdist(matrix, metric="braycurtis")
    Z = linkage(D, method="average")
    return Z, D

def spearman_distance(X):
    df = pd.DataFrame(X).T
    corr = df.corr(method="spearman").values
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = np.maximum((dist + dist.T) / 2.0, 0)
    return squareform(dist, checks=False)

def spearman_upgma(matrix, labels, tag):
    D = spearman_distance(matrix)
    Z = linkage(D, method="average")
    return Z, D



# -----------------------
# Helper Functions for Dendrogram
# -----------------------
def plot_dendro(Z, labels, title, out_png):
    # Fixed colors for each species
    species_color_map = {
        "Lb": "blue",
        "Lg": "orange",
        "Ln": "red",
        "Lp": "green"
    }

    # Map strain label -> species
    def strain_to_species(label):
        if label.startswith("Intensity Lb"): return "Lb"
        if label.startswith("Intensity Lg"): return "Lg"
        if label.startswith("Intensity Ln"): return "Ln"
        if label.startswith("Intensity Lp"): return "Lp"
        return "Unknown"

    # Clean labels: remove "Intensity" and species prefixes
    def clean_label(label):
        # Remove "Intensity " prefix
        clean_label = label.replace("Intensity ", "")
        # Remove species prefix (Lb_, Lg_, Ln_, Lp_)
        for prefix in ["Lb_", "Lg_", "Ln_", "Lp_"]:
            if clean_label.startswith(prefix):
                clean_label = clean_label[len(prefix):]
                break
        return clean_label

    # Create cleaned labels
    cleaned_labels = [clean_label(label) for label in labels]
    
    # Build strain -> species -> color maps
    strain_to_color = {lab: species_color_map.get(strain_to_species(lab), "black")
                       for lab in labels}
    strain_to_species_map = {lab: strain_to_species(lab) for lab in labels}

    # For each node, determine branch color
    from collections import defaultdict
    cluster_species = {}

    def collect_species(node_id):
        if node_id < len(labels):  # leaf
            sp = strain_to_species_map[labels[node_id]]
            cluster_species[node_id] = {sp}
        else:
            left, right = int(Z[node_id - len(labels), 0]), int(Z[node_id - len(labels), 1])
            cluster_species[node_id] = collect_species(left) | collect_species(right)
        return cluster_species[node_id]

    # Recursively assign sets of species
    root_id = len(Z) + len(labels) - 2
    collect_species(root_id)

    def _link_color_func(node_id):
        species_set = cluster_species.get(node_id, set())
        if len(species_set) == 1:
            # pure cluster → use its species color
            sp = list(species_set)[0]
            return species_color_map.get(sp, "black")
        else:
            # mixed cluster → take majority species
            leaves = []
            def collect_leaves(nid):
                if nid < len(labels):
                    leaves.append(labels[nid])
                else:
                    left, right = int(Z[nid - len(labels), 0]), int(Z[nid - len(labels), 1])
                    collect_leaves(left)
                    collect_leaves(right)
            collect_leaves(node_id)

            # count species
            sp_counts = {}
            for leaf in leaves:
                sp = strain_to_species_map[leaf]
                sp_counts[sp] = sp_counts.get(sp, 0) + 1
            # pick majority
            majority_sp = max(sp_counts, key=sp_counts.get)
            return species_color_map.get(majority_sp, "black")

    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(Z,
               labels=cleaned_labels,
               leaf_rotation=90,
               link_color_func=_link_color_func,
               above_threshold_color="black")

    # Color leaf labels and add species information
    ax = plt.gca()
    for i, lbl in enumerate(ax.get_xmajorticklabels()):
        strain = labels[i]  # Use original label for color mapping
        species = strain_to_species_map[strain]
        lbl.set_color(strain_to_color[strain])
        # Add species information to the label
        lbl.set_text(f"{lbl.get_text()} ({species})")

    # Add species legend
    legend_elements = []
    for species, color in species_color_map.items():
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=f'{species}'))
    
    plt.legend(handles=legend_elements, loc='upper right', title='Species')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# -----------------------
# Main Analysis Function
# -----------------------
def run_phylogenetic_analysis(analysis_type="protein"):
    """
    Run phylogenetic analysis for either protein-based or gene-based data.
    
    Args:
        analysis_type (str): Either "protein" or "gene"
    """
    if analysis_type == "protein":
        phylo_dir = os.path.join(base_out, "phylogenic-analysis/protein-based")
        title_suffix = "based on Protein Groups"
        data_source = "protein_df"
    elif analysis_type == "gene":
        phylo_dir = os.path.join(base_out, "phylogenic-analysis/gene-based")
        title_suffix = "based on Unique Genes"
        data_source = "gene_df"
    else:
        raise ValueError("analysis_type must be either 'protein' or 'gene'")
    
    os.makedirs(phylo_dir, exist_ok=True)
    
    print(f"🔄 Running {analysis_type}-based phylogenetic analysis...")
    
    # Initialize data processor
    processor = DataProcessor(file_path)
    
    # Load and process data
    protein_df, gene_df = processor.load_and_process_data()
    sample_info_df = processor.sample_info_df
    
    # Select appropriate dataframe
    if analysis_type == "protein":
        df = protein_df.copy()
        print(f"✅ Loaded {len(df)} protein groups for phylogenetic analysis")
    else:  # gene
        df = gene_df.copy()
        print(f"✅ Loaded {len(df)} genes with unique gene mapping for phylogenetic analysis")
    
    df = df.set_index("Protein_IDs")
    
    # Identify species intensity columns using processor information
    species_cols = {}
    for species in ['Lb', 'Lg', 'Ln', 'Lp']:
        species_cols[species] = [c for c in df.columns if c.startswith(f'Intensity {species}_')]
        if len(species_cols[species]) > 0:
            print(f"Found {len(species_cols[species])} columns for {species}")
    
    # Strain-level matrix
    all_cols = sum(species_cols.values(), [])
    X_strain = df[all_cols].copy()
    X_strain = np.log1p(X_strain)
    X_strain = pd.DataFrame(
        SimpleImputer(strategy="constant", fill_value=0.0).fit_transform(X_strain),
        index=df.index, columns=all_cols
    )
    X_strain = X_strain.T
    strain_labels = list(X_strain.index)
    
    print("Calculating protein uniqueness...")
    # Calculate protein uniqueness for each strain
    uniqueness_data = calculate_protein_uniqueness(df, all_cols)
    
    print("Building phylogenetic trees...")
    # Build trees
    Z_bray, D_bray = braycurtis_upgma(X_strain.values, strain_labels, tag="strain")
    Z_spear, D_spear = spearman_upgma(X_strain.values, strain_labels, tag="strain")
    
    print("Generating radial phylogenetic tree...")
    # Generate radial phylogenetic tree with appropriate title
    plot_radial_phtic_tree(Z_bray, strain_labels, uniqueness_data, 
                          os.path.join(phylo_dir, "radial_phylogenetic_tree.png"),
                          title_suffix)
    
    # Save traditional dendrogram for comparison
    plot_dendro(Z_bray, strain_labels, f"UPGMA (Bray-Curtis) – strain ({analysis_type}-based)",
                os.path.join(phylo_dir, "traditional_dendrogram.png"))
    
    print(f"✅ {analysis_type.capitalize()}-based phylogenetic analysis complete!")
    print(f"📁 Results saved to: {phylo_dir}")
    print("Files generated:")
    print("- radial_phylogenetic_tree.png: Radial tree with pie charts")
    print("- traditional_dendrogram.png: Traditional dendrogram for comparison")
    
    return phylo_dir

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    print("🚀 Starting comprehensive phylogenetic analysis...")
    print("=" * 60)
    
    # Run protein-based analysis
    print("\n📊 PROTEIN-BASED ANALYSIS")
    print("-" * 30)
    run_phylogenetic_analysis("protein")
    
    # Run gene-based analysis
    print("\n🧬 GENE-BASED ANALYSIS")
    print("-" * 30)
    run_phylogenetic_analysis("gene")
    
    print("\n✅ All phylogenetic analyses complete!")
    print("=" * 60)




