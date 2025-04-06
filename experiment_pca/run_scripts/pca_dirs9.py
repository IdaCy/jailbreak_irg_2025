#!/usr/bin/env python
import os
import argparse
import logging
from glob import glob

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # So it does not require an X server
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def convert_bfloat16_to_float32(obj):
    """
    Recursively convert any bfloat16 tensors in a structure (dict, list, tensor)
    into float32 tensors. Returns the modified structure.
    """
    if isinstance(obj, torch.Tensor):
        if obj.dtype == torch.bfloat16:
            return obj.to(torch.float32)
        else:
            return obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = convert_bfloat16_to_float32(v)
        return obj
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = convert_bfloat16_to_float32(obj[i])
        return obj
    else:
        # Other types (e.g., int, string, float) remain unmodified
        return obj

def perform_pca_and_plot(input_dir, output_dir, layer, logger=None):
    """
    Perform a single PCA across all subdirectories combined.
    Each subdirectory gets a different color; within each color,
    'refusing' is darker (higher alpha) and 'direct' is lighter (lower alpha).

    :param input_dir:   str, path to the directory containing subdirectories of .pt inference files
    :param output_dir:  str, path to where the PCA plot (PNG) will be saved
    :param layer:       int, which layer index to use (e.g. 0, 5, 10, etc.)
    :param logger:      optional logger object
    """
    if logger is None:
        logger = logging.getLogger("pcaLogger")
        logger.setLevel(logging.INFO)

    os.makedirs(output_dir, exist_ok=True)

    # Gather subdirectories in input_dir
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not subdirs:
        logger.warning(f"No subdirectories found in {input_dir}. Exiting.")
        return

    # For one big PCA, we'll aggregate across all subdirs
    global_vectors = []
    global_subdirs = []
    global_labels  = []

    # We'll define a simple color palette for subdirs
    base_colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    subdir_color_map = {}
    for i, s in enumerate(sorted(subdirs)):
        subdir_color_map[s] = base_colors[i % len(base_colors)]

    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        pt_files = sorted(glob(os.path.join(subdir_path, "*.pt")))
        if not pt_files:
            logger.info(f"No .pt files found under subdirectory '{subdir}'. Skipping.")
            continue

        logger.info(f"Processing subdir='{subdir}' with {len(pt_files)} .pt files.")

        valid_samples_found = False

        for pt_file in pt_files:
            try:
                # 1) Load on CPU
                data_dict = torch.load(pt_file, map_location="cpu")
                # 2) Convert any bfloat16 -> float32
                data_dict = convert_bfloat16_to_float32(data_dict)

                layer_key = f"layer_{layer}"
                if "hidden_states" not in data_dict or layer_key not in data_dict["hidden_states"]:
                    logger.warning(f"Missing {layer_key} in {pt_file}; skipping.")
                    continue

                hidden_states = data_dict["hidden_states"][layer_key]  # shape: (B, S, H)
                batch_size, seq_len, hidden_dim = hidden_states.shape

                # We assume the labeled responses are in 'response_types'
                if "response_types" not in data_dict:
                    logger.warning(f"No 'response_types' found in {pt_file}; skipping.")
                    continue

                classifications = data_dict["response_types"]  # e.g. ["refusing", "direct", ...]
                if len(classifications) != batch_size:
                    logger.warning(f"Number of labels != batch_size in {pt_file}. Skipping.")
                    continue

                # Mean-pool hidden states across seq_len to get a single embedding per sample
                hidden_mean = hidden_states.mean(dim=1)  # shape: (B, H)

                # Accumulate them
                for i in range(batch_size):
                    emb = hidden_mean[i].cpu().numpy()
                    lab = classifications[i]
                    global_vectors.append(emb)
                    global_labels.append(lab)
                    global_subdirs.append(subdir)

                valid_samples_found = True

            except Exception as e:
                logger.error(f"Failed loading {pt_file} due to {e}")

        if not valid_samples_found:
            logger.info(f"No valid samples in '{subdir}' after reading all .pt files.")

    # Once all subdirs are processed, check if we have any data at all
    if not global_vectors:
        logger.warning("No valid data found in any subdirectory. Nothing to plot.")
        return

    # Convert to NumPy arrays
    global_vectors = np.array(global_vectors)
    global_subdirs = np.array(global_subdirs)
    global_labels  = np.array(global_labels)

    # ------------------------------------
    # PCA: reduce to 2 components
    # ------------------------------------
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(global_vectors)

    # Prepare a single scatter plot for all subdirs & labels
    plt.figure(figsize=(6, 5))
    plt.title(f"Combined PCA - Layer {layer}")

    # We'll plot each subdir-label combo with the subdir's color,
    # but use a different alpha for refusing vs direct.
    # refusing => alpha=0.8 (darker), direct => alpha=0.3 (lighter).
    for subdir in sorted(set(global_subdirs)):
        color = subdir_color_map[subdir]
        # For each subdir, we have two possible labels: refusing / direct
        for lab in ["refusing", "direct"]:
            mask = (global_subdirs == subdir) & (global_labels == lab)
            if not np.any(mask):
                continue

            if lab == "refusing":
                alpha_val = 0.8
            else:
                alpha_val = 0.3

            # Label in the legend: e.g., "attack (refusing)"
            legend_label = f"{subdir} ({lab})"
            plt.scatter(
                pca_coords[mask, 0],
                pca_coords[mask, 1],
                c=color,
                alpha=alpha_val,
                s=10,  # half the typical default size
                label=legend_label
            )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()

    # Save figure with higher resolution (DPI=300)
    fig_path = os.path.join(output_dir, f"pca_all_layer{layer}.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved single PCA plot for all data: {fig_path}")

def main():
    parser = argparse.ArgumentParser(description="Perform PCA on hidden-states from multiple subdirectories, combining them into one plot.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with subdirs that contain .pt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save PCA plots")
    parser.add_argument("--layer", type=int, default=0, help="Which layer index to use for PCA (e.g. 0, 5, 10, etc.)")

    args = parser.parse_args()

    # Set up default logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pcaLogger")

    perform_pca_and_plot(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        layer=args.layer,
        logger=logger
    )

if __name__ == "__main__":
    main()
