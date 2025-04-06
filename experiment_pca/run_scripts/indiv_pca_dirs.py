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
    Perform PCA on hidden-state vectors for each subdirectory (e.g. 'attack', 'jailbreak', etc.),
    separating data by classification label (e.g. 'refusing' vs. 'direct').

    :param input_dir:   str, path to the directory containing subdirectories of .pt inference files
    :param output_dir:  str, path to where PCA plots (PNGs) will be saved
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

    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        pt_files = sorted(glob(os.path.join(subdir_path, "*.pt")))
        if not pt_files:
            logger.info(f"No .pt files found under subdirectory '{subdir}'. Skipping.")
            continue

        logger.info(f"Processing subdir='{subdir}' with {len(pt_files)} .pt files.")

        # We'll collect all vectors and labels
        all_vectors = []
        all_labels = []

        # ----------------------------------------------------------
        # Load each .pt file and extract hidden states from layer_X,
        # plus the classification label (refusing or direct).
        # ----------------------------------------------------------
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
                    all_vectors.append(emb)
                    all_labels.append(lab)

            except Exception as e:
                logger.error(f"Failed loading {pt_file} due to {e}")

        if not all_vectors:
            logger.info(f"No valid samples in '{subdir}' after reading all .pt files.")
            continue

        # Convert to numpy
        all_vectors = np.array(all_vectors)
        all_labels = np.array(all_labels)

        # Sanity check
        unique_labels = set(all_labels)
        logger.info(f"Unique labels in subdir '{subdir}': {unique_labels}")

        if len(unique_labels) < 2:
            logger.warning(
                f"In subdir '{subdir}', only one label type found. "
                "Will still do PCA, but the plot might be less interesting."
            )

        # ------------------------------------
        # PCA: reduce to 2 components
        # ------------------------------------
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(all_vectors)

        # Prepare a scatter plot
        plt.figure(figsize=(6, 5))
        plt.title(f"PCA for '{subdir}' - Layer {layer}")

        # Plot points by label
        for lab in unique_labels:
            inds = (all_labels == lab)
            plt.scatter(
                pca_coords[inds, 0],
                pca_coords[inds, 1],
                label=lab,
                alpha=0.7,  # slightly transparent for overlap
            )

        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.legend()
        # Save figure
        fig_path = os.path.join(output_dir, f"pca_{subdir}_layer{layer}.png")
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved PCA plot: {fig_path}")

def main():
    parser = argparse.ArgumentParser(description="Perform PCA on subdirectory hidden-states.")
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
