#!/usr/bin/env python
"""
Cosine Similarity Analysis Module for LLM Inference Extractions

This script loads .pt extraction files (output from your inference script),
groups hidden-state vectors by token (optionally filtered by layer or by a target token),
computes the average pairwise cosine similarity for each token group,
and saves the results as a CSV file.

It is designed to be both imported (e.g., in a Google Colab cell) and executed
directly from the command line (e.g., in a SLURM job).

Usage (command-line):
  python cosine_similarity_analysis.py --input_dir OUTPUT_DIR --output_csv RESULT_CSV \
       [--model_name MODEL_NAME] [--layers layer_0 layer_5 ...] [--target_token TOKEN] [--log_level INFO]

Usage (import):
  from cosine_similarity_analysis import run_cosine_similarity_analysis
  run_cosine_similarity_analysis(input_dir, output_csv, model_name=..., layers=..., target_token=..., log_level="INFO")
"""

import os
import torch
import argparse
import logging
import glob
import csv
import numpy as np
from collections import defaultdict

# Try to import the Hugging Face tokenizer
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

def load_extraction_files(input_dir, logger):
    """
    Loads all .pt files from the given directory.
    Each file should contain a dictionary with keys like 'hidden_states', 'input_ids', etc.
    """
    file_paths = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    logger.info(f"Found {len(file_paths)} extraction files in {input_dir}")
    if not file_paths:
        logger.error(f"No extraction files found in {input_dir}")
        return []
    extractions = []
    for fp in file_paths:
        try:
            data = torch.load(fp, map_location="cpu")
            extractions.append(data)
        except Exception as e:
            logger.exception(f"Error loading {fp}: {e}")
    return extractions

def decode_tokens(token_ids, tokenizer):
    """
    Decodes a list of token IDs using the provided tokenizer.
    If no tokenizer is available, returns a fallback representation.
    """
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return tokenizer.convert_ids_to_tokens(token_ids)
    else:
        return [f"<tok_{i}>" for i in token_ids]

def compute_token_cosine_similarity(extractions, tokenizer, layers=None, target_token=None, logger=None):
    """
    For each extraction file and sample, groups hidden-state vectors by token text
    (using the provided tokenizer for decoding), then computes average pairwise cosine similarity
    for tokens that occur more than once.

    Args:
      extractions: list of extraction dicts (each from a .pt file)
      tokenizer: a tokenizer instance for decoding input_ids
      layers: (optional) list of layer keys (e.g., ["layer_0", "layer_5"]) to analyze.
              If None, all available layers are processed.
      target_token: (optional) if provided, only compute for tokens matching this (case-insensitive).
      logger: logger for status messages.

    Returns:
      A list of dictionaries with keys:
         "token", "layer", "count", "avg_pairwise_cos_sim"
    """
    # groups[layer][token] = list of vectors (as numpy arrays)
    groups = defaultdict(lambda: defaultdict(list))
    
    for extraction in extractions:
        # Each extraction - have keys "input_ids" and "hidden_states"
        input_ids = extraction.get("input_ids")
        hidden_states = extraction.get("hidden_states")
        if input_ids is None or hidden_states is None:
            continue
        batch_size, seq_len = input_ids.shape
        for b in range(batch_size):
            token_ids = input_ids[b].tolist()
            tokens = decode_tokens(token_ids, tokenizer)
            for layer_name, tensor in hidden_states.items():
                if layers is not None and layer_name not in layers:
                    continue
                # tensor shape: [batch, seq_len, hidden_dim]
                hidden_vectors = tensor[b].float().numpy()  # shape: [seq_len, hidden_dim]
                for idx, token in enumerate(tokens):
                    if target_token is not None and token.lower() != target_token.lower():
                        continue
                    groups[layer_name][token].append(hidden_vectors[idx])
    
    results = []
    # For each layer and token group, compute cosine similarity
    for layer, token_dict in groups.items():
        for token, vectors in token_dict.items():
            if len(vectors) < 2:
                continue  # Need at least two samples to compare
            arr = np.stack(vectors, axis=0)  # shape [n, hidden_dim]
            # Normalize the vectors
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr_norm = arr / (norms + 1e-9)
            # Compute cosine similarity matrix
            sim_matrix = np.dot(arr_norm, arr_norm.T)
            # Extract upper triangle (excluding diagonal) for pairwise similarities
            n = sim_matrix.shape[0]
            sims = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
            avg_sim = float(np.mean(sims)) if sims else None
            results.append({
                "token": token,
                "layer": layer,
                "count": len(vectors),
                "avg_pairwise_cos_sim": avg_sim
            })
    return results

def run_cosine_similarity_analysis(input_dir, output_csv, model_name=None, layers=None, target_token=None, log_level="INFO"):
    """
    Loads extraction files from input_dir, computes cosine similarity results, and saves them as a CSV.

    Args:
      input_dir: Directory containing .pt extraction files.
      output_csv: Path for the output CSV file.
      model_name: (optional) Hugging Face model name or local path to load the tokenizer.
      layers: (optional) List of layer keys to restrict analysis (e.g., ["layer_0", "layer_5"]).
      target_token: (optional) If provided, only analyze hidden states for this token.
      log_level: Logging level (e.g., "INFO", "DEBUG").
    
    Returns:
      The list of result dictionaries.
    """
    logger = logging.getLogger("CosineSimilarityAnalysis")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    logger.info("Starting cosine similarity analysis")
    
    # Load extraction files
    extractions = load_extraction_files(input_dir, logger)
    if not extractions:
        logger.error("No extraction files loaded. Exiting.")
        return []
    
    # Load tokenizer
    if model_name is not None and AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded tokenizer from {model_name}")
        except Exception as e:
            logger.warning(f"Error loading tokenizer for model '{model_name}': {e}. Using fallback tokenizer.")
            tokenizer = None
    else:
        tokenizer = None

    if tokenizer is None:
        # Fallback tokenizer that simply returns a placeholder token representation.
        class FallbackTokenizer:
            def convert_ids_to_tokens(self, ids):
                return [f"<tok_{i}>" for i in ids]
        tokenizer = FallbackTokenizer()
        logger.info("Using fallback tokenizer.")

    # Ensure layers is a list (if provided)
    if layers is not None and not isinstance(layers, list):
        layers = [layers]

    # Compute cosine similarity
    results = compute_token_cosine_similarity(extractions, tokenizer, layers=layers, target_token=target_token, logger=logger)
    
    # Write results to CSV
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["token", "layer", "count", "avg_pairwise_cos_sim"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        logger.info(f"Saved cosine similarity results to {output_csv}")
    except Exception as e:
        logger.exception(f"Error saving CSV: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Cosine Similarity Analysis on LLM Inference Extractions"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .pt extraction files")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to output CSV file for cosine similarity results")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name or path to load tokenizer from (optional)")
    parser.add_argument("--layers", type=str, nargs="+", default=None,
                        help="Layer keys to include (e.g., 'layer_0' 'layer_5')")
    parser.add_argument("--target_token", type=str, default=None,
                        help="Specific token to filter (case-insensitive; optional)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level (e.g., INFO, DEBUG)")
    args = parser.parse_args()

    run_cosine_similarity_analysis(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        model_name=args.model_name,
        layers=args.layers,
        target_token=args.target_token,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()
