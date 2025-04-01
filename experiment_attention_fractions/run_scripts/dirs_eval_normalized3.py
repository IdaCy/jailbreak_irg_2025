#!/usr/bin/env python
"""
Evaluation of Attention on Extracted Harmful/Actionable Bits Across Different Prompt Types

This script performs the following:
  1. Loads PT extraction files (output from your inference runs) from a specified directory.
     Each extraction file contains, among other things, the attention maps and input_ids.
  2. Loads two JSON files:
       - One JSON has the harmful parts extracted from your prompts.
       - The other JSON has the actionable parts extracted from your prompts.
     These JSONs have the same keys as your original input (e.g., "attack", "jailbreak", etc.).
  3. For each extraction sample, using a Hugging Face tokenizer, the script decodes the input tokens.
  4. For each sample and each specified layer (e.g. "layer_0", "layer_5", etc.), it computes the fraction of attention 
     (averaged over heads and query positions) that is allocated to tokens whose text appears (via a substring match)
     in the corresponding harmful (or actionable) text.
  5. It aggregates these fractions by prompt type (inferred from the extraction file’s filename) and by layer.
  6. It outputs CSV files and produces multiple graphs comparing the attention on harmful (and actionable) bits across
     different prompt types.
     
Usage (command line):
    python evaluate_attention_on_bits.py --pt_dir path/to/pt_files \
         --harmful_json path/to/harmful.json --actionable_json path/to/actionable.json \
         --output_dir path/to/eval_outputs --harmful_csv harmful_attention.csv \
         --actionable_csv actionable_attention.csv --model_name <model_id> --layers layer_0 layer_5 layer_10 layer_15 --log_level INFO

Usage (import in Colab):
    from evaluate_attention_on_bits import run_attention_evaluation
    run_attention_evaluation(pt_dir="path/to/pt_files",
                             harmful_json="path/to/harmful.json",
                             actionable_json="path/to/actionable.json",
                             output_dir="path/to/eval_outputs",
                             harmful_csv="harmful_attention.csv",
                             actionable_csv="actionable_attention.csv",
                             model_name="<model_id>",
                             layers=["layer_0", "layer_5", "layer_10", "layer_15"],
                             log_level="INFO")
"""

import os
import glob
import json
import argparse
import logging
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Import tokenizer from Hugging Face
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# -----------------------------
# Utility functions
# -----------------------------
def load_extraction_files(pt_dir, logger):
    """
    Loads all .pt files from the specified directory and its subdirectories.
    """
    file_paths = sorted(glob.glob(os.path.join(pt_dir, "**", "*.pt"), recursive=True))
    logger.info(f"Found {len(file_paths)} PT extraction files under {pt_dir}")
    extractions = []
    for fp in file_paths:
        try:
            data = __import__("torch").load(fp, map_location="cpu")
            # Record both file name and relative folder (prompt type)
            data["source_file"] = os.path.basename(fp)
            data["source_path"] = fp  # full path for later use
            extractions.append(data)
        except Exception as e:
            logger.exception(f"Error loading {fp}: {e}")
    return extractions

def load_json_file(json_path, logger):
    """
    Loads a JSON file.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {json_path} with {len(data)} entries")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON file {json_path}: {e}")
        return None

def decode_tokens(token_ids, tokenizer):
    """
    Decodes a list of token IDs using the provided tokenizer.
    If tokenizer is unavailable, returns a fallback.
    """
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return tokenizer.convert_ids_to_tokens(token_ids)
    else:
        return [f"<tok_{i}>" for i in token_ids]

def compute_attention_on_bit(extraction, tokenizer, bit_texts, layers, logger):
    """
    For one extraction (which may contain multiple samples in a batch),
    computes for each sample and each specified layer the fraction of total attention
    that is directed to tokens that are flagged as belonging to the bit (harmful or actionable).

    Args:
      extraction: dict from a PT file (must have "input_ids" and "attentions").
      tokenizer: tokenizer for decoding token IDs.
      bit_texts: a list of strings (one per sample) representing the harmful or actionable text for each prompt.
      layers: list of layer keys to consider (e.g. ["layer_0", "layer_5", ...]).
      logger: logger for messages.

    Returns:
      A list of dictionaries with keys: sample_index, layer, fraction_bit, source_file.
    """
    results = []
    input_ids = extraction.get("input_ids")  # Tensor of shape [batch, seq_len]
    attentions = extraction.get("attentions")  # dict: layer -> Tensor [batch, n_heads, seq_len, seq_len]
    if input_ids is None or attentions is None:
        logger.error("Extraction missing input_ids or attentions.")
        return results
    batch_size, seq_len = input_ids.shape
    # Process each sample in the batch
    for b in range(batch_size):
        # Get token IDs and decode them
        token_ids = input_ids[b].tolist()
        tokens = decode_tokens(token_ids, tokenizer)
        # Get the bit text for this sample (if available)
        # Assume bit_texts is a list and index b corresponds to this sample
        bit_text = bit_texts[b].lower() if b < len(bit_texts) else ""
        # Create a boolean mask: for each token, is it in the bit_text?
        # A simple heuristic: if the token (after stripping non-alphanumerics) appears in bit_text.
        mask = [1 if token.lower().strip(" ,.!?") in bit_text else 0 for token in tokens]
        # For each specified layer, compute fraction of attention on the bit tokens.
        for layer in layers:
            if layer not in attentions:
                continue
            # Get the attention tensor for this layer: shape [batch, n_heads, seq_len, seq_len]
            attn_tensor = attentions[layer][b].float().numpy()  # shape [n_heads, seq_len, seq_len]
            # Average attention over heads => shape [seq_len, seq_len]
            avg_attn = np.mean(attn_tensor, axis=0)
            # For each query token (row), compute sum of attention on tokens flagged by mask.
            total_attn = np.sum(avg_attn)
            bit_attn = 0.0
            for q in range(seq_len):
                # Sum attention over key tokens that are flagged (mask==1)
                bit_attn += np.sum(avg_attn[q] * np.array(mask))
            num_flagged_tokens = sum(mask)
            fraction = (bit_attn / (total_attn + 1e-9)) / (num_flagged_tokens + 1e-9)

            results.append({
                "sample_index": b,
                "layer": layer,
                "fraction_bit": fraction
            })
    # Annotate with the source file name for later grouping (inference file name encodes prompt type)
    for r in results:
        r["source_file"] = extraction.get("source_file", "unknown")
    return results

def run_attention_extraction_normalized(tokenizer,
                             extraction_base_dir,
                             prompt_types,
                             harmful_json,
                             actionable_json,
                             output_dir,
                             layers=None,
                             log_level="INFO"):

    logger = logging.getLogger("CrossPromptAttentionEval")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if layers is None:
        layers = ["layer_0", "layer_5", "layer_10", "layer_15"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    prompt_types_list = prompt_types.strip().split()

    harmful_data = load_json_file(harmful_json, logger)
    actionable_data = load_json_file(actionable_json, logger)

    harmful_prompts = {pt: [] for pt in prompt_types_list}
    actionable_prompts = {pt: [] for pt in prompt_types_list}

    for entry in harmful_data:
        for pt in prompt_types_list:
            harmful_prompts[pt].append(entry.get(pt, ""))

    for entry in actionable_data:
        for pt in prompt_types_list:
            actionable_prompts[pt].append(entry.get(pt, ""))

    all_harmful_results = []
    all_actionable_results = []

    for prompt_type in prompt_types_list:
        pt_dir = os.path.join(extraction_base_dir, prompt_type)
        extractions = load_extraction_files(pt_dir, logger)

        logger.info(f"Analyzing prompt type: {prompt_type} with {len(extractions)} extraction files.")

        harmful_texts = harmful_prompts[prompt_type]
        actionable_texts = actionable_prompts[prompt_type]

        for extraction in extractions:
            harm_res = compute_attention_on_bit(extraction, tokenizer, harmful_texts, layers, logger)
            for r in harm_res:
                r["prompt_type"] = prompt_type
            all_harmful_results.extend(harm_res)

            action_res = compute_attention_on_bit(extraction, tokenizer, actionable_texts, layers, logger)
            for r in action_res:
                r["prompt_type"] = prompt_type
            all_actionable_results.extend(action_res)

    # Create and save CSVs
    harmful_df = pd.DataFrame(all_harmful_results)
    harmful_csv_path = os.path.join(output_dir, "harmful_attention_comparison.csv")
    harmful_df.to_csv(harmful_csv_path, index=False)
    logger.info(f"Saved harmful attention CSV to: {harmful_csv_path}")

    actionable_df = pd.DataFrame(all_actionable_results)
    actionable_csv_path = os.path.join(output_dir, "actionable_attention_comparison.csv")
    actionable_df.to_csv(actionable_csv_path, index=False)
    logger.info(f"Saved actionable attention CSV to: {actionable_csv_path}")

    # Aggregate results
    def aggregate_results(df, bit_label):
        agg = df.groupby(["prompt_type", "layer"]).agg(
            mean_fraction=("fraction_bit", "mean"),
            std_fraction=("fraction_bit", "std"),
            count=("fraction_bit", "count")
        ).reset_index()
        agg["bit_type"] = bit_label
        return agg

    agg_harmful = aggregate_results(harmful_df, "harmful")
    agg_actionable = aggregate_results(actionable_df, "actionable")

    agg_all = pd.concat([agg_harmful, agg_actionable])
    agg_csv_path = os.path.join(output_dir, "aggregate_attention_comparison.csv")
    agg_all.to_csv(agg_csv_path, index=False)
    logger.info(f"Saved aggregate attention CSV to: {agg_csv_path}")

    # Plot comparison
    def plot_attention_comparison(agg_df, bit_label):
        df = agg_df[agg_df["bit_type"] == bit_label]
        plt.figure(figsize=(10, 6))
        layers_unique = sorted(df["layer"].unique())
        prompt_types_unique = sorted(df["prompt_type"].unique())
        x = np.arange(len(layers_unique))
        width = 0.8 / len(prompt_types_unique)

        for i, pt in enumerate(prompt_types_unique):
            sub_df = df[df["prompt_type"] == pt].set_index("layer").reindex(layers_unique)
            plt.bar(x + i * width, sub_df["mean_fraction"], width, yerr=sub_df["std_fraction"], capsize=5, label=pt)

        plt.xticks(x + width * (len(prompt_types_unique) - 1) / 2, layers_unique)
        plt.xlabel("Layer")
        plt.ylabel(f"Mean Attention on {bit_label.capitalize()} Bits")
        plt.title(f"Comparison of Attention on {bit_label.capitalize()} Bits Across Prompt Types")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"attention_comparison_{bit_label}.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved plot for {bit_label} attention comparison to: {plot_path}")

    plot_attention_comparison(agg_all, "harmful")
    plot_attention_comparison(agg_all, "actionable")

    logger.info("Cross-prompt attention evaluation complete.")
