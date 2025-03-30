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
    Loads all .pt files from the specified directory.
    """
    file_paths = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    logger.info(f"Found {len(file_paths)} PT extraction files in {pt_dir}")
    extractions = []
    for fp in file_paths:
        try:
            data = __import__("torch").load(fp, map_location="cpu")
            # Also record the source file name for later inference of prompt type.
            data["source_file"] = os.path.basename(fp)
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
            fraction = bit_attn / (total_attn + 1e-9)
            results.append({
                "sample_index": b,
                "layer": layer,
                "fraction_bit": fraction
            })
    # Annotate with the source file name for later grouping (inference file name encodes prompt type)
    for r in results:
        r["source_file"] = extraction.get("source_file", "unknown")
    return results

# -----------------------------
# Main evaluation functions
# -----------------------------
def run_attention_evaluation(pt_dir, harmful_json, actionable_json, output_dir,
                             harmful_csv, actionable_csv, model_name=None,
                             layers=None, log_level="INFO"):
    """
    Main function that:
      - Loads PT extraction files.
      - Loads harmful and actionable JSONs.
      - For each extraction file, infers the prompt type from the filename (expects filename to include it, e.g. "cosine_similarity_attack.pt").
      - For each extraction, matches the samples (by order) with the corresponding harmful and actionable texts,
        based on the prompt type key in the JSONs.
      - Computes, per sample and layer, the fraction of attention that is on harmful and actionable bits.
      - Aggregates results across files and outputs CSVs and plots comparing the different prompt types.
    
    Args:
      pt_dir: Directory containing the .pt extraction files.
      harmful_json: Path to the JSON file with harmful bits (same structure as your input JSON).
      actionable_json: Path to the JSON file with actionable bits.
      output_dir: Directory where CSVs and graphs will be saved.
      harmful_csv: Output CSV filename for harmful attention results.
      actionable_csv: Output CSV filename for actionable attention results.
      model_name: Hugging Face model identifier to load the tokenizer.
      layers: List of layer keys to analyze (e.g., ["layer_0", "layer_5", ...]).
      log_level: Logging level.
    """
    logger = logging.getLogger("AttentionOnBitsEval")
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
    
    logger.info("Loading PT extraction files...")
    extractions = load_extraction_files(pt_dir, logger)
    if not extractions:
        logger.error("No PT files loaded. Exiting evaluation.")
        return
    
    logger.info("Loading harmful and actionable JSON files...")
    harmful_data = load_json_file(harmful_json, logger)
    actionable_data = load_json_file(actionable_json, logger)
    if harmful_data is None or actionable_data is None:
        logger.error("Error loading JSON data. Exiting evaluation.")
        return

    # where harmful_data and actionable_data are lists of dicts (one per prompt),
    # and that the keys in these dicts correspond to the prompt types (e.g., "attack", "jailbreak", etc.).
    # We will build dictionaries mapping prompt type to list of texts.
    def build_prompt_dict(json_data):
        prompt_dict = defaultdict(list)
        for entry in json_data:
            # For each entry, add the value for each key (except element_id, topic, etc.)
            for key, value in entry.items():
                if key in ["element_id", "topic"]:
                    continue
                # Append the prompt text (if non-empty)
                prompt_dict[key].append(value)
        return prompt_dict

    harmful_prompts = build_prompt_dict(harmful_data)
    actionable_prompts = build_prompt_dict(actionable_data)
    
    # Results lists for harmful and actionable attention fractions.
    harmful_results = []
    actionable_results = []
    
    # Process each PT extraction file.
    # We infer the prompt type from the filename. For example, if filename is "cosine_similarity_attack.pt",
    # then prompt_type is "attack".
    for extraction in extractions:
        source_file = extraction.get("source_file", "unknown")
        parts = source_file.split("_")
        if len(parts) > 1:
            prompt_type = parts[-1].split(".")[0].strip()  # e.g., "attack" or "jailbreak" etc.
        else:
            prompt_type = source_file.replace(".pt", "").strip()
        logger.info(f"Processing file {source_file} (prompt type: {prompt_type})")
        
        # Get corresponding harmful and actionable texts for this prompt type.
        # If not found, default to an empty string list.
        harmful_texts = harmful_prompts.get(prompt_type, [""] * extraction["input_ids"].shape[0])
        actionable_texts = actionable_prompts.get(prompt_type, [""] * extraction["input_ids"].shape[0])
        
        # Compute attention fractions for harmful bits.
        harm_res = compute_attention_on_bit(extraction, tokenizer=AutoTokenizer.from_pretrained(model_name) if model_name and AutoTokenizer else None,
                                              bit_texts=harmful_texts, layers=layers, logger=logger)
        # Tag each result with the prompt_type.
        for r in harm_res:
            r["prompt_type"] = prompt_type
        harmful_results.extend(harm_res)
        
        # Compute attention fractions for actionable bits.
        act_res = compute_attention_on_bit(extraction, tokenizer=AutoTokenizer.from_pretrained(model_name) if model_name and AutoTokenizer else None,
                                             bit_texts=actionable_texts, layers=layers, logger=logger)
        for r in act_res:
            r["prompt_type"] = prompt_type
        actionable_results.extend(act_res)
    
    # Save results as CSV files.
    harmful_df = pd.DataFrame(harmful_results)
    harmful_csv_path = os.path.join(output_dir, harmful_csv)
    harmful_df.to_csv(harmful_csv_path, index=False)
    logger.info(f"Saved harmful attention CSV to: {harmful_csv_path}")
    
    actionable_df = pd.DataFrame(actionable_results)
    actionable_csv_path = os.path.join(output_dir, actionable_csv)
    actionable_df.to_csv(actionable_csv_path, index=False)
    logger.info(f"Saved actionable attention CSV to: {actionable_csv_path}")
    
    # Aggregate results by prompt type and layer (compute mean, std, count)
    def aggregate_results(df, bit_label):
        agg = df.groupby(["prompt_type", "layer"]).agg(
            mean_fraction=( "fraction_bit", "mean" ),
            std_fraction=("fraction_bit", "std"),
            count=("fraction_bit", "count")
        ).reset_index()
        agg["bit_type"] = bit_label
        return agg
    
    agg_harmful = aggregate_results(harmful_df, "harmful")
    agg_actionable = aggregate_results(actionable_df, "actionable")
    agg_all = pd.concat([agg_harmful, agg_actionable], ignore_index=True)
    agg_csv = os.path.join(output_dir, "aggregate_attention_comparison.csv")
    agg_all.to_csv(agg_csv, index=False)
    logger.info(f"Saved aggregate attention comparison CSV to: {agg_csv}")
    
    # Create plots to compare attention on harmful and actionable bits across prompt types.
    def plot_aggregated_bar(agg, bit_label, output_dir, logger):
        df = agg[agg["bit_type"] == bit_label]
        plt.figure(figsize=(10, 6))
        # Create a grouped bar plot: x-axis: layer, bars for each prompt_type.
        prompt_types = sorted(df["prompt_type"].unique())
        layers_unique = sorted(df["layer"].unique())
        width = 0.8 / len(prompt_types)
        x = np.arange(len(layers_unique))
        for i, pt in enumerate(prompt_types):
            sub_df = df[df["prompt_type"] == pt].set_index("layer").reindex(layers_unique)
            plt.bar(x + i * width, sub_df["mean_fraction"], width=width,
                    yerr=sub_df["std_fraction"], capsize=5, label=pt)
        plt.xticks(x + width * (len(prompt_types) - 1) / 2, layers_unique)
        plt.xlabel("Layer")
        plt.ylabel(f"Mean Fraction Attention on {bit_label.capitalize()} Bits")
        plt.title(f"Comparison of Attention on {bit_label.capitalize()} Bits Across Prompt Types")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"grouped_bar_{bit_label}.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved grouped bar plot for {bit_label} to: {out_path}")
    
    def plot_boxplots_by_prompt(df, bit_label, output_dir, logger):
        plt.figure(figsize=(10, 6))
        prompt_types = sorted(df["prompt_type"].unique())
        data = []
        for pt in prompt_types:
            vals = df[df["prompt_type"] == pt]["fraction_bit"].dropna().values
            data.append(vals)
        plt.boxplot(data, labels=prompt_types)
        plt.xlabel("Prompt Type")
        plt.ylabel(f"Fraction Attention on {bit_label.capitalize()} Bits")
        plt.title(f"Box Plot of Attention on {bit_label.capitalize()} Bits by Prompt Type")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"boxplot_{bit_label}.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved box plot for {bit_label} to: {out_path}")
    
    # Generate plots for harmful bits
    plot_aggregated_bar(agg_all, "harmful", output_dir, logger)
    plot_boxplots_by_prompt(harmful_df, "harmful", output_dir, logger)
    # Generate plots for actionable bits
    plot_aggregated_bar(agg_all, "actionable", output_dir, logger)
    plot_boxplots_by_prompt(actionable_df, "actionable", output_dir, logger)
    
    logger.info("Attention-on-bits evaluation complete.")

# -----------------------------
# Main entry point (CLI)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate attention on harmful and actionable bits across prompt types"
    )
    parser.add_argument("--pt_dir", type=str, required=True,
                        help="Directory containing PT extraction (.pt) files")
    parser.add_argument("--harmful_json", type=str, required=True,
                        help="Path to the JSON file containing harmful bits")
    parser.add_argument("--actionable_json", type=str, required=True,
                        help="Path to the JSON file containing actionable bits")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation outputs (CSVs and plots)")
    parser.add_argument("--harmful_csv", type=str, default="harmful_attention.csv",
                        help="Filename for harmful attention CSV output")
    parser.add_argument("--actionable_csv", type=str, default="actionable_attention.csv",
                        help="Filename for actionable attention CSV output")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Hugging Face model name or local path for loading tokenizer")
    parser.add_argument("--layers", type=str, nargs="+", default=["layer_0", "layer_5", "layer_10", "layer_15"],
                        help="List of layer keys to analyze (e.g., layer_0 layer_5 ...)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level (e.g., INFO, DEBUG)")
    args = parser.parse_args()
    
    run_attention_evaluation(pt_dir=args.pt_dir,
                             harmful_json=args.harmful_json,
                             actionable_json=args.actionable_json,
                             output_dir=args.output_dir,
                             harmful_csv=args.harmful_csv,
                             actionable_csv=args.actionable_csv,
                             model_name=args.model_name,
                             layers=args.layers,
                             log_level=args.log_level)

if __name__ == "__main__":
    main()
