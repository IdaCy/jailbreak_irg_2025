#!/usr/bin/env python
import os
import json
import torch
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

def find_subsequence_indices(full_ids, sub_ids):
    """
    (Legacy function kept for reference.)
    Find the first occurrence of a subsequence (sub_ids) within a larger sequence (full_ids).
    Returns a list of indices if found, or None otherwise.
    """
    n = len(full_ids)
    m = len(sub_ids)
    for i in range(n - m + 1):
        if full_ids[i:i+m] == sub_ids:
            return list(range(i, i+m))
    return None

def get_mask_indices_for_bit(sample_ids, tokenizer, bit_text, logger=None):
    """
    Decodes the sample token IDs and returns a list of indices for which the decoded token,
    after lowercasing and stripping punctuation and common BPE markers (e.g. 'Ġ', '▁'),
    appears as a substring in any of the words from bit_text.
    """
    # Decode tokens using the tokenizer.
    tokens = tokenizer.convert_ids_to_tokens(sample_ids)
    indices = []
    # Clean and split the target bit_text.
    bit_text_clean = bit_text.lower().strip(" ,.!?\"'")
    bit_words = bit_text_clean.split()
    if logger:
        logger.debug(f"Target bit text (clean): '{bit_text_clean}'")
        logger.debug(f"Target bit words: {bit_words}")
    for idx, token in enumerate(tokens):
        # Remove common BPE markers and punctuation.
        token_str = token.lower().replace("Ġ", "").replace("▁", "").strip(" ,.!?\"'")
        if logger:
            logger.debug(f"Token {idx}: original='{token}', cleaned='{token_str}'")
        # Check if token_str appears in any of the bit_words.
        if token_str and any(token_str in word for word in bit_words):
            indices.append(idx)
    if logger:
        logger.debug(f"Matched indices: {indices}")
    return indices if indices else None

def compute_attention_on_span(attentions, sample_idx, span_indices):
    """
    Given the dictionary 'attentions' (keys like "layer_0", etc.) from a PT file,
    and a list of token indices (span_indices), compute the average attention value
    that all tokens (queries) give to those target positions.
    """
    attn_vals = []
    for layer_key, attn_tensor in attentions.items():
        try:
            sample_attn = attn_tensor[sample_idx]  # shape: (num_heads, seq_length, seq_length)
        except Exception as e:
            continue
        if span_indices is not None and len(span_indices) > 0:
            target_attn = sample_attn[:, :, span_indices]
            attn_mean = target_attn.mean().item()
            attn_vals.append(attn_mean)
    if attn_vals:
        return sum(attn_vals) / len(attn_vals)
    else:
        return 0.0

def analyze_attention(overall_dir, harmful_json_path, actionable_json_path, tokenizer_name, logger=None):
    """
    Analyze the attention weights on harmful and actionable substrings for a set of PT files.
    Uses a mask-based approach: For each sample, decodes the tokens and flags those that
    appear (after lowercasing, stripping punctuation, and removing BPE markers) in the target substring.
    """
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("analyze_attention")
    
    # Load JSON mappings.
    with open(harmful_json_path, "r", encoding="utf-8") as f:
        harmful_data = json.load(f)
    with open(actionable_json_path, "r", encoding="utf-8") as f:
        actionable_data = json.load(f)
    
    logger.info(f"Loaded {len(harmful_data)} harmful entries and {len(actionable_data)} actionable entries.")
    
    # Build maps: element_id -> {prompt_type: substring, ...}
    harmful_map = {entry["element_id"]: entry for entry in harmful_data}
    actionable_map = {entry["element_id"]: entry for entry in actionable_data}
    
    # Accept either a tokenizer name (string) or a tokenizer object.
    if hasattr(tokenizer_name, "encode"):
        tokenizer = tokenizer_name
        logger.info("Using provided tokenizer object.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        logger.info(f"Loaded tokenizer from '{tokenizer_name}'.")
    
    # Initialize results container.
    results = {}
    
    # Iterate over each subdirectory (prompt type) in the overall directory.
    for prompt_type in os.listdir(overall_dir):
        type_dir = os.path.join(overall_dir, prompt_type)
        if not os.path.isdir(type_dir):
            logger.debug(f"Skipping non-directory {type_dir}.")
            continue
        logger.info(f"Processing prompt type: {prompt_type}")
        results.setdefault(prompt_type, {"direct": {"harmful": [], "actionable": []},
                                           "refusing": {"harmful": [], "actionable": []}})
        
        # Process each PT file in this subdirectory.
        pt_files = [f for f in os.listdir(type_dir) if f.endswith(".pt")]
        logger.info(f"Found {len(pt_files)} PT files in {prompt_type}.")
        for pt_file in pt_files:
            pt_path = os.path.join(type_dir, pt_file)
            try:
                data = torch.load(pt_path, map_location="cpu")
            except Exception as ex:
                logger.error(f"Error loading {pt_path}: {ex}")
                continue
            
            required_keys = ["input_ids", "attentions", "response_types", "original_indices"]
            if not all(key in data for key in required_keys):
                logger.warning(f"File {pt_path} missing required keys. Skipping.")
                continue
            
            input_ids = data["input_ids"]
            attentions = data["attentions"]
            response_types = data["response_types"]
            original_indices = data["original_indices"]
            
            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                eid = original_indices[i]
                # Get the harmful and actionable substring for the current prompt type.
                harmful_sub = harmful_map.get(eid, {}).get(prompt_type, "")
                actionable_sub = actionable_map.get(eid, {}).get(prompt_type, "")
                
                sample_ids = input_ids[i].tolist()
                # Convert target substrings to lower-case once.
                harmful_target = harmful_sub.lower() if harmful_sub else ""
                actionable_target = actionable_sub.lower() if actionable_sub else ""
                # Get mask indices based on decoded tokens.
                harmful_indices = get_mask_indices_for_bit(sample_ids, tokenizer, harmful_target, logger) if harmful_sub else None
                actionable_indices = get_mask_indices_for_bit(sample_ids, tokenizer, actionable_target, logger) if actionable_sub else None

                if harmful_sub and harmful_indices is None:
                    logger.warning(f"Element id {eid} in file {pt_file} (prompt: {prompt_type}): Harmful substring '{harmful_sub}' not found in sample.")
                if actionable_sub and actionable_indices is None:
                    logger.warning(f"Element id {eid} in file {pt_file} (prompt: {prompt_type}): Actionable substring '{actionable_sub}' not found in sample.")
                
                harmful_attn = compute_attention_on_span(attentions, i, harmful_indices) if harmful_indices is not None else 0.0
                actionable_attn = compute_attention_on_span(attentions, i, actionable_indices) if actionable_indices is not None else 0.0
                
                resp_type_raw = response_types[i].lower().strip()
                resp_type = "refusing" if "refus" in resp_type_raw else "direct"
                
                results[prompt_type][resp_type]["harmful"].append(harmful_attn)
                results[prompt_type][resp_type]["actionable"].append(actionable_attn)
    
    # Summarize results.
    summary = {}
    for pt, resp_dict in results.items():
        summary[pt] = {}
        for resp_type, attn_dict in resp_dict.items():
            harmful_vals = attn_dict["harmful"]
            actionable_vals = attn_dict["actionable"]
            summary[pt][resp_type] = {
                "harmful_avg": np.mean(harmful_vals) if harmful_vals else 0.0,
                "actionable_avg": np.mean(actionable_vals) if actionable_vals else 0.0,
                "count": len(harmful_vals)
            }
            logger.info(f"Prompt type '{pt}' {resp_type}: Count={summary[pt][resp_type]['count']}, "
                        f"Harmful avg={summary[pt][resp_type]['harmful_avg']:.4f}, "
                        f"Actionable avg={summary[pt][resp_type]['actionable_avg']:.4f}")
    return results, summary

def plot_attention(summary, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    prompt_types = list(summary.keys())
    x = np.arange(len(prompt_types))
    width = 0.35  # width of the bars
    
    # Harmful attention data.
    direct_harmful = [summary[pt]["direct"]["harmful_avg"] for pt in prompt_types]
    refusing_harmful = [summary[pt]["refusing"]["harmful_avg"] for pt in prompt_types]
    
    fig, ax = plt.subplots()
    ax.bar(x - width/2, direct_harmful, width, label='Direct')
    ax.bar(x + width/2, refusing_harmful, width, label='Refusing')
    ax.set_ylabel('Average Attention on Harmful Substring')
    ax.set_title('Harmful Attention by Prompt Type and Response Type')
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_types)
    ax.legend()
    plt.tight_layout()
    harmful_plot_path = os.path.join(output_dir, "harmful_attention.png")
    plt.savefig(harmful_plot_path)
    plt.close(fig)
    
    # Actionable attention data.
    direct_actionable = [summary[pt]["direct"]["actionable_avg"] for pt in prompt_types]
    refusing_actionable = [summary[pt]["refusing"]["actionable_avg"] for pt in prompt_types]
    
    fig, ax = plt.subplots()
    ax.bar(x - width/2, direct_actionable, width, label='Direct')
    ax.bar(x + width/2, refusing_actionable, width, label='Refusing')
    ax.set_ylabel('Average Attention on Actionable Substring')
    ax.set_title('Actionable Attention by Prompt Type and Response Type')
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_types)
    ax.legend()
    plt.tight_layout()
    actionable_plot_path = os.path.join(output_dir, "actionable_attention.png")
    plt.savefig(actionable_plot_path)
    plt.close(fig)
    print(f"Plots saved in directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze attention on harmful and actionable substrings from PT files.")
    parser.add_argument("--overall_dir", type=str, required=True,
                        help="Overall directory containing subdirectories for each prompt type.")
    parser.add_argument("--harmful_json", type=str, required=True,
                        help="Path to the JSON file with harmful substrings.")
    parser.add_argument("--actionable_json", type=str, required=True,
                        help="Path to the JSON file with actionable substrings.")
    parser.add_argument("--tokenizer_name", type=str, required=True,
                        help="Tokenizer name or path (e.g., 'gpt2') used for tokenizing texts.")
    parser.add_argument("--plot_output_dir", type=str, default="plots",
                        help="Directory to save generated plots.")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("analyze_attention")
    
    results, summary = analyze_attention(
        overall_dir=args.overall_dir,
        harmful_json_path=args.harmful_json,
        actionable_json_path=args.actionable_json,
        tokenizer_name=args.tokenizer_name,
        logger=logger
    )
    
    print("Attention analysis summary:")
    for pt, resp in summary.items():
        print(f"Prompt type: {pt}")
        for rtype, stats in resp.items():
            print(f"  {rtype} (n={stats['count']}): Harmful avg = {stats['harmful_avg']:.4f}, "
                  f"Actionable avg = {stats['actionable_avg']:.4f}")
    
    plot_attention(summary, output_dir=args.plot_output_dir)

if __name__ == "__main__":
    main()
