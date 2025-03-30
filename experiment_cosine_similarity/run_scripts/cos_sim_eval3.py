#!/usr/bin/env python
"""
Evaluation of Cosine Similarity Results Across Multiple Jailbreak Extractions

This script loads all cosine similarity CSV result files (generated separately for
plain "attack" and various jailbreak prompt types) from a given directory.
It then performs extensive evaluations including:

1. Standard summary statistics (mean, median, std, count) grouped by jailbreak type and layer.
2. Baseline comparisons:
   - For a user-specified baseline type (default "attack"), it computes for each layer the
     difference in mean cosine similarity, percent difference, and (optionally) a t-test p-value
     comparing baseline with each other jailbreak type.
3. Visualizations:
   - Grouped bar plots with error bars (mean and std) for each jailbreak type by layer.
   - Line plots comparing mean cosine similarity across layers for all jailbreak types.
   - Heatmaps of mean cosine similarity and of differences relative to the baseline.
   - Box plots of cosine similarity distributions by layer and jailbreak type.

These evaluations help reveal:
  - Global differences in internal representation consistency between conditions.
  - Which layers show the most significant shifts.
  - Token-specific and distributional changes induced by different jailbreak prompts.

Usage (command line):
    python evaluate_jailbreak_comparisons.py --input_dir path/to/csv_results \
         --output_dir path/to/eval_outputs --summary_csv aggregate_summary.csv \
         [--baseline attack] [--log_level INFO]

Usage (import in Colab):
    from evaluate_jailbreak_comparisons import run_evaluation
    run_evaluation(input_dir="path/to/csv_results",
                   output_dir="path/to/eval_outputs",
                   summary_csv="aggregate_summary.csv",
                   baseline="attack",
                   log_level="INFO")
"""

import os
import glob
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

def load_all_results(input_dir, logger):
    """
    Loads all CSV files from input_dir and adds a column 'jailbreak_type' inferred from the filename.
    """
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {input_dir}")
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Infer jailbreak type from filename.
            # e.g., "cosine_similarity_attack.csv" yields "attack"
            base = os.path.basename(f)
            parts = base.split("_")
            if len(parts) > 1:
                jb_type = parts[-1].replace(".csv", "").strip()
            else:
                jb_type = base.replace(".csv", "").strip()
            df["jailbreak_type"] = jb_type
            df["source_file"] = base
            frames.append(df)
        except Exception as e:
            logger.error(f"Error loading {f}: {e}")
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        return all_df
    else:
        return pd.DataFrame()

def compute_summary_statistics(df, logger):
    """
    Computes summary statistics (mean, median, std, count) for avg_pairwise_cos_sim grouped by jailbreak_type and layer.
    """
    logger.info("Computing summary statistics...")
    summary = df.groupby(["jailbreak_type", "layer"]).agg(
        mean_cos_sim=("avg_pairwise_cos_sim", "mean"),
        median_cos_sim=("avg_pairwise_cos_sim", "median"),
        std_cos_sim=("avg_pairwise_cos_sim", "std"),
        count=("avg_pairwise_cos_sim", "count")
    ).reset_index()
    return summary

def compute_baseline_comparisons(df, baseline, logger):
    """
    For each layer, compares each jailbreak type to the baseline.
    For each non-baseline jailbreak type and each layer, computes:
      - The difference in mean cosine similarity from the baseline.
      - The percent difference relative to the baseline.
      - A t-test p-value comparing the distributions (if there are enough samples).
    
    Returns a DataFrame summarizing these comparisons.
    """
    logger.info(f"Computing baseline comparisons using baseline='{baseline}'...")
    comp_results = []
    
    # Split data by layer for statistical comparison
    layers = df["layer"].unique()
    for layer in layers:
        df_layer = df[df["layer"] == layer]
        df_baseline = df_layer[df_layer["jailbreak_type"] == baseline]
        if df_baseline.empty:
            logger.warning(f"No baseline data for layer {layer}. Skipping comparisons for this layer.")
            continue
        baseline_vals = df_baseline["avg_pairwise_cos_sim"].dropna().values
        base_mean = np.mean(baseline_vals)
        
        for jb_type in df_layer["jailbreak_type"].unique():
            if jb_type == baseline:
                continue
            df_jb = df_layer[df_layer["jailbreak_type"] == jb_type]
            jb_vals = df_jb["avg_pairwise_cos_sim"].dropna().values
            if len(jb_vals) < 2 or len(baseline_vals) < 2:
                p_val = np.nan
            else:
                # Perform two-sample t-test
                _, p_val = ttest_ind(baseline_vals, jb_vals, equal_var=False)
            jb_mean = np.mean(jb_vals)
            diff = jb_mean - base_mean
            pct_diff = (diff / base_mean * 100) if base_mean != 0 else np.nan
            comp_results.append({
                "layer": layer,
                "jailbreak_type": jb_type,
                "baseline_mean": base_mean,
                "jb_mean": jb_mean,
                "mean_difference": diff,
                "percent_difference": pct_diff,
                "baseline_count": len(baseline_vals),
                "jb_count": len(jb_vals),
                "p_value": p_val
            })
    return pd.DataFrame(comp_results)

def plot_grouped_bar(summary, output_dir, logger):
    """
    Plots a grouped bar chart showing mean cosine similarity for each layer and jailbreak type.
    """
    pivot = summary.pivot(index="layer", columns="jailbreak_type", values="mean_cos_sim")
    pivot.plot(kind="bar", figsize=(10, 6), yerr=summary.pivot(index="layer", columns="jailbreak_type", values="std_cos_sim"))
    plt.title("Mean Cosine Similarity per Layer by Jailbreak Type")
    plt.xlabel("Layer")
    plt.ylabel("Mean Cosine Similarity")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "grouped_bar_mean_cosine.png")
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved grouped bar plot: {out_path}")

def plot_line_chart(summary, output_dir, logger):
    """
    Plots a line chart comparing mean cosine similarity across layers for each jailbreak type.
    """
    plt.figure(figsize=(10, 6))
    jailbreak_types = summary["jailbreak_type"].unique()
    for jb in jailbreak_types:
        df_jb = summary[summary["jailbreak_type"] == jb].sort_values(by="layer")
        plt.plot(df_jb["layer"], df_jb["mean_cos_sim"], marker="o", label=jb)
    plt.title("Mean Cosine Similarity Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Mean Cosine Similarity")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, "lineplot_mean_cosine.png")
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved line plot: {out_path}")

def plot_heatmap(summary, output_dir, logger):
    """
    Produces a heatmap of mean cosine similarity with layers on one axis and jailbreak types on the other.
    """
    pivot = summary.pivot(index="jailbreak_type", columns="layer", values="mean_cos_sim")
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot, aspect="auto", cmap="viridis")
    plt.colorbar(label="Mean Cosine Similarity")
    plt.xticks(ticks=range(len(pivot.columns)), labels=pivot.columns)
    plt.yticks(ticks=range(len(pivot.index)), labels=pivot.index)
    plt.title("Heatmap of Mean Cosine Similarity")
    plt.xlabel("Layer")
    plt.ylabel("Jailbreak Type")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "heatmap_mean_cosine.png")
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved heatmap: {out_path}")

def plot_boxplots(df, output_dir, logger):
    """
    Produces box plots of cosine similarity values per layer for each jailbreak type.
    """
    layers = sorted(df["layer"].unique())
    for layer in layers:
        plt.figure(figsize=(10, 6))
        data = []
        labels = []
        for jb in df["jailbreak_type"].unique():
            vals = df[(df["layer"] == layer) & (df["jailbreak_type"] == jb)]["avg_pairwise_cos_sim"].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(jb)
        if data:
            plt.boxplot(data, labels=labels)
            plt.title(f"Distribution of Cosine Similarity at {layer}")
            plt.xlabel("Jailbreak Type")
            plt.ylabel("Cosine Similarity")
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"boxplot_{layer}.png")
            plt.savefig(out_path)
            plt.close()
            logger.info(f"Saved box plot for {layer}: {out_path}")

def run_evaluation(input_dir, output_dir, summary_csv, comp_csv, baseline="attack", log_level="INFO"):
    """
    Main evaluation function:
      - Loads all cosine similarity CSV result files.
      - Computes summary statistics grouped by jailbreak type and layer.
      - Performs baseline comparisons (differences, percent differences, and t-test p-values).
      - Saves both the aggregate summary and the baseline comparison table as CSVs.
      - Produces multiple plots (grouped bar, line chart, heatmap, and box plots).
    """
    logger = logging.getLogger("JailbreakCosineEval")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading all cosine similarity CSV files...")
    df = load_all_results(input_dir, logger)
    if df.empty:
        logger.error("No data loaded; exiting evaluation.")
        return

    # Compute overall summary statistics.
    summary = compute_summary_statistics(df, logger)
    summary.to_csv(summary_csv, index=False)
    logger.info(f"Saved aggregate summary CSV to: {summary_csv}")

    # Compute baseline comparisons (if baseline exists in the data).
    baseline_df = df[df["jailbreak_type"] == baseline]
    if baseline_df.empty:
        logger.warning(f"Baseline type '{baseline}' not found. Skipping baseline comparisons.")
    else:
        comp_df = compute_baseline_comparisons(df, baseline, logger)
        comp_df.to_csv(comp_csv, index=False)
        logger.info(f"Saved baseline comparison CSV to: {comp_csv}")

    # Generate plots.
    plot_grouped_bar(summary, output_dir, logger)
    plot_line_chart(summary, output_dir, logger)
    plot_heatmap(summary, output_dir, logger)
    plot_boxplots(df, output_dir, logger)

    logger.info("Evaluation complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Cosine Similarity Results Across Multiple Jailbreak Extractions"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing cosine similarity CSV result files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation plots and outputs")
    parser.add_argument("--summary_csv", type=str, required=True,
                        help="Path to save the aggregated summary CSV")
    parser.add_argument("--comp_csv", type=str, default="baseline_comparison.csv",
                        help="Path to save the baseline comparison CSV")
    parser.add_argument("--baseline", type=str, default="attack",
                        help="Baseline jailbreak type for comparisons (default 'attack')")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level (e.g., INFO, DEBUG)")
    args = parser.parse_args()

    run_evaluation(args.input_dir, args.output_dir, args.summary_csv, args.comp_csv,
                   baseline=args.baseline, log_level=args.log_level)

if __name__ == "__main__":
    main()
