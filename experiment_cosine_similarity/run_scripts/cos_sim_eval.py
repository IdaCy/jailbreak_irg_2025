#!/usr/bin/env python
"""
Evaluation of Cosine Similarity Results Across Multiple Jailbreak Extractions

This script loads all cosine similarity CSV result files from a given directory,
aggregates them (adding a 'jailbreak_type' column inferred from each filename),
computes summary statistics (mean, median, std, count) grouped by jailbreak type and layer,
and produces several graphs:
  - Bar plots of average cosine similarity (with error bars) per layer for each jailbreak type.
  - Histograms of the cosine similarity distributions per jailbreak type.
  - A box plot of cosine similarity values across layers (all types combined).

Usage (command line):
    python evaluate_cosine_similarity.py --input_dir path/to/csv_results \
         --output_dir path/to/eval_outputs --summary_csv aggregate_summary.csv --log_level INFO

Usage (import in Colab):
    from evaluate_cosine_similarity import run_evaluation
    run_evaluation(input_dir="path/to/csv_results",
                   output_dir="path/to/eval_outputs",
                   summary_csv="aggregate_summary.csv",
                   log_level="INFO")
"""

import os
import glob
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def load_all_results(input_dir, logger):
    """
    Loads all CSV files from input_dir and adds a column 'jailbreak_type' (inferred from filename).
    """
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {input_dir}")
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Infer jailbreak type from the file name.
            # For example, if filename is "cosine_similarity_foo.csv", set type to "foo".
            base = os.path.basename(f)
            parts = base.split("_")
            if len(parts) > 1:
                jb_type = parts[-1].replace(".csv", "")
            else:
                jb_type = base.replace(".csv", "")
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
    Computes summary statistics (mean, median, std, count) of avg_pairwise_cos_sim,
    grouped by jailbreak_type and layer.
    """
    logger.info("Computing summary statistics...")
    summary = df.groupby(["jailbreak_type", "layer"]).agg(
        mean_cos_sim=("avg_pairwise_cos_sim", "mean"),
        median_cos_sim=("avg_pairwise_cos_sim", "median"),
        std_cos_sim=("avg_pairwise_cos_sim", "std"),
        count=("avg_pairwise_cos_sim", "count")
    ).reset_index()
    return summary

def plot_summary_statistics(summary, output_dir, logger):
    """
    Produces bar plots (with error bars) of mean cosine similarity per layer for each jailbreak type.
    """
    jailbreak_types = summary["jailbreak_type"].unique()
    for jb_type in jailbreak_types:
        df_jb = summary[summary["jailbreak_type"] == jb_type]
        plt.figure(figsize=(8, 6))
        # Plot mean with error bars (std deviation)
        plt.bar(df_jb["layer"], df_jb["mean_cos_sim"], yerr=df_jb["std_cos_sim"], capsize=5)
        plt.title(f"Avg Cosine Similarity per Layer - {jb_type}")
        plt.xlabel("Layer")
        plt.ylabel("Mean Cosine Similarity")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"bar_avg_cosine_{jb_type}.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved bar plot: {out_path}")

def plot_histograms(df, output_dir, logger):
    """
    Produces histograms of avg_pairwise_cos_sim for each jailbreak type.
    """
    jailbreak_types = df["jailbreak_type"].unique()
    for jb_type in jailbreak_types:
        df_jb = df[df["jailbreak_type"] == jb_type]
        plt.figure(figsize=(8, 6))
        plt.hist(df_jb["avg_pairwise_cos_sim"].dropna(), bins=20, alpha=0.75)
        plt.title(f"Cosine Similarity Histogram - {jb_type}")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"histogram_cosine_{jb_type}.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved histogram: {out_path}")

def plot_boxplot_all_layers(df, output_dir, logger):
    """
    Produces a box plot for cosine similarity values per layer (aggregated across jailbreak types).
    """
    # Group values by layer.
    layers = sorted(df["layer"].unique())
    data_by_layer = []
    for layer in layers:
        values = df[df["layer"] == layer]["avg_pairwise_cos_sim"].dropna().values
        data_by_layer.append(values)
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_by_layer, labels=layers)
    plt.title("Distribution of Cosine Similarity per Layer (All Jailbreak Types)")
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "boxplot_cosine_all_layers.png")
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved box plot: {out_path}")

def run_evaluation(input_dir, output_dir, summary_csv, log_level="INFO"):
    """
    Main evaluation function:
      - Loads all CSV result files from input_dir.
      - Aggregates the data and computes summary statistics.
      - Saves the summary CSV and produces several plots.
    """
    logger = logging.getLogger("CosineEval")
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
        logger.error("No data loaded; exiting.")
        return

    summary = compute_summary_statistics(df, logger)
    summary.to_csv(summary_csv, index=False)
    logger.info(f"Saved aggregated summary CSV to: {summary_csv}")

    logger.info("Plotting bar charts of summary statistics...")
    plot_summary_statistics(summary, output_dir, logger)
    logger.info("Plotting histograms...")
    plot_histograms(df, output_dir, logger)
    logger.info("Plotting box plot (all layers)...")
    plot_boxplot_all_layers(df, output_dir, logger)

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
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level (e.g., INFO, DEBUG)")
    args = parser.parse_args()

    run_evaluation(args.input_dir, args.output_dir, args.summary_csv, log_level=args.log_level)

if __name__ == "__main__":
    main()
