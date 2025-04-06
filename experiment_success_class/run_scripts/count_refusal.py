#!/usr/bin/env python
import os
import argparse
import torch
import logging

def count_responses_in_file(file_path, logger):
    """
    Load a PT file and count 'refusing' and 'direct' responses.
    Expects the PT file to have a key 'response_types' which is a list.
    """
    try:
        data = torch.load(file_path, map_location="cpu")
    except Exception as ex:
        logger.error(f"Error loading {file_path}: {ex}")
        return (0, 0)
    
    if "response_types" not in data:
        logger.warning(f"File {file_path} does not contain 'response_types'. Skipping.")
        return (0, 0)
    
    refusing = sum(1 for r in data["response_types"] if r == "refusing")
    direct = sum(1 for r in data["response_types"] if r == "direct")
    return (refusing, direct)

def count_responses_in_folder(folder_path, logger):
    """
    For a given folder, counts 'refusing' and 'direct' responses in all PT files.
    """
    refusing_total = 0
    direct_total = 0

    files = [f for f in os.listdir(folder_path) if f.endswith(".pt")]
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        r_ref, r_dir = count_responses_in_file(file_path, logger)
        refusing_total += r_ref
        direct_total += r_dir
    
    return refusing_total, direct_total

def main():
    parser = argparse.ArgumentParser(
        description="Count 'refusing' and 'direct' responses in PT files in each subfolder of a given main directory."
    )
    parser.add_argument(
        "main_dir",
        type=str,
        help="Path to the main directory that contains subfolders with PT files."
    )
    args = parser.parse_args()

    # Set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("count_responses")

    if not os.path.isdir(args.main_dir):
        logger.error(f"The provided main directory '{args.main_dir}' does not exist or is not a directory.")
        return

    # Iterate over immediate subfolders in the main directory.
    subfolders = [os.path.join(args.main_dir, d) for d in os.listdir(args.main_dir) if os.path.isdir(os.path.join(args.main_dir, d))]

    if not subfolders:
        logger.warning("No subfolders found in the main directory.")
        return

    summary = {}
    for folder in subfolders:
        refusing, direct = count_responses_in_folder(folder, logger)
        folder_name = os.path.basename(folder)
        summary[folder_name] = {"refusing": refusing, "direct": direct}
        logger.info(f"Folder '{folder_name}': refusing = {refusing}, direct = {direct}")

    # Print summary report
    print("=== Summary Report ===")
    for folder, counts in summary.items():
        print(f"Folder: {folder}")
        print(f"  Refusing responses: {counts['refusing']}")
        print(f"  Direct responses: {counts['direct']}")
        print("-" * 30)

if __name__ == "__main__":
    main()
