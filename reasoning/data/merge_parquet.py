"""
Merge all train and test parquet files from different data sources.
"""

import argparse
import os
from glob import glob

from datasets import Dataset, concatenate_datasets, load_dataset


def merge_parquet_files(input_dir, output_dir):
    """Merge all parquet files in train/ and test/ directories"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge train files
    train_files = glob(os.path.join(input_dir, "train/*.parquet"))
    if train_files:
        print(f"Found {len(train_files)} train parquet files:")
        for f in train_files:
            print(f"  - {f}")
        
        train_datasets = []
        for f in train_files:
            ds = load_dataset("parquet", data_files=f, split="train")
            print(f"  Loaded {len(ds)} samples from {os.path.basename(f)}")
            train_datasets.append(ds)
        
        merged_train = concatenate_datasets(train_datasets)
        train_output = os.path.join(output_dir, "train.parquet")
        merged_train.to_parquet(train_output)
        print(f"Merged train: {len(merged_train)} samples -> {train_output}")
    else:
        print("No train parquet files found")
    
    # Merge test files
    test_files = glob(os.path.join(input_dir, "test/*.parquet"))
    if test_files:
        print(f"\nFound {len(test_files)} test parquet files:")
        for f in test_files:
            print(f"  - {f}")
        
        test_datasets = []
        for f in test_files:
            ds = load_dataset("parquet", data_files=f, split="train")
            print(f"  Loaded {len(ds)} samples from {os.path.basename(f)}")
            test_datasets.append(ds)
        
        merged_test = concatenate_datasets(test_datasets)
        test_output = os.path.join(output_dir, "test.parquet")
        merged_test.to_parquet(test_output)
        print(f"Merged test: {len(merged_test)} samples -> {test_output}")
    else:
        print("No test parquet files found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge train/test parquet files from multiple data sources")
    parser.add_argument("--input_dir", required=True, help="Input directory containing train/ and test/ subdirs with parquet files")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: same as input_dir)")
    
    args = parser.parse_args()
    
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = args.output_dir or input_dir
    output_dir = os.path.expanduser(output_dir)
    
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}\n")
    
    merge_parquet_files(input_dir, output_dir)
