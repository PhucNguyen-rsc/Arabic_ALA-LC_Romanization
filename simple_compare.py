#!/usr/bin/env python3
"""
Simple line-by-line comparison between CAMeL output and gold standard.

Usage:
  python3 simple_compare.py predictions_out/camelmorph/dev/camel_morph.out data/processed/dev.tsv [--limit N] [--output FILE]

Options:
  --limit N     Limit comparison to first N lines
  --output FILE Write comparison to a file instead of console
"""

import sys
import argparse
import pandas as pd
import os
import difflib

def compare_files(file1, file2, limit=None, output_file=None):
    """Compare two files line by line"""
    # Read the first file
    with open(file1, 'r', encoding='utf-8') as f:
        lines1 = [line.strip() for line in f]
    
    # Read the second file (TSV)
    df = pd.read_csv(file2, sep='\t')
    if 'rom' not in df.columns or 'ar' not in df.columns:
        print(f"Error: 'rom' or 'ar' column not found in {file2}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    lines2 = df['rom'].astype(str).tolist()
    arabic_lines = df['ar'].astype(str).tolist()
    
    # Ensure same length for comparison
    min_len = min(len(lines1), len(lines2))
    if min_len < len(lines1) or min_len < len(lines2):
        print(f"Warning: Files have different lengths. File1: {len(lines1)}, File2: {len(lines2)}")
        print(f"Comparing only the first {min_len} lines")
    
    lines1 = lines1[:min_len]
    lines2 = lines2[:min_len]
    arabic_lines = arabic_lines[:min_len]
    
    # Apply limit if specified
    if limit:
        lines1 = lines1[:limit]
        lines2 = lines2[:limit]
        arabic_lines = arabic_lines[:limit]
    
    # Prepare output
    if output_file:
        out_file = open(output_file, 'w', encoding='utf-8')
        write = lambda s: out_file.write(s + '\n')
    else:
        write = print
    
    # Compare and print
    matches = 0
    for i, (line1, line2, arabic) in enumerate(zip(lines1, lines2, arabic_lines)):
        match = "✓" if line1 == line2 else "✗"
        if match == "✓":
            matches += 1
        
        write(f"Line {i+1}: {arabic}")
        write(f"Ground truth: {line2}")
        write(f"CAMeL tools: {line1}")
        write(f"Match: {match}")
        
        # Show differences when they don't match
        if match == "✗":
            # Find differences using difflib
            d = difflib.Differ()
            diff = list(d.compare([line2], [line1]))
            
            # Create a more readable diff
            diff_str = ""
            for i, s in enumerate(line2):
                if i < len(line1) and s == line1[i]:
                    diff_str += s
                else:
                    # Find the different part
                    diff_str += f"[{s}≠{line1[i] if i < len(line1) else ''}]"
            
            write(f"Diff: {diff_str}")
        
        write("")  # Empty line for spacing
    
    # Print summary
    total = len(lines1)
    match_percent = (matches / total * 100) if total > 0 else 0
    write("-" * 40)
    write(f"Total lines: {total}")
    write(f"Matches: {matches} ({match_percent:.2f}%)")
    write(f"Differences: {total - matches} ({100-match_percent:.2f}%)")
    
    # Close file if opened
    if output_file:
        out_file.close()
        print(f"Comparison written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Simple line-by-line comparison")
    parser.add_argument("file1", help="Path to CAMeL output file")
    parser.add_argument("file2", help="Path to ground truth standard TSV file")
    parser.add_argument("--limit", type=int, help="Limit comparison to first N lines")
    parser.add_argument("--output", help="Write comparison to a file instead of console")
    args = parser.parse_args()
    
    compare_files(args.file1, args.file2, args.limit, args.output)

if __name__ == "__main__":
    main()