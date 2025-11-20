#!/usr/bin/env python3
"""
Line-by-line comparison between CAMeL output, morph rules output, and gold standard with colorful diff highlighting.

Usage:
  python3 color_compare.py predictions_out/camelmorph/dev/camel_morph.out data/processed/dev.tsv predictions_out/morph/dev/morph.out [--limit N] [--output FILE]

  # Or with named arguments:
  python3 color_compare.py predictions_out/camelmorph/dev/camel_morph.out data/processed/dev.tsv --morph-file predictions_out/morph/dev/morph.out

Options:
  --limit N      Limit comparison to first N lines
  --output FILE  Write comparison to a file instead of console
  --html         Output HTML file with colored differences
  --no-color     Disable colored output
  --morph-file FILE  Path to morph rules output file for additional comparison
"""

import sys
import argparse
import pandas as pd
import os
import re
import difflib

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def generate_html_table(gold_str, camel_str, morph_str):
    """Generates an HTML table for a single entry to align words vertically."""
    gold_words = gold_str.split()
    camel_words = camel_str.split()
    morph_words = morph_str.split() if morph_str else []
    
    max_len = max(len(gold_words), len(camel_words), len(morph_words))

    # Pad shorter lists with empty strings to make them equal length
    gold_words.extend([''] * (max_len - len(gold_words)))
    camel_words.extend([''] * (max_len - len(camel_words)))
    morph_words.extend([''] * (max_len - len(morph_words)))

    table_html = "<table class='comparison-table'>\n"
    
    # Ground Truth Row
    table_html += "  <tr>\n    <th class='row-label'>Ground Truth</th>\n"
    for word in gold_words:
        table_html += f"    <td>{word}</td>\n"
    table_html += "  </tr>\n"
    
    # CAMeL Row
    table_html += "  <tr>\n    <th class='row-label'>CAMeL</th>\n"
    for i, word in enumerate(camel_words):
        style = "style='background-color: #ffff99;'" if word != gold_words[i] else ""
        table_html += f"    <td {style}>{word}</td>\n"
    table_html += "  </tr>\n"
    
    # Morph Rules Row
    if morph_str is not None:
        table_html += "  <tr>\n    <th class='row-label'>Morph Rules</th>\n"
        for i, word in enumerate(morph_words):
            style = "style='background-color: #ffff99;'" if word != gold_words[i] else ""
            table_html += f"    <td {style}>{word}</td>\n"
        table_html += "  </tr>\n"
        
    table_html += "</table>"
    return table_html


def highlight_word_diff_term(gold_str, candidate_str):
    """
    Compares two strings word by word and highlights differences in the candidate string
    with a yellow background for terminal output.
    """
    gold_words = gold_str.split()
    candidate_words = candidate_str.split()
    highlighted_words = []
    
    max_len = max(len(gold_words), len(candidate_words))
    for i in range(max_len):
        if i < len(gold_words) and i < len(candidate_words):
            if gold_words[i] == candidate_words[i]:
                highlighted_words.append(candidate_words[i])
            else:
                # Word is different, highlight it
                highlighted_words.append(f"{YELLOW}{candidate_words[i]}{RESET}")
        elif i < len(candidate_words):
            # Extra word in candidate, highlight it
            highlighted_words.append(f"{YELLOW}{candidate_words[i]}{RESET}")

    return ' '.join(highlighted_words)


def colorize_diff(gold, camel):
    """Create a colorful diff between gold and camel strings"""
    # This function is now a wrapper for the new terminal highlighting logic
    return highlight_word_diff_term(gold, camel)

def compare_files(file1, file2, morph_file=None, limit=None, output_file=None, use_color=True, html_output=False):
    """Compare two files line by line"""
    # Read the first file (CAMeL output)
    with open(file1, 'r', encoding='utf-8') as f:
        camel_lines = [line.strip() for line in f]
    
    # Read the second file (TSV with gold standard)
    df = pd.read_csv(file2, sep='\t')
    if 'rom' not in df.columns or 'ar' not in df.columns:
        print(f"Error: 'rom' or 'ar' column not found in {file2}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    gold_lines = df['rom'].astype(str).tolist()
    arabic_lines = df['ar'].astype(str).tolist()
    
    # Read the morph rules output file if provided
    morph_lines = None
    if morph_file and os.path.exists(morph_file):
        with open(morph_file, 'r', encoding='utf-8') as f:
            morph_lines = [line.strip() for line in f]
    
    # Ensure same length for comparison
    file_lengths = [len(camel_lines), len(gold_lines)]
    if morph_lines:
        file_lengths.append(len(morph_lines))
        
    min_len = min(file_lengths)
    if len(set(file_lengths)) > 1:
        print(f"Warning: Files have different lengths. CAMeL: {len(camel_lines)}, Gold: {len(gold_lines)}")
        if morph_lines:
            print(f", Morph Rules: {len(morph_lines)}")
        print(f"Comparing only the first {min_len} lines")
    
    camel_lines = camel_lines[:min_len]
    gold_lines = gold_lines[:min_len]
    arabic_lines = arabic_lines[:min_len]
    if morph_lines:
        morph_lines = morph_lines[:min_len]
    
    # Apply limit if specified
    if limit:
        camel_lines = camel_lines[:limit]
        gold_lines = gold_lines[:limit]
        arabic_lines = arabic_lines[:limit]
        if morph_lines:
            morph_lines = morph_lines[:limit]
    
    # Prepare output
    if output_file:
        out_file = open(output_file, 'w', encoding='utf-8')
        write = lambda s: out_file.write(s + '\n')
    else:
        write = print
    
    # Compare and print
    camel_matches = 0
    morph_matches = 0
    
    # For HTML output, we use a different format
    if html_output:
        for i, (camel, gold, arabic) in enumerate(zip(camel_lines, gold_lines, arabic_lines)):
            morph = morph_lines[i] if morph_lines else None
            
            camel_match = "✓" if camel == gold else "✗"
            camel_match_class = "match-yes" if camel_match == "✓" else "match-no"
            if camel_match == "✓":
                camel_matches += 1
            
            morph_match = "✓" if morph and morph == gold else "✗" if morph else ""
            morph_match_class = "match-yes" if morph_match == "✓" else "match-no" if morph else ""
            if morph_match == "✓":
                morph_matches += 1
            
            # Write entry with HTML formatting
            write(f"<div class='entry'>")
            write(f"<div class='arabic'>Line {i+1}: {arabic}</div>")

            # Generate and write the comparison table
            table = generate_html_table(gold, camel, morph)
            write(table)
            
            write("</div>")
            write("<div style='height:10px;'></div>")
    else:
        # Standard text output
        for i, (camel, gold, arabic) in enumerate(zip(camel_lines, gold_lines, arabic_lines)):
            morph = morph_lines[i] if morph_lines else None
            
            # Count matches for summary statistics only
            if camel == gold:
                camel_matches += 1
            if morph and morph == gold:
                morph_matches += 1
            
            # Write basic info
            write(f"- Line {i+1}: {arabic}")
            write(f"    * Ground Truth: {gold}")
            
            # Show CAMeL with differences highlighted
            if use_color and camel != gold:
                diff = highlight_word_diff_term(gold, camel)
                write(f"    * CAMeL: {diff}")
            else:
                write(f"    * CAMeL: {camel}")
                
            # Add morph rules output if available
            if morph:
                if use_color and morph != gold:
                    morph_diff = highlight_word_diff_term(gold, morph)
                    write(f"    * Morph Rules: {morph_diff}")
                else:
                    write(f"    * Morph Rules: {morph}")
            
            write("<div style='height:10px;'></div>")  # One more empty line
    
    # Print summary
    total = len(camel_lines)
    camel_match_percent = (camel_matches / total * 100) if total > 0 else 0
    morph_match_percent = (morph_matches / total * 100) if total > 0 and morph_lines else 0
    
    if html_output:
        write("<div class='summary'>")
        write(f"<h2>Summary</h2>")
        write(f"<p>Total lines: {total}</p>")
        write(f"<p>CAMeL Matches: {camel_matches} ({camel_match_percent:.2f}%)</p>")
        write(f"<p>CAMeL Differences: {total - camel_matches} ({100-camel_match_percent:.2f}%)</p>")
        if morph_lines:
            write(f"<p>Morph Rules Matches: {morph_matches} ({morph_match_percent:.2f}%)</p>")
            write(f"<p>Morph Rules Differences: {total - morph_matches} ({100-morph_match_percent:.2f}%)</p>")
        write("</div>")
    else:
        write("-" * 40)
        write(f"Total lines: {total}")
        write(f"CAMeL Matches: {camel_matches} ({camel_match_percent:.2f}%)")
        write(f"CAMeL Differences: {total - camel_matches} ({100-camel_match_percent:.2f}%)")
        if morph_lines:
            write(f"Morph Rules Matches: {morph_matches} ({morph_match_percent:.2f}%)")
            write(f"Morph Rules Differences: {total - morph_matches} ({100-morph_match_percent:.2f}%)")
        # Removing the now-redundant explanation line.
    
    # Close file if opened
    if output_file:
        out_file.close()
        print(f"Comparison written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Colorful line-by-line comparison")
    parser.add_argument("file1", help="Path to CAMeL output file")
    parser.add_argument("file2", help="Path to ground truth standard TSV file")
    parser.add_argument("file3", nargs="?", default=None, help="Path to morph rules output file (optional positional argument)")
    parser.add_argument("--morph-file", help="Path to morph rules output file (alternative to file3)")
    parser.add_argument("--limit", type=int, help="Limit comparison to first N lines")
    parser.add_argument("--output", help="Write comparison to a file instead of console")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--html", action="store_true", help="Output HTML file with colored differences")
    args = parser.parse_args()
    
    # If HTML output is requested, force output to a file
    if args.html and not args.output:
        print("Error: --html requires --output to be specified")
        sys.exit(1)
        
    # If HTML output is requested, add .html extension if not present
    if args.html and args.output and not args.output.lower().endswith('.html'):
        args.output = args.output + '.html'
    
    # If HTML output is requested, write the HTML header
    if args.html and args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write("<meta charset=\"UTF-8\">\n")
            f.write("<title>CAMeL vs Gold Comparison</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: 'SF Mono', 'Courier New', monospace; line-height: 1.5; max-width: 95%; margin: 0 auto; padding: 20px; }\n")
            f.write(".entry { background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; overflow-x: auto; }\n")
            f.write(".arabic { font-family: 'Arial', sans-serif; font-size: 18px; direction: rtl; margin-bottom: 10px; }\n")
            f.write(".comparison-table { border-collapse: collapse; width: 100%; margin-top: 10px; }\n")
            f.write(".comparison-table th, .comparison-table td { padding: 8px 12px; text-align: left; border: 1px solid #e0e0e0; min-width: 100px; }\n")
            f.write(".comparison-table th.row-label { background-color: #f0f7ff; font-weight: bold; white-space: nowrap; }\n")
            f.write(".summary { background-color: #eee; padding: 15px; margin-top: 30px; border-radius: 5px; }\n")
            f.write("</style>\n</head>\n<body>\n")
            f.write("<h1>CAMeL vs Gold Standard Comparison</h1>\n")
    
    # Use either positional argument file3 or named argument morph-file
    morph_file = args.file3 if args.file3 else args.morph_file
    compare_files(args.file1, args.file2, morph_file, args.limit, args.output, not args.no_color, args.html)
    
    # If HTML output is requested, close the HTML tags
    if args.html and args.output:
        with open(args.output, 'a', encoding='utf-8') as f:
            f.write("</body>\n</html>")
            print(f"HTML comparison written to {args.output}")

if __name__ == "__main__":
    main()
