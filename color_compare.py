#!/usr/bin/env python3
"""
Line-by-line comparison between CAMeL output and gold standard with colorful diff highlighting.

Usage:
  python3 color_compare.py predictions_out/camelmorph/dev/camel_morph.out data/processed/dev.tsv [--limit N] [--output FILE]

Options:
  --limit N     Limit comparison to first N lines
  --output FILE Write comparison to a file instead of console
  --html        Output HTML file with colored differences
  --no-color    Disable colored output
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

def colorize_diff(gold, camel):
    """Create a colorful diff between gold and camel strings"""
    # Split both strings into words
    gold_words = gold.split()
    camel_words = camel.split()
    
    # Compare word by word
    result = []
    i, j = 0, 0
    
    # Process words until we run out in either string
    while i < len(gold_words) and j < len(camel_words):
        if gold_words[i] == camel_words[j]:
            # Words match, keep as is
            result.append(gold_words[i])
            i += 1
            j += 1
        else:
            # Words differ - check if it's a hyphenation difference
            g_word = gold_words[i]
            c_word = camel_words[j]
            
            # Check if it's just a hyphenation difference
            if g_word.replace('-', '') == c_word.replace('-', ''):
                # Highlight hyphenation differences
                if '-' in g_word and '-' not in c_word:
                    # Gold has hyphen that camel doesn't
                    parts = g_word.split('-')
                    result.append(f"{parts[0]}{BLUE}-{RESET}{parts[1]}")
                elif '-' in c_word and '-' not in g_word:
                    # Camel has hyphen that gold doesn't
                    result.append(f"{RED}{g_word}{RESET}")
                else:
                    # Different hyphen position
                    result.append(f"{YELLOW}{g_word}{RESET}")
            else:
                # More complex difference - character by character
                char_diff = []
                for k in range(max(len(g_word), len(c_word))):
                    if k < len(g_word) and k < len(c_word) and g_word[k] == c_word[k]:
                        char_diff.append(g_word[k])
                    elif k < len(g_word):
                        if g_word[k] == '-':
                            char_diff.append(f"{BLUE}-{RESET}")
                        else:
                            char_diff.append(f"{GREEN}{g_word[k]}{RESET}")
                    elif k < len(c_word):
                        char_diff.append(f"{RED}{c_word[k]}{RESET}")
                
                result.append(''.join(char_diff))
            
            i += 1
            j += 1
    
    # Handle remaining words
    while i < len(gold_words):
        result.append(f"{GREEN}{gold_words[i]}{RESET}")
        i += 1
        
    while j < len(camel_words):
        result.append(f"{RED}{camel_words[j]}{RESET}")
        j += 1
    
    return ' '.join(result)

def html_colorize_diff(gold, camel):
    """Create an HTML-colored diff between gold and camel strings"""
    # Split both strings into words
    gold_words = gold.split()
    camel_words = camel.split()
    
    # Compare word by word
    result = []
    i, j = 0, 0
    
    # Process words until we run out in either string
    while i < len(gold_words) and j < len(camel_words):
        if gold_words[i] == camel_words[j]:
            # Words match, keep as is
            result.append(gold_words[i])
            i += 1
            j += 1
        else:
            # Words differ - check if it's a hyphenation difference
            g_word = gold_words[i]
            c_word = camel_words[j]
            
            # Check if it's just a hyphenation difference
            if g_word.replace('-', '') == c_word.replace('-', ''):
                # Highlight hyphenation differences
                if '-' in g_word and '-' not in c_word:
                    # Gold has hyphen that camel doesn't
                    parts = g_word.split('-')
                    result.append(f"<span style='color:#0000cc;font-weight:bold'>{parts[0]}-{parts[1]}</span>")
                elif '-' in c_word and '-' not in g_word:
                    # Camel has hyphen that gold doesn't
                    result.append(f"<span style='color:#cc0000;font-weight:bold'>{g_word}</span>")
                else:
                    # Different hyphen position
                    result.append(f"<span style='color:#cc9900;font-weight:bold'>{g_word}</span>")
            else:
                # More complex difference - character by character
                char_diff = []
                for k in range(max(len(g_word), len(c_word))):
                    if k < len(g_word) and k < len(c_word) and g_word[k] == c_word[k]:
                        char_diff.append(g_word[k])
                    elif k < len(g_word):
                        if g_word[k] == '-':
                            char_diff.append(f"<span style='color:#0000cc;font-weight:bold'>-</span>")
                        else:
                            char_diff.append(f"<span style='color:#006600;font-weight:bold'>{g_word[k]}</span>")
                    elif k < len(c_word):
                        char_diff.append(f"<span style='color:#cc0000;font-weight:bold'>{c_word[k]}</span>")
                
                result.append(''.join(char_diff))
            
            i += 1
            j += 1
    
    # Handle remaining words
    while i < len(gold_words):
        result.append(f"<span style='color:#006600;font-weight:bold'>{gold_words[i]}</span>")
        i += 1
        
    while j < len(camel_words):
        result.append(f"<span style='color:#cc0000;font-weight:bold'>{camel_words[j]}</span>")
        j += 1
    
    return ' '.join(result)

def compare_files(file1, file2, limit=None, output_file=None, use_color=True, html_output=False):
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
    
    # Ensure same length for comparison
    min_len = min(len(camel_lines), len(gold_lines))
    if min_len < len(camel_lines) or min_len < len(gold_lines):
        print(f"Warning: Files have different lengths. CAMeL: {len(camel_lines)}, Gold: {len(gold_lines)}")
        print(f"Comparing only the first {min_len} lines")
    
    camel_lines = camel_lines[:min_len]
    gold_lines = gold_lines[:min_len]
    arabic_lines = arabic_lines[:min_len]
    
    # Apply limit if specified
    if limit:
        camel_lines = camel_lines[:limit]
        gold_lines = gold_lines[:limit]
        arabic_lines = arabic_lines[:limit]
    
    # Prepare output
    if output_file:
        out_file = open(output_file, 'w', encoding='utf-8')
        write = lambda s: out_file.write(s + '\n')
    else:
        write = print
    
    # Compare and print
    matches = 0
    
    # For HTML output, we use a different format
    if html_output:
        for i, (camel, gold, arabic) in enumerate(zip(camel_lines, gold_lines, arabic_lines)):
            match = "✓" if camel == gold else "✗"
            match_class = "match-yes" if match == "✓" else "match-no"
            if match == "✓":
                matches += 1
            
            # Write entry with HTML formatting
            write(f"<div class='entry'>")
            write(f"<div class='arabic'>Line {i+1}: {arabic}</div>")
            write(f"<div class='gold'>Gold: {gold}</div>")
            write(f"<div class='camel'>CAMeL: {camel}</div>")
            write(f"<div class='match'>Match: <span class='{match_class}'>{match}</span></div>")
            
            # Show differences when they don't match
            if match == "✗" and use_color:
                diff = html_colorize_diff(gold, camel)
                write(f"<div class='diff'>Diff: {diff}</div>")
            
            write("</div>")
            write("<div style='height:10px;'></div>")
    else:
        # Standard text output
        for i, (camel, gold, arabic) in enumerate(zip(camel_lines, gold_lines, arabic_lines)):
            match = "✓" if camel == gold else "✗"
            if match == "✓":
                matches += 1
            
            # Write basic info
            write(f"- Line {i+1}: {arabic}")
            write(f"    * Ground truth: {gold}")
            write(f"    * CAMeL: {camel}")
            write(f"    * Match: {match}")
            
            # Show differences when they don't match
            if match == "✗" and use_color:
                diff = colorize_diff(gold, camel)
                write(f"    * Diff: {diff}")
            
            write("<div style='height:10px;'></div>")  # One more empty line
    
    # Print summary
    total = len(camel_lines)
    match_percent = (matches / total * 100) if total > 0 else 0
    
    if html_output:
        write("<div class='summary'>")
        write(f"<h2>Summary</h2>")
        write(f"<p>Total lines: {total}</p>")
        write(f"<p>Matches: {matches} ({match_percent:.2f}%)</p>")
        write(f"<p>Differences: {total - matches} ({100-match_percent:.2f}%)</p>")
        write("</div>")
    else:
        write("-" * 40)
        write(f"Total lines: {total}")
        write(f"Matches: {matches} ({match_percent:.2f}%)")
        write(f"Differences: {total - matches} ({100-match_percent:.2f}%)")
    
    # Close file if opened
    if output_file:
        out_file.close()
        print(f"Comparison written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Colorful line-by-line comparison")
    parser.add_argument("file1", help="Path to CAMeL output file")
    parser.add_argument("file2", help="Path to ground truth standard TSV file")
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
        
        # If HTML output is requested, wrap the output in HTML tags
        if args.html and args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write("<!DOCTYPE html>\n<html>\n<head>\n")
                f.write("<meta charset=\"UTF-8\">\n")
                f.write("<title>CAMeL vs Gold Comparison</title>\n")
                f.write("<style>\n")
                f.write("body { font-family: 'Courier New', monospace; line-height: 1.5; max-width: 1200px; margin: 0 auto; padding: 20px; }\n")
                f.write(".entry { background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }\n")
                f.write(".arabic { font-family: 'Arial', sans-serif; font-size: 18px; direction: rtl; margin-bottom: 10px; }\n")
                f.write(".gold { background-color: #f0f7ff; padding: 5px; border-left: 4px solid #0066cc; margin: 5px 0; }\n")
                f.write(".camel { background-color: #fff6f0; padding: 5px; border-left: 4px solid #cc6600; margin: 5px 0; }\n")
                f.write(".match { font-weight: bold; }\n")
                f.write(".match-yes { color: green; }\n")
                f.write(".match-no { color: red; }\n")
                f.write(".diff { background-color: #fffaf0; padding: 10px; margin: 10px 0; border-left: 4px solid #ffcc00; font-size: 16px; }\n")
                f.write(".diff span { font-weight: bold; }\n")
                f.write(".summary { background-color: #eee; padding: 15px; margin-top: 30px; border-radius: 5px; }\n")
                f.write(".red, span.red { color: #cc0000; font-weight: bold; }\n")
                f.write(".green, span.green { color: #006600; font-weight: bold; }\n")
                f.write(".blue, span.blue { color: #0000cc; font-weight: bold; }\n")
                f.write(".gold, span.gold { color: #cc9900; font-weight: bold; }\n")
                f.write("</style>\n</head>\n<body>\n")
                f.write("<h1>CAMeL vs Gold Standard Comparison</h1>\n")
    
    compare_files(args.file1, args.file2, args.limit, args.output, not args.no_color, args.html)
    
    # If HTML output is requested, close the HTML tags
    if args.html and args.output:
        with open(args.output, 'a', encoding='utf-8') as f:
            f.write("</body>\n</html>")
            print(f"HTML comparison written to {args.output}")

if __name__ == "__main__":
    main()
