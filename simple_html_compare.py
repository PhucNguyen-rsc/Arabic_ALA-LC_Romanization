#!/usr/bin/env python3
"""
Simple HTML comparison with very clear color highlighting.

Usage:
  python3 simple_html_compare.py camel_output.txt gold_standard.tsv --output comparison.html
"""

import sys
import argparse
import pandas as pd
import os

def compare_files(file1, file2, limit=None, output_file=None):
    """Compare two files and generate HTML with clear color highlighting"""
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
    
    # Create HTML output
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write HTML header
        f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CAMeL vs Gold Comparison</title>
    <style>
        body {
            font-family: 'Courier New', monospace;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.8;
            background-color: #f5f5f5;
        }
        .entry {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 30px;
        }
        .arabic {
            font-family: 'Arial', sans-serif;
            font-size: 18px;
            direction: rtl;
            margin-bottom: 15px;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 4px;
        }
        .gold {
            background-color: #e6f3ff;
            padding: 8px;
            border-left: 4px solid #0066cc;
            margin: 10px 0;
        }
        .camel {
            background-color: #fff2e6;
            padding: 8px;
            border-left: 4px solid #cc6600;
            margin: 10px 0;
        }
        .match-yes {
            color: green;
            font-weight: bold;
        }
        .match-no {
            color: red;
            font-weight: bold;
        }
        .diff {
            background-color: #fffbf0;
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #ffcc00;
            font-size: 16px;
            line-height: 1.6;
        }
        .highlight-add {
            background-color: #ffcccc;
            color: #cc0000;
            font-weight: bold;
            padding: 2px;
        }
        .highlight-remove {
            background-color: #ccffcc;
            color: #006600;
            font-weight: bold;
            padding: 2px;
        }
        .highlight-change {
            background-color: #ffffcc;
            color: #cc6600;
            font-weight: bold;
            padding: 2px;
        }
        .summary {
            background-color: #eee;
            padding: 15px;
            margin-top: 30px;
            border-radius: 5px;
        }
        h1 {
            text-align: center;
            color: #333;
        }
    </style>
</head>
<body>
    <h1>CAMeL vs Gold Standard Comparison</h1>
""")
        
        # Process each line
        matches = 0
        for i, (camel, gold, arabic) in enumerate(zip(camel_lines, gold_lines, arabic_lines)):
            match = "✓" if camel == gold else "✗"
            if match == "✓":
                matches += 1
            
            # Start entry div
            f.write(f'<div class="entry">\n')
            f.write(f'<div class="arabic">Line {i+1}: {arabic}</div>\n')
            f.write(f'<div class="gold">Gold: {gold}</div>\n')
            f.write(f'<div style="height:8px;"></div>\n')  # Small spacer
            f.write(f'<div class="camel">CAMeL: {camel}</div>\n')
            f.write(f'<div style="height:10px;"></div>\n')  # Small spacer
            
            match_class = "match-yes" if match == "✓" else "match-no"
            f.write(f'<div>Match: <span class="{match_class}">{match}</span></div>\n')
            
            # Show differences when they don't match
            if match == "✗":
                # Add space before diff
                f.write(f'<div style="height:12px;"></div>\n')  # Spacer before diff
                # Create a character-by-character diff
                diff_html = []
                
                # Split into words for better comparison
                gold_words = gold.split()
                camel_words = camel.split()
                
                # Compare words
                if len(gold_words) == len(camel_words):
                    # Same number of words, compare each word
                    for g_word, c_word in zip(gold_words, camel_words):
                        if g_word == c_word:
                            diff_html.append(g_word)
                        else:
                            # Highlight differences
                            diff_html.append(f'<span class="highlight-change">{g_word}≠{c_word}</span>')
                else:
                    # Different number of words, show side by side
                    gold_html = ' '.join([f'<span class="highlight-remove">{w}</span>' for w in gold_words])
                    camel_html = ' '.join([f'<span class="highlight-add">{w}</span>' for w in camel_words])
                    diff_html.append(f"{gold_html} ⟹ {camel_html}")
                
                f.write(f'<div class="diff">Diff: {" ".join(diff_html)}</div>\n')
            
            # End entry div
            f.write('</div>\n')
        
        # Write summary
        total = len(camel_lines)
        match_percent = (matches / total * 100) if total > 0 else 0
        
        f.write('<div class="summary">\n')
        f.write('<h2>Summary</h2>\n')
        f.write(f'<p>Total lines: {total}</p>\n')
        f.write(f'<p>Matches: {matches} ({match_percent:.2f}%)</p>\n')
        f.write(f'<p>Differences: {total - matches} ({100-match_percent:.2f}%)</p>\n')
        f.write('</div>\n')
        
        # Close HTML
        f.write('</body>\n</html>')
    
    print(f"HTML comparison written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Simple HTML comparison with clear color highlighting")
    parser.add_argument("file1", help="Path to CAMeL output file")
    parser.add_argument("file2", help="Path to ground truth standard TSV file")
    parser.add_argument("--limit", type=int, help="Limit comparison to first N lines")
    parser.add_argument("--output", required=True, help="Output HTML file path")
    args = parser.parse_args()
    
    compare_files(args.file1, args.file2, args.limit, args.output)

if __name__ == "__main__":
    main()
