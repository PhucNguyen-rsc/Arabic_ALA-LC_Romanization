"""
Generate ALA-LC (LOC) predictions using CAMeL Tools disambiguation as morph backoff.

Usage:
  python3 src/predict/camel_morph_backoff.py data/processed/dev.tsv \
          predictions_out/camelmorph/dev/camel_morph.out

Then use as MLE backoff:
  python3 src/loc_transcribe.py predict mle dev -m models/mle/size1.0.tsv \
          -b predictions_out/camelmorph/dev/camel_morph.out
"""

import sys
import os
from pathlib import Path
import pandas as pd
import argparse
import json

try:
    from repackage import up; up()
except Exception:
    pass

from data.make_dataset import tokenize_skiphyph, recompose
from predict.translit_rules import (
    load_loc_mappings,
    load_exceptional_spellings,
    translit_simple,
    capitalize_loc,
)

from camel_tools.disambig.mle import MLEDisambiguator
try:
    from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
    HAS_BERT = True
except Exception:
    HAS_BERT = False
import re


def pretty_print_analysis(token, dw, max_analyses=3):
    """Print a readable version of the disambiguated word analysis"""
    print(f"\n{'='*60}")
    print(f"TOKEN: {token}")
    
    if not dw.analyses:
        print("  NO ANALYSES FOUND")
        return
        
    print(f"  TOP ANALYSIS:")
    top = dw.analyses[0]
    print(f"    Score: {top.score:.4f}")
    print(f"    Diac: {top.analysis.get('diac', 'N/A')}")
    print(f"    Lemma: {top.analysis.get('lemma', 'N/A')}")
    print(f"    POS: {top.analysis.get('pos', 'N/A')}")
    print(f"    Gloss: {top.analysis.get('gloss', 'N/A')}")
    
    features = []
    for key in ['gen', 'num', 'per', 'asp', 'mod', 'vox', 'stt', 'cas']:
        if key in top.analysis and top.analysis[key]:
            features.append(f"{key}:{top.analysis[key]}")
    if features:
        print(f"    Features: {', '.join(features)}")
        
    if 'bw' in top.analysis:
        print(f"    BW: {top.analysis['bw']}")
        
    # Show clitics
    clitics = []
    for i in range(4):
        key = f'prc{i}'
        if key in top.analysis and top.analysis[key]:
            clitics.append(f"prc{i}:{top.analysis[key]}")
    if 'enc0' in top.analysis and top.analysis['enc0']:
        clitics.append(f"enc0:{top.analysis['enc0']}")
    if clitics:
        print(f"    Clitics: {', '.join(clitics)}")
    
    # Show alternatives
    if len(dw.analyses) > 1 and max_analyses > 1:
        print(f"\n  ALTERNATIVE ANALYSES (top {min(max_analyses-1, len(dw.analyses)-1)}):")
        for i, analysis in enumerate(dw.analyses[1:max_analyses], 1):
            print(f"    {i}. Score: {analysis.score:.4f}, " + 
                  f"Diac: {analysis.analysis.get('diac', 'N/A')}, " +
                  f"POS: {analysis.analysis.get('pos', 'N/A')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_tsv', help='TSV with column ar')
    parser.add_argument('output_txt', help='One ALA-LC line per sentence')
    parser.add_argument('--bert', action='store_true', help='Use CAMeL BERT disambiguator if available')
    parser.add_argument('--debug', action='store_true', help='Print detailed analysis for first sentence')
    parser.add_argument('--debug-count', type=int, default=5, help='Number of tokens to debug (with --debug)')
    args = parser.parse_args()

    input_tsv, output_txt = args.input_tsv, args.output_txt
    Path(os.path.dirname(output_txt)).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_tsv, sep='\t')
    if 'ar' not in data.columns:
        raise SystemExit('Input TSV must have an "ar" column')

    loc_map = load_loc_mappings()
    loc_exceptional = load_exceptional_spellings()

    # Load CAMeL disambiguator (downloads models on first use)
    if args.bert and HAS_BERT:
        disamb = BERTUnfactoredDisambiguator.pretrained('msa')
    else:
        disamb = MLEDisambiguator.pretrained()

    predictions: list[str] = []

    for sentence in data['ar'].astype(str):
        # Debug first sentence if requested
        debug_this = args.debug and len(predictions) == 0
        
        # Use the project's tokenization so recompose() behaves identically
        tokens = tokenize_skiphyph(sentence).split()

        # Disambiguate one sentence worth of tokens
        # Returns a list of DisambiguatedWord objects aligned with tokens
        # TODO: run normalize_unicode for all tokens before
        disamb_words = disamb.disambiguate(tokens)

        # Print detailed debug info if requested
        if debug_this:
            print(f"\nDEBUG ANALYSIS FOR SENTENCE: {sentence}")
            for i, (tok, dw) in enumerate(zip(tokens, disamb_words)):
                if i >= args.debug_count:
                    print(f"\n... (showing {args.debug_count} of {len(tokens)} tokens)")
                    break
                pretty_print_analysis(tok, dw)

        rom_tokens: list[str] = []
        for idx, (tok, dw) in enumerate(zip(tokens, disamb_words)): 
            analysis = (dw.analyses[0].analysis if dw.analyses else {})
            diac = analysis.get('diac') or tok

            # TODO: Delete certain characters at the end of the diac token : [\u064B-\u0652]*\b >> nil 



            # Map to ALA-LC
            rom_tok = translit_simple(diac, loc_map, loc_exceptional)
            
            # Fix Al- attachment - ensure al- is separated with hyphen and lowercase
            if analysis.get('prc0') == 'Al_det' or (analysis.get('bw', '') and 'Al/DET' in analysis.get('bw', '')):
                # Make sure al- is properly separated with hyphen and lowercase
                if 'al' in rom_tok.lower():
                    # First ensure there's a hyphen
                    rom_tok = re.sub(r'^([Aa]l)([^-])', r'\1-\2', rom_tok)
                    # Then ensure 'al-' is lowercase, but keep the word after the hyphen with its original case
                    rom_tok = re.sub(r'^([Aa]l)(-[A-Za-zʼʻ])', lambda m: 'al' + m.group(2), rom_tok)
            
            # Capitalization - ONLY for sentence-initial or proper nouns
            if idx == 0:  # First token already capitalized by translit_simple
                pass
            elif idx > 0 and tokens[idx-1] == '.':
                # After period (sentence start)
                rom_tok = capitalize_loc(rom_tok)
            elif analysis.get('pos') == 'noun_prop':
                # Only capitalize proper nouns, not every noun
                rom_tok = capitalize_loc(rom_tok)
            else:
                # Remove capitalization for regular words (fix over-capitalization)
                if rom_tok and rom_tok[0].isupper() and not (rom_tok.startswith("'") or rom_tok.startswith('ʻ')):
                    rom_tok = rom_tok[0].lower() + rom_tok[1:]
                
                # Ensure first letter after al- is capitalized for proper nouns
                if analysis.get('pos') == 'noun_prop' and re.search(r'^al-[a-z]', rom_tok):
                    rom_tok = re.sub(r'^(al-)([a-z])', lambda m: m.group(1) + m.group(2).upper(), rom_tok)

            rom_tokens.append(rom_tok)

        predictions.append(recompose(' '.join(rom_tokens), mode='rom'))

    # Write plain text: one line per sentence, no quotes
    with open(output_txt, 'w', encoding='utf-8') as o:
        for line in predictions:
            o.write(f"{line}\n")
    print(f"Wrote {len(predictions)} lines to {output_txt}")


if __name__ == "__main__":
    main()