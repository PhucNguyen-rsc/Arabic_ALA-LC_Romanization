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
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar
from camel_tools.ner import NERecognizer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
import re
from tqdm import tqdm
import translit_rules

'''

#from camel_tools.morphology.database import MorphologyDB
#from camel_tools.morphology.analyzer import Analyzer

# Load a morphological database. Here we use calima-msa-s31.db.
morphdb = MorphologyDB('calima-msa-s31.db')

# Instantiate an analyzer that uses the morphological database, and specifies the backoff mode to be NOAN_PROP.
#     There are five backoff modes:
#         1. NONE No back off analyses are generated (Default).
#         2. NOAN_ALL Generate all backoff analyses only if no analyses are generated.
#         3. NOAN_PROP Generate proper noun backoff analyses if no analyses are generated.
#         4. ADD_ALL Generate all backoff analyses in addition to generated analyses.
#         5. ADD_PROP Generate proper noun backoff analyses in addition to generated analyses.
#     For more info, see: https://camel-tools.readthedocs.io/en/latest/api/morphology/analyzer.html

analyzer = Analyzer(morphdb,'NOAN_PROP')

# Load the default MLE disambiguator for MSA provided by CAMeL Tools.
MLE_DISAMBIG = MLEDisambiguator.pretrained('calima-msa-r13', cache_size=2000000)

# Replace the default analyzer (calima-msa-r13) with the calima-msa-s31 analyzer.
# This is the same analyzer we defined earlier:

MLE_DISAMBIG._analyzer = analyzer
'''

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
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_tsv', help='TSV with column ar')
    parser.add_argument('output_txt', help='One ALA-LC line per sentence')
    parser.add_argument('output_arabic_txt', help='One Arabic line per sentence')
    parser.add_argument('--bert', action='store_true', help='Use CAMeL BERT disambiguator if available')
    parser.add_argument('--custom-analyzer', action='store_true', help='Use custom calima-msa-s31.db analyzer instead of default')
    parser.add_argument('--debug', action='store_true', help='Print detailed analysis for first sentence')
    parser.add_argument('--debug-count', type=int, default=5, help='Number of tokens to debug (with --debug)')
    args = parser.parse_args()

    input_tsv, output_txt = args.input_tsv, args.output_txt
    output_arabic_txt = args.output_arabic_txt  
    Path(os.path.dirname(output_txt)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(output_arabic_txt)).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_tsv, sep='\t')
    if 'ar' not in data.columns:
        raise SystemExit('Input TSV must have an "ar" column')

    loc_map = load_loc_mappings()
    loc_exceptional = load_exceptional_spellings()

    # Set up analyzer based on flag
    if args.custom_analyzer:
        print("Using custom calima-msa-s31.db analyzer")
        morphdb = MorphologyDB('analyser/calima-msa-s31.db')
        analyzer = Analyzer(morphdb,'NOAN_PROP')
    else:
        print("Using default CAMeL analyzer")
        analyzer = None  # Will use the default analyzer that comes with the disambiguator

    # Load CAMeL disambiguator (downloads models on first use)
    if args.bert:
        disamb = BERTUnfactoredDisambiguator.pretrained('msa')
    else:
        disamb = MLEDisambiguator.pretrained('calima-msa-r13', cache_size=2000000)

    # Only override the analyzer if we're using custom analyzer
    if args.custom_analyzer:
        disamb._analyzer = analyzer
    
    # No need for custom GPU handling - BERTUnfactoredDisambiguator uses GPU by default
    import torch
    if torch.cuda.is_available() and args.bert:
        print(f"GPU available: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
        print("Using GPU for BERT (handled by CAMeL Tools internally)")
    else:
        print("Using CPU for disambiguation (not using BERT or GPU not available)")

    predictions: list[str] = []
    arabic_predictions: list[str] = []
    sentences = tqdm(data['ar'].astype(str), desc="Romanizing", unit="sentence")

    prc0_set = set()
    prc1_set = set()

    for sentence in sentences:
        tokens = sentence.split() # raw arabic tokens

        disamb_words = disamb.disambiguate(tokens) # list of DisambiguationWord objects

        rom_tokens: list[str] = []
        diacritized_words: list[str] = []

        for idx, (tok, dw) in enumerate(zip(tokens, disamb_words)): 
            analysis = (dw.analyses[0].analysis)

            if analysis.get('prc0') and analysis.get('prc0') != '0' and analysis.get('prc0') != 'na':
                prc0_set.add(analysis.get('prc0'))
            if analysis.get('prc1') and analysis.get('prc1') != '0' and analysis.get('prc1') != 'na':
                prc1_set.add(analysis.get('prc1'))

            diac = analysis.get('diac')
            diacritized_words.append(diac)

            if tok in loc_exceptional:
                diac = loc_exceptional[tok]
                if idx == 0 and not diac.endswith(capschar):
                    diac = diac+capschar
                
                rom_tokens.append(diac)
                continue

            if idx < len(disamb_words)-1:
                nextanalysis = disamb_words[idx+1].analyses[0].analysis
            else:
                nextanalysis = None

            bw = analysis.get('bw') 
            bwsplit = bw.split('+') 
            bwending = bwsplit[-1]   
            bwbeginning = bwsplit[0]

            if analysis['prc0'] == 'Al_det' and analysis['prc1'] == 'li_prep':
                find = re.escape('لِ+ال')
                diac = re.sub(find,r'لِل',diac)

            elif analysis.get('prc0') == 'Al_det':
                if diac.startswith('ال'):
                     diac = diac.replace('ال', 'ال+', 1)
            
            # Handle 'bi' (bi_prep, bi_part)
            elif analysis.get('prc1') in ['bi_prep', 'bi_part']:
                if diac.startswith('بِ') :
                    diac = diac.replace('بِ', 'بِ+', 1)

            # Add '+' to li_prep (Preposition 'li') - when NOT part of 'lil'
            elif analysis.get('prc1') in ['li_prep', 'li_jus', 'li_sub']:
                if diac.startswith('لِ'):
                    diac = diac.replace('لِ', 'لِ+', 1)

            elif analysis.get('prc2') in ['wa_conj', 'wa_part', 'wa_sub']:
                if diac.startswith('وَ'):
                    diac = diac.replace('وَ', 'وَ+', 1)

            if ('DO' in bwending) or ('POSS_PRON' in bwending): 
                pass
            # elif word ends with case ending, nominall suffix, or a verb (imperfective, perfective, and command)
            elif ('CASE' in bwending) or ('IV' in bwending) or ('PV' in bwending) or ('CV' in bwending) or ('NSUFF' in bwending):
                # remove final diacritic, including alif for tanween
                diac = re.sub(r'اً(±)?$',r'\1',diac) #alif tanween must be first
                diac = re.sub(r'[ًٌٍَُِ](±)?$',r'\1',diac)
            
            if bw and ('DO' in bwending or 'POSS_PRON' in bwending): 
                pass

            elif bw and any(marker in bwending for marker in ['CASE', 'IV', 'PV', 'CV', 'NSUFF']):
                diac = re.sub(r'اً$', '', diac)
                diac = re.sub(r'[ًٌٍَُِ]$', '', diac)

            
            if 'ة' in diac:
                if analysis.get('stt') == 'c':
                    # caveat: cannot be construct if followed by prep (additional rule for handling odd madamira analysis)
                    if nextanalysis and 'PREP' in nextanalysis['bw'].split('+')[0]:
                        pass
                    else:
                        # put a sukun on the ta-marbuta for transliterator to spell it
                        diac = re.sub(r'ة',r'ةْ',diac)

            
            if 'PREP' in bwbeginning and len(bwsplit)>1:  # length condition to make sure letters are actual proclitics
                an = analysis.get('lex').split('_')[0] #lex in camel tools analysis, lemma in madamira
                if an in {'لِ-','بِ'}:
                    diac = re.sub(r'([لب][َُِ]?)',r'\1-',diac)

            # RULE 5: CAPITALIZATION RULES 
            capschar = '±' 
            
            if idx == 0 and not diac.endswith(capschar):
                diac = diac + capschar
            elif idx > 0:
                prev_diac = disamb_words[idx-1].analyses[0].analysis.get('diac')
                
                if prev_diac == '.':
                    diac = diac + capschar

                # capitalize if gloss is capitalized and pos is proper noun or adjective 
                # (and word is not arabic punctuation and not capitalized for other reasons)
                elif analysis.get('pos') in {'noun_prop', 'adj'} and \
                     analysis.get('gloss') and analysis.get('gloss')[0].isupper() and \
                     tok not in {'،','؛'} and not diac.endswith(capschar):
                    diac = diac + capschar
                
                 # elif pos in nounprop NOTE: this is to make sure proper nouns are capitalized regardless of gloss 
                 # and other conditions but maybe better to collapse with previous condition
                elif analysis.get('pos') == 'noun_prop' and not diac.endswith(capschar):
                    diac = diac + capschar
            
            if '+' in diac:
                tmp = diac.replace('+','- ').split(' ')
                rom_tokens += tmp
            else:
                rom_tokens.append(diac)

        transliterated_words = [] 
        for diac in rom_tokens:
            if diac:
                # capitalize  NOTE: THEN tranlisterate so as to remove final capschar for proper transliteration
                if diac.endswith(capschar):
                    diac = diac[:-1]
                    transliterated_diac = translit_rules.translit(diac,loc_map)
                    transliterated_diac = capitalize_loc(transliterated_diac)
                else:
                    transliterated_diac = translit_rules.translit(diac,loc_map)
                transliterated_words.append(transliterated_diac)      
            else:
                if diac == '':
                    pass
                else:
                    print(f'whats this diac?: <{diac}>') #TODO: remove this since it doesn't seem to be doing anything

        transliterated_sentence = ' '.join(transliterated_words)
        arabic_sentence = ' '.join(diacritized_words)

        transliterated_sentence = translit_rules.recompose(transliterated_sentence,mode='rom') 
        predictions.append(transliterated_sentence)
        arabic_predictions.append(arabic_sentence)

    with open(output_txt, 'w', encoding='utf-8') as o:
        for line in predictions:
            o.write(f"{line}\n")

    with open(output_arabic_txt, 'w', encoding='utf-8') as o:
        for line in arabic_predictions:
            o.write(f"{line}\n")

    print(f"Wrote {len(predictions)} lines to {output_txt}")
    print(f"Wrote {len(arabic_predictions)} lines to {output_arabic_txt}")
    print(f"Unique prc0 found: {sorted(list(prc0_set))}")
    print(f"Unique prc1 found: {sorted(list(prc1_set))}")


if __name__ == "__main__":
    main()