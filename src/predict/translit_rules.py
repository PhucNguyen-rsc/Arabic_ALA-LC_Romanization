"""
Transliteration rules for Arabic to ALA-LC romanization.

This module provides functions for transliterating Arabic text to ALA-LC romanization.
It includes both simple character-by-character transliteration and more complex
morphology-aware transliteration using MADAMIRA analysis.
"""

import os
import re
import sys
import json
import logging
import argparse
import pandas as pd
from pathlib import Path
from repackage import up
up()

from data.make_dataset import tokenize_skiphyph, recompose
from madamira import analyse
from utils import log_durations

# Set up project paths
project_dir = str(Path(__file__).resolve().parents[2])

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{project_dir}/reports/translit_rules.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up rule logging
class RuleLogger:
    def __init__(self):
        self.morph_rules = {}
        self.exceptional_rules = {}
    
    def count_morph_rules(self, rule_tuple):
        self.morph_rules[rule_tuple] = self.morph_rules.get(rule_tuple, 0) + 1
    
    def count_exceptional_rules(self, rule_tuple):
        self.exceptional_rules[rule_tuple] = self.exceptional_rules.get(rule_tuple, 0) + 1
    
    def write_rules(self, setname, name):
        # write morph rules
        rule_dir = Path(f'{project_dir}/reports')
        rule_dir.mkdir(exist_ok=True)
        
        with open(f'{project_dir}/reports/morph_rules_freq-{setname}_{name}.tsv', 'w') as o:
            o.write('rule\ttype\tcount\n')
            for (rule, type_), count in sorted(self.morph_rules.items(), key=lambda x: x[1], reverse=True):
                o.write(f'{rule}\t{type_}\t{count}\n')
        
        with open(f'{project_dir}/reports/exceptional_rules_freq-{setname}_{name}.tsv', 'w') as o:
            o.write('arabic\tromanized\tcount\n')
            for (arabic, romanized), count in sorted(self.exceptional_rules.items(), key=lambda x: x[1], reverse=True):
                o.write(f'{arabic}\t{romanized}\t{count}\n')
        
        with open(f'{project_dir}/reports/translit_rules_freq-{setname}_{name}.tsv', 'w') as o:
            o.write('arabic\tromanized\tcount\n')
            for (arabic, romanized), count in sorted(self.exceptional_rules.items(), key=lambda x: x[1], reverse=True):
                o.write(f'{arabic}\t{romanized}\t{count}\n')

rules_logger = RuleLogger()

def load_loc_mappings():
    """
    Load the ALA-LC character mapping from JSON file
    """
    with open(f'{project_dir}/src/predict/loc_map.json', 'r') as f:
        return json.load(f)

def load_exceptional_spellings():
    """
    Load exceptional spellings for Arabic words from JSON file
    """
    with open(f'{project_dir}/src/predict/exceptional_spellings.json', 'r') as f:
        return json.load(f)

def translit(token, mapdict):
    """
    Transliterate a single Arabic token using character mapping
    
    Args:
        token: Arabic string to transliterate
        mapdict: Dictionary mapping Arabic characters to ALA-LC romanization
        
    Returns:
        Romanized string
    """
    transliterated = []
    i = 0
    while i < len(token):
        # Try to match multi-character sequences first
        matched = False
        for seq_len in range(min(3, len(token) - i), 0, -1):
            seq = token[i:i+seq_len]
            if seq in mapdict:
                transliterated.append(mapdict[seq])
                i += seq_len
                matched = True
                break
        
        # If no match found, keep the character as is
        if not matched:
            transliterated.append(token[i])
            i += 1
    
    return ''.join(transliterated)

def capitalize_loc(word): # for capitalizing hyphen '-' separated words and words beginning with hamza or 3ayn
    """
    Apply proper capitalization rules for ALA-LC romanization
    
    This function handles special cases like:
    - Words with hyphens (capitalize each part)
    - Words beginning with hamza or ayn (capitalize the second letter)
    
    Args:
        word: Romanized word to capitalize
        
    Returns:
        Properly capitalized word according to ALA-LC standards
    """
    capitalize_symbol = '±'
    
    # Remove capitalize symbol
    if word.endswith(capitalize_symbol):
        word = word[:-1]

    # In case of hyphens
    if '-' in word and not word.endswith('-') and not word.startswith('-'):
        split_tokens = word.split('-')
        main_token = split_tokens[-1]
        first_letter = main_token[0]
        # In case of hamza or 3ayn, next letter is capitalized
        if first_letter in {'ʼ','ʻ'} and len(main_token) > 1:  
            chars = [x for x in main_token]
            chars[1] = chars[1].capitalize()
            main_token = ''.join(chars)
        elif first_letter in {'ʼ','ʻ'} and len(main_token) <= 1:
            # Handle the case where the token is just a hamza/ayn
            main_token = first_letter
        else:
            main_token = main_token.capitalize()
        capitalized = '-'.join(split_tokens)

    # For strings with no hyphen
    else:
        # In case of hamza or 3ayn, next letter is capitalized
        if len(word)>1 and word[0] in {'ʼ','ʻ'}: 
            chars = [x for x in word]
            chars[1] = chars[1].capitalize()
            capitalized =  ''.join(chars)
        elif len(word)==1 and word[0] in {'ʼ','ʻ'}:
            # Handle the case where the token is just a hamza/ayn
            capitalized = word
        # In normal case without hamza, 3ayn, or hyphen
        else:
            capitalized =  word.capitalize()
    return capitalized


def translit_simple(sentence,mapdict,exceptional_spelling_dict,logger=rules_logger):
    '''
    Simple transliteration with exceptional spellings and first token capitalization
    
    Args:
        sentence: Arabic text to transliterate
        mapdict: Dictionary mapping Arabic characters to ALA-LC romanization
        exceptional_spelling_dict: Dictionary of special cases with custom romanizations
        logger: Logger object to track rule applications
        
    Returns:
        Romanized text with basic rules applied
    '''
    transliterated = []
    sentence = str(sentence)
    for tok_index,token in enumerate(sentence.split()):
        # Exceptional replacement
        if token in exceptional_spelling_dict: # exceptional spelling
            logger.count_exceptional_rules((token,exceptional_spelling_dict[token]))
            token = exceptional_spelling_dict[token]

        transliterated.append(translit(token,mapdict)) # map arabic character to LOC romanization
        if tok_index == 0: # Capitalize the first token
            logger.count_morph_rules(('index-0 capitalize','capitalized'))
            transliterated[-1] = capitalize_loc(transliterated[-1])
    return ' '.join(transliterated)

# def translit_morph(mada_sentnece_object,loc_mapdict,exceptional_spellings):

def get_diac(analysis):
    """
    Extract diacritized forms from analysis object
    
    Args:
        analysis: Analysis object containing sentence data
        
    Returns:
        Recomposed string with diacritics
    """
    diacritized = []
    for sent in analysis:
        for word in sent.words:
            diac_word = sent.analysis[word]['diac']
            diacritized.append(diac_word)
    return recompose(' '.join(diacritized))

# def translit_diac(diac):
#     diacritized =


def translit_morph(mada_sentnece_object,loc_mapdict,exceptional_spellings,logger=rules_logger): #TODO: 1) break up into smaller functions. 2) turn transliterator into class with constructor
    '''
    Apply MADAMIRA-style post-processing rules to romanize Arabic text
    
    This function takes a MADAMIRA sentence object (containing words, tokens, and analyses)
    and applies a series of linguistic rules to properly romanize Arabic text according
    to ALA-LC standards. It handles exceptional spellings, case endings, ta marbuta,
    clitics, capitalization, and more.
    
    Args:
        mada_sentnece_object: Object from parse_analyser_output() with words, tokens, and analyses
        loc_mapdict: Dictionary mapping Arabic characters to their romanized forms
        exceptional_spellings: Dictionary of special cases with custom romanizations
        logger: Logger object to track rule applications (default: rules_logger)
        
    Returns:
        String containing properly romanized text with all rules applied
    '''
    # Special character used to mark tokens for capitalization
    # This is processed at the end when converting to romanized form
    capschar = '±'
    
    # Extract data from the MADAMIRA sentence object
    words = mada_sentnece_object.words  # Original Arabic words
    toks = mada_sentnece_object.toks    # Diacritized and tokenized words
    sentence_analysis = mada_sentnece_object.analysis  # Morphological analysis for each word
    
    # Will hold modified Arabic words before final romanization
    modified_words = []
    
    # Process each token in the sentence
    for tidx in range(len(toks)):
        # Get current word, token, and its analysis
        word = words[tidx]
        tok = toks[tidx]
        analysis = sentence_analysis[word]

        # EXCEPTIONAL SPELLINGS
        # Handle words with special/exceptional romanization patterns
        # These override all other rules and processing
        if word in exceptional_spellings:
            logger.count_exceptional_rules((word,exceptional_spellings[word]))
            tok = exceptional_spellings[word]
            
            # Mark sentence-initial tokens for capitalization
            if tidx == 0 and not tok.endswith(capschar):
                logger.count_morph_rules(('index-0 capitalize','capitalized'))
                tok = tok+capschar

            modified_words.append(tok)
            continue  # Skip the rest of processing for this token

        ## MORPHOLOGICAL RULE PROCESSING BEGINS

        # Set up look-ahead for context-aware rules
        # This allows rules to consider the next word when making decisions
        if tidx < len(toks)-1:
            nextword = words[tidx+1]
            nexttok = toks[tidx+1]
            nextanalysis = sentence_analysis[nextword]
        else:
            nextword = None
            nexttok = None
            nextanalysis = None
        
        # Extract Buckwalter representation for morphological analysis
        # This provides detailed information about the word structure
        bw = analysis['bw']
        bwsplit = bw.split('+')  # Split into morphemes
        bwending = bwsplit[-1]   # Last morpheme (usually contains case info)
        bwbeginning = bwsplit[0] # First morpheme (usually contains proclitics)
        
        # RULE 1: 'LIL' RULE
        # When the preposition 'li' (to/for) combines with the definite article 'al',
        # they form 'lil' in Arabic (لِل) instead of 'li-al' (لِ+ال)
        if analysis['prc0'] == 'Al_det' and analysis['prc1'] == 'li_prep':
            logger.count_morph_rules(('li+al','lil'))
            find = re.escape('لِ+ال')
            tok = re.sub(find,r'لِل',tok)

        # RULE 2: CASE ENDINGS REMOVAL
        # Arabic has grammatical case markers that are often omitted in romanization
        
        # Exception: If word ends with direct object or possessive pronoun, keep case endings
        # These are integral to the word's pronunciation and meaning
        if ('DO' in bwending) or ('POSS_PRON' in bwending): 
            logger.count_morph_rules(('case-ending','kept'))
            pass
        # If word ends with case marker or certain suffixes, remove the final diacritics
        elif ('CASE' in bwending) or ('IV' in bwending) or ('PV' in bwending) or ('CV' in bwending) or ('NSUFF' in bwending):
            logger.count_morph_rules(('case-ending','removed'))
            # Remove alif tanween (ـًا) first - this is the accusative indefinite marker
            tok = re.sub(r'اً(±)?$',r'\1',tok)
            # Remove other diacritics (ـَ ـُ ـِ ـً ـٌ ـٍ) at the end of the word
            # The (±)? pattern preserves the capitalization marker if present
            tok = re.sub(r'[ًٌٍَُِ](±)?$',r'\1',tok)

        # RULE 3: TA MARBUTA HANDLING
        # In Arabic, taa marbuta (ة) is pronounced as 'h' in pausal form
        # but as 't' when in construct state (idafa/إضافة)
        if 'ة' in tok:
            logger.count_morph_rules(('ta-marbuta','total'))
            # Check if word is in construct state
            if analysis['stt'] == 'c':
                # Exception: Cannot be construct if followed by preposition
                # This handles cases where MADAMIRA analysis is incorrect
                if nextanalysis and 'PREP' in nextanalysis['bw'].split('+')[0]:
                    logger.count_morph_rules(('ta-marbuta','not-construct (followed by prep)'))
                    pass
                else:
                    logger.count_morph_rules(('ta-marbuta','construct'))
                    # Add sukun (ْ) on ta-marbuta to force its transliteration as 't' instead of 'h'
                    tok = re.sub(r'ة',r'ةْ',tok)

        # RULE 4: SINGLE LETTER PROCLITICS
        # Arabic has single-letter prepositions (ب/bi, ل/li)
        # These should be separated from the main word with hyphens in romanization
        if 'PREP' in bwbeginning and len(bwsplit)>1:  # Length condition ensures it's a real proclitic
            an = analysis['lemma'].split('_')[0]
            # Only handle the common single-letter prepositions ب (bi) and ل (li)
            if an in {'لِ-','بِ'}:
                logger.count_morph_rules(('single-letter clitic','split'))
                # Add hyphen after the single letter: 'ب' → 'ب-', 'ل' → 'ل-'
                # The [َُِ]? pattern matches optional diacritics on the letter
                tok = re.sub(r'([لب][َُِ]?)',r'\1-',tok)
  
        # RULE 5: CAPITALIZATION RULES
        # Mark tokens for capitalization based on position and type
        
        # Capitalize sentence-initial tokens
        if tidx == 0 and not tok.endswith(capschar):
            logger.count_morph_rules(('index-0 capitalize','capitalized'))
            tok = tok+capschar
        # Capitalization rules for non-initial tokens
        elif tidx > 0:
            # Get the previous token for context
            before = toks[tidx-1]

            # Capitalize after period (sentence boundary)
            if before == '.':
                logger.count_morph_rules(('after . capitalize','capitalized'))
                tok = tok+capschar
            
            # Capitalize proper nouns and adjectives with capitalized glosses
            # Skip Arabic punctuation and already capitalized tokens
            if analysis['pos'] in {'noun_prop','adj'} and analysis['gloss'] and analysis['gloss'][0].isupper() and word not in {'،','؛'} and not tok.endswith(capschar):
                logger.count_morph_rules(('adj/nounprop capitalized gloss','capitalized'))
                tok = tok+capschar

            # Ensure all proper nouns are capitalized regardless of gloss
            elif analysis['pos'] in {'noun_prop'} and not tok.endswith(capschar):
                logger.count_morph_rules(('nounprop','capitalized'))
                tok = tok+capschar
   
        # TOKENIZATION HANDLING
        # Process tokens with '+' markers (indicating morpheme boundaries)
        if '+' in tok:
            # Replace '+' with '- ' and split into separate tokens
            # This converts morpheme boundaries to proper hyphenated forms
            tmp = tok.replace('+','- ').split(' ') 
            modified_words += tmp
        else:
            # Add the token as is if no '+' markers
            modified_words.append(tok)
            
    # FINAL ROMANIZATION
    # Convert modified Arabic tokens to romanized form
    transliterated_words = [] 
    for tok in modified_words:
        if tok:
            # Process capitalization markers
            if tok.endswith(capschar):
                # Remove capitalization marker before transliteration
                tok = tok[:-1]
                # Transliterate first, then apply capitalization
                transliterated_tok = translit(tok,loc_mapdict)
                transliterated_tok = capitalize_loc(transliterated_tok)
            else:
                # Standard transliteration for non-capitalized tokens
                transliterated_tok = translit(tok,loc_mapdict)
                
            transliterated_words.append(transliterated_tok)      
        else:
            # Skip empty tokens
            if tok == '':
                pass
            else:
                # Debug output for unexpected token types
                print(f'whats this tok?: <{tok}>') #TODO: remove this since it doesn't seem to be doing anything

    # Join all transliterated tokens into a sentence
    transliterated_sentence = ' '.join(transliterated_words)
    
    # Apply final recomposition to handle any remaining issues
    return recompose(transliterated_sentence,mode='rom') 





@log_durations(logging.info)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default='data/processed/dev.tsv', help='path to file containing Arabic lines, must specify -i txt or -i tsv with optional headername, default: -i tsv ar ')
    parser.add_argument('output', help= 'specify output location for predictions')
    parser.add_argument('-i','--input_type', required=True, nargs='+', default=['tsv','ar'], help='options: 1)txt: if file is single column txt file (no header); 2)tsv <optional:input-header>: input is multicolumn tsv.  unless specified, header defaults to "ar"') #TODO: add csv option?
    parser.add_argument('-m','--mode', required=True, help= 'options: 1)translit_simple: apply rules to raw text without diacritics or morphological information; 2)translit_morph: apply rules to morphologically analyzed text')
    parser.add_argument('-da','--dont_reanalyse', action='store_true',help= "don't reanalyse sentences as analysis is already saved")
    parser.add_argument('-l','--log_rules', action='store_true',help= "create a rule freq tsv in reports directory") #TODO: make logger counts optional
    args = parser.parse_args()
    
    setname = args.input.split('/')[-1].replace('.tsv','')
    name = args.mode.split('_')[-1]
    print(f'predicting {args.input}')
    logging.info(f'predicting {args.input}')    


    # input
    input_type = {idx:value for idx,value in enumerate(args.input_type)}
    if input_type[0]=='tsv':
        lines = pd.read_csv(args.input,delimiter='\t')
        if input_type.get(1,'ar') not in lines.columns:
            raise Exception('specified input column is not in input file')
        ar_lines = lines[input_type.get(1,'ar')] # defaults to 'ar' column if no column is specified
    elif input_type[0] == 'txt':
        ar_lines = pd.read_csv(args.input,delimiter='\t',header=None)[0]
    else:
        raise Exception('no -i --input_type selected, please choose -i txt for single column input or -i tsv <optional column name; default: ar > for tsv')

    # load loc mappings and exceptional spellings
    locmap = load_loc_mappings()
    locexceptional = load_exceptional_spellings()

    # predict
    if args.mode == 'simple':
        predictions = ar_lines.apply(lambda x: translit_simple(x,locmap,locexceptional))

    elif args.mode == 'morph':
        Path(f'{project_dir}/data/processed_for_madamira/analyser_input').mkdir(parents=True,exist_ok=True)
        Path(f'{project_dir}/data/processed_for_madamira/analyser_output').mkdir(parents=True,exist_ok=True)
        
        analyse_input_path = f'{project_dir}/data/processed_for_madamira/analyser_input/{setname}.xml'
        analyse_output_path = f'{project_dir}/data/processed_for_madamira/analyser_output/{setname}.xml'
        

        if not args.dont_reanalyse:  #TODO: change this to see if analyse_output_path exists and ask if reanalysis is necessary
            # tokenize to make ready for config file
            tokenized_ar_lines = list(ar_lines.apply(tokenize_skiphyph).values)

            # generate config file
            analyser_input = analyse.generate_analyser_input(tokenized_ar_lines)
            
            # write config file
            analyse.write_xml(analyser_input,analyse_input_path)

            # anlyse
            analyse.analyse_standalone(analyse_input_path,analyse_output_path)

        # load analysis
        analysis = analyse.load_analysis(analyse_output_path)

        # parse analysis
        analysis_sentences = analyse.parse_analyser_output(analysis)
        
        # transliterate
        predictions = []
        for sentence in analysis_sentences:
            predictions.append(translit_morph(sentence,locmap,locexceptional))
    else:
        raise Exception('mode not recognized')
    
    # write predictions
    with open(args.output,'w') as o:
        for prediction in predictions:
            o.write(prediction+'\n')
    
    # write rule frequencies
    if args.log_rules:
        rules_logger.write_rules(setname, name)
    
    print(f'wrote predictions to {args.output}')
    logging.info(f'wrote predictions to {args.output}')


if __name__ == '__main__':
    main()