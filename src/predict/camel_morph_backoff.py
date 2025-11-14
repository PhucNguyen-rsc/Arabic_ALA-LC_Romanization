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

# Dictionary mapping clitic codes to their romanized forms with hyphens
PROCLITIC_MAP = {
    # prc0 (including the existing Al_det)
    'Al_det': 'al-',
    'mA_neg': 'mā-',
    'mA_part': 'mā-',
    'mA_rel': 'mā-',
    'lA_neg': 'lā-',
    
    # prc1
    'bi_prep': 'bi-',
    'ka_prep': 'ka-',
    'li_prep': 'li-',
    'li_jus': 'li-',
    'la_emph': 'la-',
    'la_rc': 'la-',
    'la_prep': 'la-',
    'fiy_prep': 'fī-',
    'ta_prep': 'ta-',
    'wa_prep': 'wa-',
    'sa_fut': 'sa-',
    'yA_voc': 'yā-',
    'wA_voc': 'wā-',
    'hA_dem': 'hā-',
    'bi_part': 'bi-',
    
    # prc2
    'fa_conj': 'fa-',
    'fa_conn': 'fa-',
    'fa_rc': 'fa-',
    'fa_sub': 'fa-',
    'wa_conj': 'wa-',
    'wa_part': 'wa-',
    'wa_sub': 'wa-',
    
    # prc3
    '>a_ques': 'a-'
}

ENCLITIC_MAP = {
    # enc0
    'hu': '-hu',
    'hA': '-hā',
    'hum': '-hum',
    'humA': '-humā',
    'hunna': '-hunna',
    'ka': '-ka',
    'kum': '-kum',
    'kumA': '-kumā',
    'kunna': '-kunna',
    'nA': '-nā',
    'niy': '-nī',
    'ya': '-ya',
    'ma': '-mā',   # Interrogative particle
    'mA': '-mā'    # Indefinite particle
}

def handle_clitics(analysis, word):
    """Process proclitics and enclitics to add proper hyphenation.
    This function is aligned with the clitic handling in translit_rules.py.
    """
    result = word
    
    # Handle proclitics (prefixes)
    for prc_type in ['prc0', 'prc1', 'prc2', 'prc3']:
        if prc_type in analysis and analysis[prc_type] in PROCLITIC_MAP:
            prefix = PROCLITIC_MAP[analysis[prc_type]]
            # Special handling for 'Al_det' to ensure 'al-' is always lowercase and hyphenated
            if analysis[prc_type] == 'Al_det':
                # Ensure 'Al' or 'AL' at the beginning of the word becomes 'al-'
                if result.lower().startswith('al') and not result.startswith('al-'):
                    result = 'al-' + result[2:]
                elif result.startswith('Al') and not result.startswith('al-'):
                    result = 'al-' + result[2:]
                elif result.startswith('AL') and not result.startswith('al-'):
                    result = 'al-' + result[2:]
                # If the word already starts with 'al-' (e.g., from transliteration), ensure it's correct
                elif result.startswith('al-'):
                    pass
                else: # Prepend if not already handled
                    result = prefix + result
            else:
                # For other proclitics, if the word already starts with this prefix without a hyphen, replace it
                if result.lower().startswith(prefix.replace('-', '')):
                    result = prefix + result[len(prefix.replace('-', '')):]  # Fixed to use the full prefix length
                else:
                    # Otherwise prepend it (though this case should be rare)
                    result = prefix + result
    
    # Handle enclitics (suffixes)
    if 'enc0' in analysis and analysis['enc0'] in ENCLITIC_MAP:
        suffix = ENCLITIC_MAP[analysis['enc0']]
        # Check if the word already ends with this suffix without a hyphen
        if result.lower().endswith(suffix.replace('-', '')):
            result = result[:-len(suffix.replace('-', ''))] + suffix
        else:
            result = result + suffix
    
    # For compound clitics - implement patterns from translit_rules.py's fix_clitic_attachments
    # li-al -> lil-
    result = re.sub(r'li-al-', r'lil-', result)
    # bi-al -> bil-
    result = re.sub(r'bi-al-', r'bil-', result)
    # wa-al -> wal-
    result = re.sub(r'wa-al-', r'wal-', result)
    # wa-al -> wal-
    result = re.sub(r'wa-al-', r'wal-', result)
    
    # Fix duplicated characters that appear in the comparison output
    # This pattern finds duplicated consonants at word boundaries
    result = re.sub(r'([^aeiouāīū])\1\b', r'\1', result)  # Word ending
    result = re.sub(r'r\b', r'r', result)  # Fix specifically 'Dārr' → 'Dār'
    
    # Remove extra periods at the end of words
    result = re.sub(r'\.+\b', r'.', result)
    
    return result

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

def fix_case_endings(romanized_tokens, analyses_list):
    """
    Fix inconsistent case endings by using morphological analysis to determine
    when to use -t vs -h for taa marbuta
    """
    for i, (rom_token, analysis_obj) in enumerate(zip(romanized_tokens, analyses_list)):
        # Skip if no analysis available
        if not analysis_obj.analyses:
            continue
            
        # Check if token ends with 'h' (potential taa marbuta)
        if rom_token.endswith('h'):
            # Extract top analysis
            top_analysis = analysis_obj.analyses[0].analysis
            
            # Check if the Arabic word has taa marbuta in construct state
            diac = top_analysis.get('diac', '')
            
            # In construct state (iḍāfa), taa marbuta should be -t
            if 'idafa' in top_analysis.get('features', '').lower() or \
               'construct' in top_analysis.get('features', '').lower():
                romanized_tokens[i] = rom_token[:-1] + 't'
            
            # For titles and specific patterns, always use -t
            if rom_token.lower().endswith(('ah 1.', 'ah 2.', 'ah 3.', 'ah 4.', 'ah 5.')):
                romanized_tokens[i] = rom_token[:-1] + 't'
    
    return romanized_tokens

def fix_titles_and_bibliographic_terms(romanized_tokens):
    """
    Fix capitalization of titles and bibliographic terms
    """
    # Comprehensive list of bibliographic terms
    BIBLIO_TERMS = {
        'ṭabʻah': 'Ṭabʻah',        # edition
        'maktabat': 'Maktabat',     # library
        'jāmiʻat': 'Jāmiʻat',       # university
        'maṭbaʻat': 'Maṭbaʻat',     # press
        'dār': 'Dār',               # house
        'muʼassasat': 'Muʼassasat', # foundation
        'majallat': 'Majallat',     # journal
        'kitāb': 'Kitāb',           # book
        'rasāʼil': 'Rasāʼil',       # essays/letters
        'sharḥ': 'Sharḥ',           # commentary
        'fiqhīyah': 'Fiqhīyah',     # jurisprudential
        'thawrah': 'Thawrah',       # revolution
        'falsafah': 'Falsafah',     # philosophy
        'islām': 'Islām',           # Islam
        'markaz': 'Markaz',         # center (added from comparison)
        'maqūlāt': 'Maqūlāt',       # sayings
        'maʻlahī': 'Maʻlahī',       # royal
        'hayʼah': 'Hayʼah',         # organization/committee
        'iḏārat': 'Iḏārat',         # administration
        'siyar': 'Siyar',           # biographies
        'riwāyāt': 'Riwāyāt',       # novels
        'muqaddimāt': 'Muqaddimāt', # introductions
        'taqwīm': 'Taqwīm',         # calendar/evaluation
        'thaqafat': 'Thaqafat',     # culture
        'tanzīm': 'Tanzīm',         # organization
        'tafsīr': 'Tafsīr',         # interpretation
    }
    
    for i, token in enumerate(romanized_tokens):
        # Check for bibliographic terms after al-
        if token.lower().startswith('al-'):
            term_after_al = token[3:].lower()
            
            for term, capitalized in BIBLIO_TERMS.items():
                if term_after_al.startswith(term):
                    romanized_tokens[i] = f'al-{capitalized}'
                    break
        
        # Check for bibliographic terms at start of token
        else:
            for term, capitalized in BIBLIO_TERMS.items():
                if token.lower().startswith(term) and (i == 0 or romanized_tokens[i-1].endswith('.')):
                    # Check if this is the first token in a bibliographic entry
                    romanized_tokens[i] = capitalized + token[len(term):]
                    break
    
    return romanized_tokens

def fix_clitic_attachments(romanized_tokens):
    """
    Fix incorrect clitic attachments, especially compound clitics
    """
    for i, token in enumerate(romanized_tokens):
        # Fix li-al-L pattern
        token = re.sub(r'li-al-Li([A-Za-z])', r'lil-\1', token)
        
        # Fix other common clitic patterns
        token = re.sub(r'bi-al-([A-Za-z])', r'bil-\1', token)
        token = re.sub(r'wa-al-([A-Za-z])', r'wal-\1', token)
        
        # Fix missing space after comma in names
        token = re.sub(r',([A-Za-z])', r', \1', token)
        
        romanized_tokens[i] = token
    
    return romanized_tokens

def fix_name_capitalization(romanized_tokens, analyses_list):
    """
    Fix capitalization of personal names and related terms
    """
    # Common names and special terms
    NAME_DICT = {
        'allāh': 'Allāh',
        'Āllāh': 'Allāh',
        'allāh,': 'Allāh,',
        'muḥammad': 'Muḥammad',
        'aḥmad': 'Aḥmad',
        'mullā': 'Mullā',
        'malaʼā': 'Mullā',
        'ulugh': 'Ulugh',  # Added based on comparison output
        'beg': 'Beg',      # Added based on comparison output
        'ʻalī': 'ʻAlī',    # Added based on comparison output
        'ḥasan': 'Ḥasan',  # Added based on comparison output
        'abū': 'Abū',      # Added based on comparison output
    }
    
    for i, (token, analysis_obj) in enumerate(zip(romanized_tokens, analyses_list)):
        # Apply name dictionary fixes
        lower_token = token.lower()
        for name_lower, correct_name in NAME_DICT.items():
            if name_lower == lower_token:
                romanized_tokens[i] = correct_name
                break
        
        # Fix capitalization for proper nouns
        if analysis_obj.analyses and analysis_obj.analyses[0].analysis.get('pos') == 'noun_prop':
            # Handle specific capitalization patterns for proper nouns
            if token.startswith("'") or token.startswith("ʻ"):
                # Capitalize letter after apostrophe
                if len(token) > 1 and not token[1].isupper():
                    romanized_tokens[i] = token[0] + token[1].upper() + token[2:]
    
    return romanized_tokens

def fix_vowels_and_diacritics(romanized_tokens):
    """
    Fix inconsistent vowel romanization - aligned with translit_rules.py patterns
    """
    vowel_fixes = {
        'yw': 'yū',     # yū - matching long vowel pattern
        'iy': 'ī',     # ī - matching long vowel pattern
        'ww': 'ū',     # ū - matching long vowel pattern
        'yy': 'ī',     # ī - Additional common case
        'aa': 'ā',     # ā - Additional common case
        'ywlywa': 'yūlyū',  # Specific fix for the month of July
        'Ywlywa': 'Yūlyū',  # Capitalized version
        'Ywlyw': 'Yūlyū',   # Another variant
        'ywlyw': 'yūlyū',   # Another variant
        # Additional vowel fixes from translit_rules patterns
        'wy': 'wī',     # wī
        'ay': 'ay',       # Keep as is
        'aw': 'aw'        # Keep as is
    }
    
    for i, token in enumerate(romanized_tokens):
        # Apply vowel fixes
        for incorrect, correct in vowel_fixes.items():
            token = token.replace(incorrect, correct)
        
        # Fix missing hyphen in names
        token = re.sub(r'al([A-Z])', r'al-\1', token)
        
        # For numbers at end of words (years, dates)
        if re.search(r'\d+h$', token):
            token = re.sub(r'(\d+)h$', r'\1', token)  # Remove 'h' after numbers
        
        romanized_tokens[i] = token
    
    return romanized_tokens

def fix_common_expressions(romanized_tokens):
    """
    Fix common multi-word expressions
    """
    # Dictionary of expressions to fix (lowercase for matching)
    expressions = {
        "fī mā": "fī-mā",
        "min mā": "mimmā",
        "ʻan mā": "ʻammā",
        "bi mā": "bi-mā",
        "li mā": "li-mā"
    }
    
    # Look for bigrams that match expressions
    i = 0
    while i < len(romanized_tokens) - 1:
        bigram = f"{romanized_tokens[i].lower()} {romanized_tokens[i+1].lower()}"
        if bigram in expressions:
            # Replace with the fixed expression
            romanized_tokens[i] = expressions[bigram]
            # Remove the second token
            romanized_tokens.pop(i+1)
        else:
            i += 1
    
    return romanized_tokens

def handle_sun_letter_assimilation(romanized_tokens):
    """
    Handle sun letter assimilation with al- prefix
    """
    # Sun letters - letters that cause assimilation of the lam in al-
    sun_letters = {'t', 'd', 'r', 'z', 's', 'sh', 'ṣ', 'ḍ', 'ṭ', 'ẓ', 'l', 'n'}
    
    for i, token in enumerate(romanized_tokens):
        # Check if token starts with al- followed by a sun letter
        for sun in sun_letters:
            if re.search(rf'^al-{sun}', token.lower()):
                # In ALA-LC, we don't actually show the assimilation in writing
                # but in case it's needed for some specific formatting:
                # token = re.sub(rf'^al-({sun})', rf'a{sun}-\1', token.lower())
                pass
        
        romanized_tokens[i] = token
    
    return romanized_tokens

def handle_hamzat_wasl(romanized_tokens):
    """
    Handle hamzat al-wasl (initial alif/hamza when preceded by vowels)
    """
    for i in range(1, len(romanized_tokens)):
        # Check if current token starts with al- and previous token ends with a vowel
        if (romanized_tokens[i].lower().startswith('al-') and 
            i > 0 and romanized_tokens[i-1] and 
            romanized_tokens[i-1][-1].lower() in 'aeiouāīū'):
            # In ALA-LC, we don't actually change al- to l-, but some systems might
            # romanized_tokens[i] = romanized_tokens[i].replace('al-', 'l-', 1)
            pass
    
    return romanized_tokens

def handle_name_format_reordering(tokens, analyses_list):
    """
    Handle name format reordering for bibliographic entries
    
    In bibliographic entries, names are often reordered from the Arabic form
    "First Last" to the Western form "Last, First". This function detects such
    patterns and applies the reordering.
    """
    # Check if this sentence looks like a bibliographic entry with a name
    if len(tokens) < 2:
        return tokens
    
    # Name patterns to detect (Arabic name format that should be reordered)
    # Example: "Ali bin Muhammad" -> "Bin Muhammad, Ali"
    
    # Build a list of common Arabic name patterns
    first_name_indicators = ['ibn', 'bin', 'b.', 'bt.', 'bint', 'abū', 'abī', 'abd']
    
    # Check if the first token is a common first name and the second contains a family name indicator
    if (any(a.analyses and a.analyses[0].analysis.get('pos') == 'noun_prop' 
            for a in analyses_list[:2] if a.analyses) and
            any(tokens[i].lower().startswith(prefix) for i in range(1, min(3, len(tokens))) 
                for prefix in first_name_indicators)):
        
        # This looks like a name that should be reordered
        # Extract the relevant parts: "First Second[+] Last"
        name_parts = []
        for i, token in enumerate(tokens[:3]):
            if i < len(analyses_list) and analyses_list[i].analyses and \
               analyses_list[i].analyses[0].analysis.get('pos') == 'noun_prop':
                name_parts.append(token)
        
        if len(name_parts) >= 2:
            # Reorder to "Last, First Second"
            last_name = name_parts[-1].rstrip('.,;:')
            first_names = ' '.join(name_parts[:-1]).rstrip('.,;:')
            tokens[0] = f"{last_name}, {first_names}."
            
            # Remove the other tokens that were part of the name
            return [tokens[0]] + tokens[len(name_parts):]
    
    return tokens

def fix_grammar_patterns(romanized_tokens, analyses_list):
    """
    Fix special grammatical patterns like dual forms
    
    This function handles special grammatical patterns like dual forms (-ayn endings)
    and other Arabic grammatical structures that need special handling in romanization.
    """
    for i, (token, analysis_obj) in enumerate(zip(romanized_tokens, analyses_list)):
        # Handle dual form endings
        if token.endswith('aynay') or token.endswith('ayny'):
            romanized_tokens[i] = token[:-5] + 'aynay'
        elif token.endswith('ayna'):
            romanized_tokens[i] = token[:-4] + 'ayn'
            
        # Handle li- + aynayn pattern (for eye/eyes)
        if token.startswith('Li-') and 'ayn' in token:
            romanized_tokens[i] = 'li-' + token[3:]
            
        # Handle specific patterns from observed errors
        if token == 'malaʼā':
            romanized_tokens[i] = 'Mullā'
            
    return romanized_tokens

def post_process_romanization(romanized_tokens, analyses_list):
    """
    Apply all post-processing fixes to romanized tokens
    
    This function orchestrates the application of multiple post-processing rules
    to improve romanization quality. Each rule is applied in sequence matching
    the order in translit_rules.py as closely as possible.
    """
    # Order of operations aligned with translit_rules.py's translit_morph function:
    
    # 1. Fix case endings: Correct -h/-t endings for taa marbuta based on grammatical state
    # This corresponds to the ta-marbuta handling in translit_rules.py
    romanized_tokens = fix_case_endings(romanized_tokens, analyses_list)
    
    # 2. Fix clitic attachments: Handle compound clitics like 'lil-', 'bil-', 'wal-'
    # This corresponds to various clitic rules in translit_rules.py
    romanized_tokens = fix_clitic_attachments(romanized_tokens)
    
    # 3. Fix vowel sequences: Convert sequences like 'yw' → 'yū', 'iy' → 'ī'
    # This corresponds to transliteration mapping in translit_rules.py
    romanized_tokens = fix_vowels_and_diacritics(romanized_tokens)
    
    # 4. Fix special grammar patterns like dual forms
    romanized_tokens = fix_grammar_patterns(romanized_tokens, analyses_list)
    
    # 5. Fix capitalization (proper nouns, titles, etc.)
    # This corresponds to capitalization rules in translit_rules.py
    romanized_tokens = fix_name_capitalization(romanized_tokens, analyses_list)
    
    # 6. Fix bibliographic terms: Capitalize terms like 'Ṭabʻah' (edition), 'Maktabat' (library)
    # This is part of capitalization handling in translit_rules.py
    romanized_tokens = fix_titles_and_bibliographic_terms(romanized_tokens)
    
    # 7. Fix common expressions: Handle multi-word expressions like "fī mā" → "fī-mā"
    # This corresponds to special handling in translit_rules.py
    romanized_tokens = fix_common_expressions(romanized_tokens)
    
    # 8. Handle sun letter assimilation: Special handling for 'al-' before certain letters
    # This is part of clitic handling in translit_rules.py
    romanized_tokens = handle_sun_letter_assimilation(romanized_tokens)
    
    # 9. Handle hamzat al-wasl: Special handling for 'al-' after vowels
    # This is part of special case handling in translit_rules.py
    romanized_tokens = handle_hamzat_wasl(romanized_tokens)
    
    # 10. Name format reordering for bibliographic entries
    # This is a special case for bibliographic formatting
    romanized_tokens = handle_name_format_reordering(romanized_tokens, analyses_list)
    
    return romanized_tokens
    
def remove_end_diacritics_from_token(token):
    # For a single token, we can just match diacritics at the end of the string
    return re.sub(r'[\u064B-\u0652]$', '', token)

def remove_end_diacritics(sentence):
    # Process a whole sentence by removing end diacritics from each word
    words = sentence.split()
    return ' '.join(remove_end_diacritics_from_token(word) for word in words)

def capitalize_using_pos(analyses_list, romanized_tokens):
    """
    Properly capitalize romanized tokens based on POS tags following ALA-LC rules:
    - Keep all clitics lowercase
    - Properly capitalize proper nouns
    - Handle special cases (ibn/bin between names, etc.)
    
    Args:
        analyses_list: List of DisambiguatedWord objects from CAMeL Tools
        romanized_tokens: List of romanized tokens to be modified
        
    Returns:
        List of romanized tokens with proper capitalization
    """
    # Common Arabic clitics that should always remain lowercase
    CLITICS = [
        'al-', 'bi-', 'li-', 'la-', 'wa-', 'fa-', 'ka-', 'sa-', 'fī-', 
        'fiy-', 'min-', 'ʻan-', 'ilá-', 'ʻalá-', 'lil-', 'bil-', 'wal-'
    ]

    # Add this at the beginning of your capitalize_using_pos function
    # Bibliographic terms that should always be capitalized after al-
    BIBLIO_TERMS = [
        'ṭabʻah',    # edition
        'maktabat',  # library
        'jāmiʻat',   # university
        'maṭbaʻat',  # press
        'dār',       # house/publishing house
        'muʼassasat', # institution/foundation
        'majallat',  # journal/magazine
        'kitāb',     # book
        'maʻhad',    # institute
        'markaz',    # center
        'kullīyat',  # college
        'qism',      # department
        'majlis',    # council
        'hayʼat',    # organization/commission
        'wizārat',   # ministry
        'mudīrīyat', # directorate
        'idārat',    # administration
        'nahḍah',    # renaissance
        'thawrah'    # revolution
    ]
    
    # Words that should be lowercase in specific contexts (like ibn between names)
    SPECIAL_LOWERCASE = ['ibn', 'bin', 'b.', 'bt.', 'bint']
    
    # First identify proper noun sequences to handle multi-word names
    is_proper_noun = [False] * len(romanized_tokens)
    for i, analysis_obj in enumerate(analyses_list):
        if not analysis_obj.analyses:
            continue
        
        top_analysis = analysis_obj.analyses[0].analysis
        if top_analysis.get('pos') == 'noun_prop':
            is_proper_noun[i] = True
    
    # First pass: Lowercase all clitics and fix special cases
    for i, rom_token in enumerate(romanized_tokens):
        # Lowercase all clitics
        for clitic in CLITICS:
            uppercase_clitic = clitic[0].upper() + clitic[1:]
            # Replace at beginning of word
            rom_token = re.sub(f'^{re.escape(uppercase_clitic)}', clitic, rom_token)
            # Replace after hyphen
            rom_token = re.sub(f'-{re.escape(uppercase_clitic)}', f'-{clitic}', rom_token)
        
        # Special case: ibn/bin/bint between names should be lowercase
        if i > 0 and i < len(romanized_tokens) - 1:
            lower_token = rom_token.lower()
            if (lower_token in SPECIAL_LOWERCASE and 
                    is_proper_noun[i-1] and is_proper_noun[i+1]):
                rom_token = lower_token
        
        romanized_tokens[i] = rom_token
    
    # Second pass: Apply capitalization rules for proper nouns
    for i, rom_token in enumerate(romanized_tokens):
        if not is_proper_noun[i]:
            # If it's not a proper noun and not start of sentence, ensure lowercase
            if i > 0 and romanized_tokens[i-1][-1] != '.':
                # Check if it's not already all lowercase
                if not rom_token.islower() and rom_token not in SPECIAL_LOWERCASE:
                    # But don't lowercase words with special characters
                    if re.match(r'^[a-zA-Z]+$', rom_token[0]):
                        rom_token = rom_token[0].lower() + rom_token[1:]
            continue
        
        # For proper nouns: capitalize each part except clitics
        parts = []
        current_part = ""
        in_clitic = False
        
        # Split by hyphens but preserve them
        hyphen_parts = re.split(r'(-)', rom_token)
        
        for part in hyphen_parts:
            if part == '-':
                parts.append(part)
                continue
                
            # Check if part is or starts with a clitic
            is_clitic = False
            for clitic in CLITICS:
                if part.lower().startswith(clitic.rstrip('-')):
                    # Keep the clitic lowercase
                    clitic_len = len(clitic.rstrip('-'))
                    if len(part) > clitic_len:
                        # Capitalize what follows the clitic
                        parts.append(part[:clitic_len].lower())
                        parts.append(part[clitic_len].upper() + part[clitic_len+1:])
                    else:
                        parts.append(part.lower())
                    is_clitic = True
                    break
            
            # If not a clitic, capitalize first letter
            if not is_clitic:
                # Only capitalize if not already capitalized
                if part and not part[0].isupper():
                    parts.append(part[0].upper() + part[1:])
                else:
                    parts.append(part)
        
        romanized_tokens[i] = ''.join(parts)
        
        # Special fix for al-X in proper nouns: the X should be capitalized
        romanized_tokens[i] = re.sub(r'al-([a-z])', lambda m: f'al-{m.group(1).upper()}', romanized_tokens[i])
    
    # Third pass: Fix specific capitalization issues
    for i, rom_token in enumerate(romanized_tokens):
        # Fix capitalization after apostrophes (ʻAbd not ʻabd)
        romanized_tokens[i] = re.sub(r'(^|\s|\-)([ʻʼ])([a-z])', 
                                     lambda m: f'{m.group(1)}{m.group(2)}{m.group(3).upper()}', 
                                     rom_token)

    for i, token in enumerate(romanized_tokens):
    # Check if token starts with al-
        if token.lower().startswith('al-'):
            # Check if it's a bibliographic term
            term_part = token[3:].lower()  # Get part after "al-"
            
            # Check if it's a bibliographic term or starts with one
            if any(term_part.startswith(term) for term in BIBLIO_TERMS):
                # Ensure "al-" is lowercase and what follows is capitalized
                if len(token) > 3:
                    romanized_tokens[i] = 'al-' + token[3].upper() + token[4:]
            
            # Special case for place names (if they're not already capitalized)
            elif len(token) > 3 and not token[3].isupper() and is_proper_noun[i]:
                # Place names should have the first letter capitalized after al-
                romanized_tokens[i] = 'al-' + token[3].upper() + token[4:]
    return romanized_tokens

def remove_non_arabic(text):
    """
    Remove all non-Arabic characters from text.
    
    Args:
        text: String containing mixed characters
        
    Returns:
        String with only Arabic characters
    """
    # Arabic Unicode range: U+0600 to U+06FF
    # Additional Arabic-related ranges: U+0750 to U+077F (Arabic Supplement)
    #                                  U+08A0 to U+08FF (Arabic Extended-A)
    #                                  U+FB50 to U+FDFF (Arabic Presentation Forms-A)
    #                                  U+FE70 to U+FEFF (Arabic Presentation Forms-B)
    arabic_pattern = re.compile(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    
    # Replace non-Arabic characters with empty string
    return arabic_pattern.sub('', text)


    """
    Remove all non-Arabic characters from text.
    
    Args:
        text: String containing mixed characters
        
    Returns:
        String with only Arabic characters
    """
    # Arabic Unicode range: U+0600 to U+06FF
    # Additional Arabic-related ranges: U+0750 to U+077F (Arabic Supplement)
    #                                  U+08A0 to U+08FF (Arabic Extended-A)
    #                                  U+FB50 to U+FDFF (Arabic Presentation Forms-A)
    #                                  U+FE70 to U+FEFF (Arabic Presentation Forms-B)
    arabic_pattern = re.compile(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    
    # Replace non-Arabic characters with empty string
    return arabic_pattern.sub('', text)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_tsv', help='TSV with column ar')
    parser.add_argument('output_txt', help='One ALA-LC line per sentence')
    parser.add_argument('--bert', action='store_true', help='Use CAMeL BERT disambiguator if available')
    parser.add_argument('--custom-analyzer', action='store_true', help='Use custom calima-msa-s31.db analyzer instead of default')
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
    # when use_gpu=True (which is the default setting)
    import torch
    if torch.cuda.is_available() and args.bert:
        print(f"GPU available: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
        print("Using GPU for BERT (handled by CAMeL Tools internally)")
    else:
        print("Using CPU for disambiguation (not using BERT or GPU not available)")

    predictions: list[str] = []
    sentences = tqdm(data['ar'].astype(str), desc="Romanizing", unit="sentence")

    for sentence in sentences:
        # Debug first sentence if requested
        debug_this = args.debug and len(predictions) == 0

        # Process the sentence
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
            # diac = remove_end_diacritics_from_token(diac)

            # Apply MADAMIRA-style post-processing rules to the Arabic text first
            # Process the Arabic diacritized text before mapping to ALA-LC
            
            # RULE 1: 'LIL' RULE - Exactly matching translit_rules.py implementation
            # When the preposition 'li' (to/for) combines with the definite article 'al',
            # they form 'lil' in Arabic (لِل) instead of 'li-al' (لِ+ال)
            if analysis.get('prc0') == 'Al_det' and analysis.get('prc1') == 'li_prep':
                find = re.escape('لِ+ال')
                diac = re.sub(find, r'لِل', diac)
            
            # RULE 2: CASE ENDINGS REMOVAL - Exactly matching translit_rules.py implementation
            # Arabic has grammatical case markers that are often omitted in romanization
            bw = analysis.get('bw', '')  # Get Buckwalter representation of the word
            bwsplit = bw.split('+') if bw else []  # Split by morpheme boundaries
            bwending = bwsplit[-1] if bwsplit else ''  # Get the last morpheme (usually contains case info)
            
            # Exception: If word ends with direct object or possessive pronoun, keep case endings
            # These are integral to the word's pronunciation and meaning
            if bw and ('DO' in bwending or 'POSS_PRON' in bwending): 
                # Keep case endings intact
                pass
            # If word ends with case marker or certain suffixes, remove the final diacritics
            elif bw and any(marker in bwending for marker in ['CASE', 'IV', 'PV', 'CV', 'NSUFF']):
                # Remove alif tanween (ـًا) first - this is the accusative indefinite marker
                diac = re.sub(r'اً$', '', diac)
                # Remove other diacritics (ـَ ـُ ـِ ـً ـٌ ـٍ) at the end of the word
                diac = re.sub(r'[ًٌٍَُِ]$', '', diac)
            
            # RULE 3: TA MARBUTA HANDLING - Exactly matching translit_rules.py implementation
            # In Arabic, taa marbuta (ة) is pronounced as 'h' in pausal form
            # but as 't' when in construct state (idafa/إضافة)
            if 'ة' in diac:
                # Check if word is in construct state
                if analysis.get('stt') == 'c':
                    # Exception: Cannot be construct if followed by preposition
                    # This handles cases where analysis might be incorrect
                    next_is_prep = False
                    if idx < len(disamb_words) - 1:
                        next_analysis = disamb_words[idx+1].analyses[0].analysis if disamb_words[idx+1].analyses else {}
                        next_bw = next_analysis.get('bw', '').split('+')[0]
                        next_is_prep = 'PREP' in next_bw
                    
                    if not next_is_prep:
                        # Add sukun (ْ) on ta-marbuta to force its transliteration as 't' instead of 'h'
                        diac = re.sub(r'ة', r'ةْ', diac)
            
            # RULE 4: SINGLE LETTER PROCLITICS - Exactly matching translit_rules.py implementation
            # Arabic has single-letter prepositions (ب/bi, ل/li)
            # These should be separated from the main word with hyphens in romanization
            bwbeginning = bwsplit[0] if bwsplit else ''  # Get the first morpheme
            if bw and 'PREP' in bwbeginning and len(bwsplit) > 1:  # Length condition ensures it's a real proclitic
                lemma = analysis.get('lemma', '').split('_')[0]
                # Only handle the common single-letter prepositions ب (bi) and ل (li)
                if lemma in {'لِ-', 'بِ'}:
                    # Add hyphen after the single letter: 'ب' → 'ب-', 'ل' → 'ل-'
                    # The [َُِ]? pattern matches optional diacritics on the letter
                    diac = re.sub(r'^([لب][َُِ]?)', r'\1-', diac)
            
            # RULE 5: CAPITALIZATION RULES - Exactly matching translit_rules.py implementation
            # Mark tokens for capitalization based on position and type
            capschar = '\u00b1'  # Special marker used to mark tokens for capitalization (± symbol)
            
            # Capitalize sentence-initial tokens
            if idx == 0 and not diac.endswith(capschar):
                diac = diac + capschar
            # Capitalization rules for non-initial tokens
            elif idx > 0:
                # Get the previous token for context
                prev_token = tokens[idx-1]
                
                # Capitalize after period, question mark, exclamation mark (sentence boundary)
                if prev_token in {'.', '?', '!', '؟', '!'}:
                    diac = diac + capschar
                
                # Capitalize proper nouns and adjectives with capitalized glosses
                # Skip Arabic punctuation
                elif analysis.get('pos') in {'noun_prop', 'adj'} and \
                     analysis.get('gloss') and analysis.get('gloss')[0].isupper() and \
                     tok not in {'،', '؛', '.', '?', '!'} and not diac.endswith(capschar):
                    diac = diac + capschar
                
                # Ensure all proper nouns are capitalized regardless of gloss
                elif analysis.get('pos') == 'noun_prop' and not diac.endswith(capschar):
                    diac = diac + capschar
            
            # # Now map to ALA-LC after all Arabic processing is done
            # # Check if the token has the capitalization marker
            if diac.endswith(capschar):
                # Remove marker before transliteration
                diac_without_marker = diac[:-1]
                # Transliterate first
                rom_tok = translit_simple(diac_without_marker, loc_map, loc_exceptional)
                # Then capitalize
                rom_tok = capitalize_loc(rom_tok)
            else:
                # Standard transliteration for non-capitalized tokens
                rom_tok = translit_simple(diac, loc_map, loc_exceptional)
            
            # Handle all other clitics attachment with hyphens (prefixes and suffixes)
            # This uses the PROCLITIC_MAP and ENCLITIC_MAP dictionaries to add proper hyphenation
            rom_tok = handle_clitics(analysis, rom_tok)
            
            # Special case: Ensure proper noun capitalization with articles
            # For proper nouns with the definite article (al-), capitalize the main part: 'al-qahirah' → 'al-Qahirah'
            if analysis.get('pos') == 'noun_prop' and re.search(r'^al-[a-z]', rom_tok):
                rom_tok = re.sub(r'^(al-)([a-z])', lambda m: m.group(1) + m.group(2).upper(), rom_tok)
                
            # # Special case: Ensure proper noun capitalization with articles
            # # For proper nouns with the definite article (al-), capitalize the main part: 'al-qahirah' → 'al-Qahirah'
            # if analysis.get('pos') == 'noun_prop' and re.search(r'^al-[a-z]', rom_tok):
            #     rom_tok = re.sub(r'^(al-)([a-z])', lambda m: m.group(1) + m.group(2).upper(), rom_tok)

            rom_tokens.append(rom_tok)
        # Apply all additional post-processing rules in the same order as translit_rules.py
        rom_tokens = post_process_romanization(rom_tokens, disamb_words)

        predictions.append(recompose(' '.join(rom_tokens), mode='rom'))

    # Write plain text: one line per sentence, no quotes
    with open(output_txt, 'w', encoding='utf-8') as o:
        for line in predictions:
            o.write(f"{line}\n")
    print(f"Wrote {len(predictions)} lines to {output_txt}")


if __name__ == "__main__":
    main()