#!/bin/bash
python3 src/predict/camel_morph_backoff.py data/processed/dev.tsv predictions_out/camelmorph/dev/camel_morph.out --bert --custom-analyzer
python3 src/loc_transcribe.py evaluate predictions_out/camelmorph/dev/camel_morph.out data/processed/dev.tsv > evaluation_results.txt
python3 color_compare.py predictions_out/camelmorph/dev/camel_morph.out data/processed/dev.tsv --output comparison_output.html --html