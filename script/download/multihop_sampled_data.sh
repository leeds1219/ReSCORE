# This script is based on code from the IRCOT project by Stony Brook NLP.
# Source: https://github.com/StonyBrookNLP/ircot

#!/bin/bash

set -e
set -x

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

mkdir -p .temp/

# URL: https://drive.google.com/file/d/1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC/view?usp=sharing
gdown "1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC&confirm=t" -O .temp/sampled_data.zip

# Create the destination directory if it doesn't exist
mkdir -p ./data

# Unzip the file into the desired location, excluding any .DS_Store files
unzip -o .temp/sampled_data.zip -d ./data -x "*.DS_Store"

# Remove the temporary directory
rm -rf .temp/

# The resulting processed_data/ directory should look like:
# ├── 2wikimultihopqa
# │   ├── annotated_only_train.jsonl
# │   ├── dev.jsonl
# │   ├── dev_subsampled.jsonl
# │   ├── test_subsampled.jsonl
# │   └── train.jsonl
# ├── hotpotqa
# │   ├── annotated_only_train.jsonl
# │   ├── dev.jsonl
# │   ├── dev_subsampled.jsonl
# │   ├── test_subsampled.jsonl
# │   └── train.jsonl
# ├── iirc
# │   ├── annotated_only_train.jsonl
# │   ├── dev.jsonl
# │   ├── dev_subsampled.jsonl
# │   ├── test_subsampled.jsonl
# │   └── train.jsonl
# └── musique
#     ├── annotated_only_train.jsonl
#     ├── dev.jsonl
#     ├── dev_subsampled.jsonl
#     ├── test_subsampled.jsonl
#     └── train.jsonl
