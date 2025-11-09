#!/bin/bash

# Exit on error
set -e

# Function to strip notebook outputs and stage the changes
process_notebook() {
    if grep -q '"execution_count":' "$1" || grep -q '"outputs":' "$1"; then
        echo "Stripping outputs from notebook: $1"
        nbstripout "$1"
        git add "$1"  # Re-stage the cleaned notebook
    fi
}

# Process each file passed as argument
for file in "$@"; do
    if [[ $file == *.ipynb ]]; then
        process_notebook "$file"
    fi
done