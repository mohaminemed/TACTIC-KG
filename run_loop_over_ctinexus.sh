#!/bin/bash

MODELS=(
  "ctinexus"
)

for MODEL in "${MODELS[@]}"; do
  CONFIG="configs/${MODEL}.yaml"
  
  echo "========================================="
  echo "Running eval for $MODEL"
  echo "Using config: $CONFIG"
  echo "========================================="


  ######## EVAL ########
  
  python src/utils/evaluate_ctinexus.py --config $CONFIG

  echo "Finished $MODEL"
  echo
done

echo "All models completed."
