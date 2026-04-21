#!/bin/bash

MODELS=(
  #"ministral-8b"
  "ministral-3b"
  #"foundation-sec-8b"
  #"qwen3-8b"
  #"qwen3-vl-8b"
  )

for MODEL in "${MODELS[@]}"; do
  CONFIG="configs/${MODEL}.yaml"
  
  echo "========================================="
  echo "Running pipeline for $MODEL"
  echo "Using config: $CONFIG"
  echo "========================================="

  ######## Load fine-tuned models and run TACTIC-KG pipeline ########
  #python src/load_ft_models/load_ft_extractor.py --config $CONFIG
  #python src/load_ft_models/load_ft_typer.py --config $CONFIG
  #python src/load_ft_models/load_ft_verifier.py --config $CONFIG
  #python src/load_ft_models/load_ft_curator.py --config $CONFIG
 

  ######## EVALUATION ########
  python src/utils/evaluate_semantic.py --config $CONFIG


  echo "Finished $MODEL"
  echo
done

echo "All models completed."
