#!/bin/bash

MODEL= "ministral-8b"


CONFIG="configs/${MODEL}.yaml"
  
echo "========================================="
echo "Running pipeline for $MODEL"
echo "Using config: $CONFIG"
echo "========================================="

######## Load fine-tuned models and run TACTIC-KG pipeline ########
python src/load_ft_models/load_ft_extractor.py --config $CONFIG
python src/load_ft_models/load_ft_typer.py --config $CONFIG
python src/load_ft_models/load_ft_verifier.py --config $CONFIG
python src/load_ft_models/load_ft_curator.py --config $CONFIG
 
######## EVALUATION ########
python src/utils/evaluate_semantic.py --config $CONFIG


echo "Finished $MODEL"
echo
done


