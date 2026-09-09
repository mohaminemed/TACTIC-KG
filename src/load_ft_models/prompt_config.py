from pathlib import Path

import yaml


def load_prompt_config(config_path):
    ontology_path = Path(config_path).resolve().parent / "ontology.yaml"
    with ontology_path.open("r", encoding="utf-8") as ontology_file:
        return yaml.safe_load(ontology_file)


def render_prompt(template, **values):
    prompt = template
    for name, value in values.items():
        prompt = prompt.replace("${" + name + "}", str(value))
    return prompt