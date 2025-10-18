import argparse
import json
import yaml

from datasets import load_dataset


def load_config():
    parser = argparse.ArgumentParser(description='Load configuration from a YAML file.')
    parser.add_argument('config_file', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def collect_top10_items(generated):

    save_path = "/root/samlrs/data/ml-1m-processed/prior.json"
    top10_items = []
    for items in generated:
        top10 = [str(x).strip() for x in items[:10] if str(x).strip()]
        top10_items.append(top10)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(top10_items, f, ensure_ascii=False, indent=2)

    return top10_items
