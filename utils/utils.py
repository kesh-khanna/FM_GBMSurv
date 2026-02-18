import yaml
from typing import Dict, Any

def check_censoring(data, split_name):
    """
    check the censoring rate of the dataset
    """
    if data:
        num_events = sum([d['event'] for d in data])
        num_censored = len(data) - num_events
        censoring_rate = num_censored / len(data)
        print(f"{split_name} - Total: {len(data)}, Events: {num_events}, Censored: {num_censored}, Censoring Rate: {censoring_rate:.2f}")
    else:
        print(f"{split_name} data is empty or not provided.")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    load out config from the yaml, should be structured
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
