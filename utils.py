from omegaconf import OmegaConf

def load_config_yaml(path: str):
    """
    Loads a YAML configuration file using OmegaConf and returns the config.

    Args:
        path (str): Path to the config.yaml file

    Returns:
        OmegaConf.DictConfig: Configuration object loaded from the YAML file
    """
    config = OmegaConf.load(path)
    return config

