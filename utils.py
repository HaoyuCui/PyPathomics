import yaml


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f.read())


def get_config():
    return read_yaml('config.yaml')
