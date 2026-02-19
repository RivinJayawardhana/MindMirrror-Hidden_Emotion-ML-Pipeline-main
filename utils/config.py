import os
import yaml
import logging
from typing import Dict, Any, List
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),'config.yaml')

def load_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f'Error loading configuration: {e}')
        return {}

def get_data_paths():
    config = load_config()
    return config.get('data_paths', {})

def get_emotion_categories():
    config = load_config()
    return config.get('attributes', {}).get('emotion_categories', [])

def get_required_attributes():
    config = load_config()
    return config.get('attributes', {}).get('required_attributes', [])