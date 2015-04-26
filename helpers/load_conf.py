"""
Script used to load network and dataset config files
"""

import json
import logging

logger = logging.getLogger(__name__)


def load_config(path):
    """ Tries to load configuration file. If failes, returns None """

    try:
        with open(path, 'r') as f:
            j = json.load(f)
        logger.info("Successfully loaded cofiguration file %s", path)
    except IOError:
        logger.error("Cannot open configuration file %s", path)
        return None
    return j


def convert_to_function_params(d):
    """
    Creates copy of dictionary d and changes '-' to '_' in dicy keys
    """

    new_d = d.copy()
    keys_to_change = filter(lambda x: '-' in x, new_d.keys())
    for key in keys_to_change:
        new_key = key.replace('-', '_')
        new_d[new_key] = new_d.pop(key)

    return new_d
