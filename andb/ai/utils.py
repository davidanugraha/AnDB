import os
import logging

from andb.errno.errors import ConfigError

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1024

class ModelConfig(dict):
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            raise ConfigError(f'Not set configuration {key}')