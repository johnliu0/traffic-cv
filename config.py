"""Module for application configuration loading."""

import configparser
from os.path import expanduser

training_imgs_dir = None
models_dir = None

def load_config():
    """Loads the config file.

    The config file is named 'config.ini' and exists at the root directory.
    """

    global training_imgs_dir
    global models_dir

    config = configparser.ConfigParser()
    config.read('.config')
    sect = config['DEFAULT']

    training_imgs_dir = expanduser(sect.get('trainingimgsdir', '/data/traffic_lights'))
    models_dir = expanduser(sect.get('modelsdir', '/data/models'))
