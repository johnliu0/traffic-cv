"""Module for application configuration loading."""

import configparser
from os.path import expanduser

training_positives_dir = None
training_negatives_dir = None
models_dir = None
input_image_target_height = 640
convnet_image_input_size = (32, 32)

def load_config():
    """Loads the config file.

    The config file is named 'config.ini' and exists at the root directory.
    """

    global training_positives_dir
    global training_negatives_dir
    global models_dir

    config = configparser.ConfigParser()
    config.read('.config')
    sect = config['DEFAULT']

    # read the variables
    training_positives_dir = expanduser(sect.get('training_positives_dir', './data/positives'))
    training_negatives_dir = expanduser(sect.get('training_negatives_dir', './data/negatives'))
    models_dir = expanduser(sect.get('models_dir', './data/models'))
    input_image_target_height = int(sect.get('input_image_target_height', 640))
    size = int(sect.get('convnet_input_target_size', 32))
    convnet_image_input_size = (size, size)
