"""Entry point for the application.

Run python main.py -h for help on all the commands available. This project is
available on GitHub at https://github.com/johnliu4/traffic-cv.
"""

import configparser
import argparse
import re

def load_config():
    """Loads the config file.

    A default config file is provided in the root directory of this project.
    Modify the file as you wish.

    Raises:
        IOError: A key-value pair in the config file is missing.
    """
    config = configparser.ConfigParser()
    config.read('config')

    if 'DEFAULT' in config:
        print('hello')


    with open('config', 'w') as configfile:
        config.write(configfile)

if __name__ == '__main__':
    load_config()

    parser = argparse.ArgumentParser(description='Welcome to TrafficCV by John Liu!')
    args = parser.parse_args()
    subparsers = parser.add_subparsers(help='sub-command help')

    # command: train, for training the SVM classifier
    parser_train = subparsers.add_parser('train', help='train help')
    #start_cmd_line();
