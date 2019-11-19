"""Entry point for the application.

Run python main.py -h for help on all the commands available. This project is
available on GitHub at https://github.com/johnliu4/traffic-cv.
"""

import config
import argparse
import ai
from os import path

if __name__ == '__main__':
    config.load_config()

    parser = argparse.ArgumentParser(description='Welcome to TrafficCV by John Liu!')
    subparsers = parser.add_subparsers(dest='subcmd', help='commands available')

    # command: train; for training the SVM classifier
    parser_train = subparsers.add_parser('train', help='trains the system to detect traffic lights in an image given training samples')

    # command: predict; for predicting traffic lights given an input image
    parser_predict = subparsers.add_parser('predict', help='attempts to find all traffic lights in an image')
    parser_predict.add_argument('--path', help='path to image', required=True)

    # command: mine; for mining positive and negative examples from a given input image
    parser_mine = subparsers.add_parser('mine', help='tool for hard mining positive and negative training data')
    parser_mine.add_argument('--path', help='path to image', required=True)
    parser_mine.add_argument('--use-predicted', help='whether or not to use the boxes that were predicted as positive samples', action='store_true')

    args = parser.parse_args()

    if args.subcmd == 'train':
        ai.train()
    elif args.subcmd == 'predict':
        ai.predict(path.expanduser(args.path))
    elif args.subcmd == 'mine':
        ai.mine(path.expanduser(args.path), args.use_predicted)
