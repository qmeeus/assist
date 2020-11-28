'''@file train_test.py
do training followed by testing
'''

import os
import sys
sys.path.append(os.getcwd())
import argparse
from assist.scripts import train, test
import multiprocessing as mp


def main(expdir):
    '''main function'''

    train.main(expdir)
    test.main(expdir)

if __name__ == "__main__":

    #create the arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('expdirs', nargs="+")
    args = parser.parse_args()

    if len(args.expdirs) > 1:
        with mp.Pool(len(args.expdirs)):
            pool.map(main, args.expdirs)
    else:
        main(args.expdirs[0])
