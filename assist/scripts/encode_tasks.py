'''@file encode_tasks.py
write binary encoded tasks to numpy file
'''

import os
import sys
sys.path.append(os.getcwd())
import argparse
from ConfigParser import ConfigParser
import numpy as np
from assist.tasks.structure import Structure
from assist.tasks import coder_factory
from assist.tools import tools
from assist.tasks.read_task import read_task

def main(expdir):
    '''main function'''

    #read the coder config file
    coderconf = ConfigParser()
    coderconf.read(os.path.join(expdir, 'coder.cfg'))

    #create a task structure file
    structure = Structure(os.path.join(expdir, 'structure.xml'))

    #create a coder
    coder = coder_factory.factory(coderconf.get('coder', 'name'))(
        structure, coderconf)

    #read the traintasks
    with open(os.path.join(expdir, 'traintasks')) as f:
        lines = f.readlines()
        for line in lines:
            splitline = line.strip().split(' ')
            taskstring = ' '.join(splitline[1:])
            print(splitline[0] + " : " + taskstring )
            task = read_task(taskstring)
            vs = coder.encode(task).astype(int)
            print "%d : [" % len(vs),
            sys.stdout.write(" ".join(str(x) for x in vs))
            print "]\n"





if __name__ == "__main__":

    #create the arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir')
    args = parser.parse_args()

    main(args.expdir)
