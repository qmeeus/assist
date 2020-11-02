import os
import sys
sys.path.append(os.getcwd())
import argparse
from assist.tasks.read_task import read_task
from ConfigParser import ConfigParser
import numpy as np
from assist.tasks.structure import Structure
from assist.tasks import coder_factory
from functools import reduce

#parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='the experiments directory')
args = parser.parse_args()

expdir = args.expdir
#read the coder config file
coderconf = ConfigParser()
coderconf.read(os.path.join(expdir, 'coder.cfg'))

#create a task structure file
structure = Structure(os.path.join(expdir, 'structure.xml'))

#create a coder
coder = coder_factory.factory(coderconf.get('coder', 'name'))(
    structure, coderconf)

#ref=dict()
with open(os.path.join(expdir, 'testtasks')) as f:
    ref = dict([l.split(' ',1) for l in f])
    # lines = f.readlines()
    # for line in lines:
    #     splitline = line.strip().split(' ')
    #     taskstrings[splitline[0]] = ' '.join(splitline[1:])
with open(os.path.join(args.expdir, 'dectasks')) as f:
    hyp = dict([l.split(' ',1) for l in f])

# reftasks = [read_task(task) for task in ref.values()]
# hyptasks = [read_task(task) for task in hyp.values()]
tasks=[(read_task(ref[k]),read_task(hyp[k])) for k in hyp.keys()]

vs_ref = np.array([coder.encode(t[0],None,0.0) for t in tasks])
vs_hyp = np.array([coder.encode(t[1],None,0.0) for t in tasks])

eq = vs_ref == vs_hyp
eqtasks = eq.all(axis=1)
corr = np.sum(eqtasks)
# tasks=[(read_task(ref[k]),read_task(hyp[k])) for k in hyp.keys()]
# print (len(tasks))
# sum = reduce((lambda x,y: int(x==y)),tasks)
pct = 100.0*float(corr)/float(len(eqtasks))
print( '%10.3g\n' % pct )