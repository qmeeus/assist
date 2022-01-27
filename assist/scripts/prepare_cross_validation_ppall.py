'''@file prepare_cross_validation_ppall.py
do all the preparations for cross validation'''

import os
import sys
sys.path.append(os.getcwd())
import shutil
import argparse
import cPickle as pickle
from ConfigParser import ConfigParser
import random
import itertools
import numpy as np
from assist.tasks.structure import Structure
from assist.tasks import coder_factory
from assist.tasks.read_task import read_task
from assist.experiment.make_blocks import make_blocks
import train_test
from assist.tools import tools

def main(expdir, recipe, computing):
    '''main function'''

    overwrite = False
    if os.path.isdir(expdir):
        text = ''
        while text not in ('o', 'r'):
            text = raw_input('%s already exists, do you want to '
                             'resume experiment (r) or overwrite (o) '
                             '(respond with o or r)' % expdir)
        if text == 'o':
            overwrite = True

    else:
        #create the experiments directory
        os.makedirs(expdir)

    #copy the config files
    if overwrite:
        shutil.copyfile(os.path.join(recipe, 'acquisition.cfg'),
                        os.path.join(expdir, 'acquisition.cfg'))
    else:
        tools.safecopy(os.path.join(recipe, 'acquisition.cfg'),
                       os.path.join(expdir, 'acquisition.cfg'))

    shutil.copyfile(os.path.join(recipe, 'coder.cfg'),
                    os.path.join(expdir, 'coder.cfg'))
    shutil.copyfile(os.path.join(recipe, 'structure.xml'),
                    os.path.join(expdir, 'structure.xml'))
    shutil.copyfile(os.path.join(recipe, 'database.cfg'),
                    os.path.join(expdir, 'database.cfg'))
    shutil.copyfile(os.path.join(recipe, 'cross_validation_ppall.cfg'),
                    os.path.join(expdir, 'cross_validation_ppall.cfg'))

    acquisitionconf = ConfigParser()
    acquisitionconf.read(os.path.join(recipe, 'acquisition.cfg'))
    modelname = acquisitionconf.get('acquisition', 'name')
    shutil.copyfile(os.path.join(os.getcwd(),'assist','acquisition','defaults',modelname+'.cfg'),os.path.join(expdir,modelname+'.cfg'))

    #read the cross_validation config file
    expconf = ConfigParser()
    expconf.read(os.path.join(recipe, 'cross_validation_ppall.cfg'))

    #default conf file
    default = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'defaults',
        'cross_validation_ppall.cfg')

    #apply the defaults
    if os.path.exists(default):
        tools.default_conf(expconf, default)

    expconf = dict(expconf.items('cross_validation_ppall'))

    #read the data config file
    if not os.path.exists(os.path.join(recipe, 'database.cfg')):
        raise Exception('cannot find database.cfg in %s' % recipe)

    dataconf = ConfigParser()
    dataconf.read(os.path.join(recipe, 'database.cfg'))

    #read the coder config file
    coderconf = ConfigParser()
    coderconf.read(os.path.join(expdir, 'coder.cfg'))

    # for word specific thresholds (not used anymore)
    #if os.path.isfile(os.path.join(recipe,'word_thresholds.pkl')):
    #    print('File with wordthresholds found in recipe')
    #    shutil.copyfile(os.path.join(recipe, 'word_thresholds.pkl'),
    #        os.path.join(expdir, 'word_thresholds.pkl'))
    #    thresholdsarepresent = True
    #else:
    #    print('No file found with wordthresholds, using a fixed one')
    #    thresholdsarepresent = False

    labelvecs = []
    names = []
    taskstrings = dict()
    features = dict()

    print 'Searching for all speakers...'

    for speaker in dataconf.sections():

        print '     speaker: %s' % (speaker)

        #create a task structure file
        structure = Structure(os.path.join(expdir, 'structure.xml'))
        #create a coder
        coder = coder_factory.factory(coderconf.get('coder', 'name'))(
            structure, coderconf)
        # typesplit_coder.py line 51 to see all labels and corresponding output capsule numbers

        #read and code all the tasks
        for line in open(dataconf.get(speaker, 'tasks')):  #'recording1_Voice_10 <move_rel direction="forward" distance="little" throttle="fast" />'
            splitline = line.strip().split(' ')
            name = speaker + '_' + splitline[0]   #'recording1_Voice_10'
            names.append(name)
            taskstring = ' '.join(splitline[1:])  #'<move_rel_direction="forwqrd" distance="little" throttle="fast"/>'
            taskstrings[name] = taskstring
            task = read_task(taskstring)
            labelvecs.append(coder.encode(task))

        # read the feature files
        for l in open(os.path.join(dataconf.get(speaker, 'features'), 'feats')):
            splitline = l.strip().split(' ')  #['recording1_Voice_10', '/esat/spchtemp/scratch/r0580562/databases/grabo_features/pp2/recording1_Voice_10.npy']
            featname = speaker + '_' + splitline[0]
            features[featname] = ' '.join(splitline[1:])

    print 'Devide data into blocks...'
    #devide the data into blocks, look for existing blocksfile in recipe because takes a very long time to make!!!
    blocksfile = os.path.join(recipe, 'blocks.pkl')
    if os.path.exists(blocksfile):
        print 'Loading found blocks file (check if number of blocks is still the same)'
        with open(blocksfile, 'rb') as fid:
            blocks = pickle.load(fid)
    else:
        print 'No blocksfile found in  recipe, making new one'
        blocks = make_blocks(np.array(labelvecs), expconf, expdir)  #massive list, matrix of [[..,..,..,..],[..,...    .]]  numbers between 1-350 approx
        with open(blocksfile, 'wb') as fid:
            pickle.dump(blocks, fid)


    print 'Shuffle speakers...'

    # look for existing train and test sets and load them in ('saved_ids' in recipe), because takes a very long time to make!!!

    sets_properties = {}
    if os.path.isdir(os.path.join(recipe,'saved_ids')):
        saved_ids = ConfigParser()
        saved_ids.read(os.path.join(recipe, 'saved_ids', 'cross_validation_ppall.cfg'))
        sets_properties = dict(saved_ids.items('cross_validation_ppall'))
    else:
        sets_properties['numblocks'] = 0
        sets_properties['numexp'] = 0

    if (sets_properties['numblocks'] == expconf['numblocks']) and (sets_properties['numexp'] == expconf['numexp']):
        print '     Loading found test recipe'
        trainids_saved = os.path.join(recipe, 'saved_ids','trainids.pkl')
        with open(trainids_saved, 'rb') as fid:
            trainids = pickle.load(fid)
        testids_saved = os.path.join(recipe, 'saved_ids','testids.pkl')
        with open(testids_saved, 'rb') as fid:
            testids = pickle.load(fid)
    else:
        print '     No saved test and train sets found with same crossvalidation configuration in the recipe'
        # seed the random number generator
        random.seed(3105)
        trainids = [None]*(len(blocks)-1)  #len(blocks)=15
        testids = [None]*(len(blocks)-1)
        print '     Number of blocks: %d' % (len(blocks))
        b = 0
        while b < (len(blocks)-1):
        #for b in range(len(blocks)-1):
            print '         block %d' %b
            trainids[b] = [None]*int(expconf['numexp'])
            testids[b] = [None]*int(expconf['numexp'])
            for e in range(int(expconf['numexp'])):
                trainids[b][e] = list(
                    itertools.chain.from_iterable(random.sample(blocks, b+1)))
                testids[b][e] = [x for x in range(len(names))
                                 if x not in trainids[b][e]]
            #scale factor to use more smaller blocks and less bigger blocks (for the curve, it saturates)
            newb = int(np.floor((b + 1) * float(expconf['scale']) + int(expconf['increment']) - 1))
            newb = min(newb, len(blocks) - 2)
            if b == newb:
                break
            else:
                b = newb

        os.makedirs(os.path.join(expdir, 'saved_ids'))
        trainids_saved = os.path.join(expdir, 'saved_ids', 'trainids.pkl')
        testids_saved = os.path.join(expdir, 'saved_ids', 'testids.pkl')
        with open(trainids_saved, 'wb') as fid:
            pickle.dump(trainids, fid)
        with open(testids_saved, 'wb') as fid:
            pickle.dump(testids, fid)
        shutil.copyfile(os.path.join(recipe, 'cross_validation_ppall.cfg'),
                        os.path.join(expdir,'saved_ids', 'cross_validation_ppall.cfg'))


    #create an expdir for each experiment
    b = int(expconf['startblocks']) - 1  #0

    print 'Launch the experiments...'
    while True:
        for e in range(int(expconf['numexp'])):

            print '     train blocks: %d, experiment %s' % (b+1, e)

            #creat the directory
            subexpdir = os.path.join(expdir, '%dblocks_exp%d' % (b+1, e))

            if os.path.exists(os.path.join(subexpdir, 'f1')):
                continue

            if not os.path.isdir(subexpdir):
                os.makedirs(subexpdir)

            #create pointers to the config files
            tools.symlink(os.path.join(expdir, 'acquisition.cfg'),
                          os.path.join(subexpdir, 'acquisition.cfg'))
            tools.symlink(os.path.join(expdir, 'coder.cfg'),
                          os.path.join(subexpdir, 'coder.cfg'))
            tools.symlink(os.path.join(expdir, 'structure.xml'),
                          os.path.join(subexpdir, 'structure.xml'))
            tools.symlink(os.path.join(expdir, 'database.cfg'),
                          os.path.join(subexpdir, 'database.cfg'))
            #if thresholdsarepresent:
            #    tools.symlink(os.path.join(expdir, 'word_thresholds.pkl'),
            #                  os.path.join(subexpdir, 'word_thresholds.pkl'))

            if not os.path.exists(os.path.join(subexpdir, 'trainfeats')):
                trainutts = [names[i] for i in trainids[b][e]]
                print 'number of examples: %d' % len(trainutts)
                testutts = [names[i] for i in testids[b][e]]

                #create the train and test sets
                tools.writefile(
                    os.path.join(subexpdir, 'trainfeats'),
                    {utt: features[utt] for utt in trainutts})
                tools.writefile(
                    os.path.join(subexpdir, 'traintasks'),
                    {utt: taskstrings[utt] for utt in trainutts})
                tools.writefile(
                    os.path.join(subexpdir, 'testfeats'),
                    {utt: features[utt] for utt in testutts})
                tools.writefile(
                    os.path.join(subexpdir, 'testtasks'),
                    {utt: taskstrings[utt] for utt in testutts})

            if computing in ('condor', 'condor_gpu'):
                #create the outputs directory
                if not os.path.isdir(os.path.join(subexpdir, 'outputs')):
                    os.makedirs(os.path.join(subexpdir, 'outputs'))

                if computing == 'condor_gpu':
                    jobfile = 'run_script_GPU.job'
                else:
                    jobfile = 'run_script.job'

                #only submit the job if it not running yet
                in_queue = os.popen(
                    'if condor_q -nobatch -wide | grep -q %s; '
                    'then echo true; else echo false; fi' %
                    subexpdir).read().strip() == 'true'

                #submit the condor job
                if not in_queue:
                    os.system('condor_submit expdir=%s script=train_test'
                              ' assist/condor/%s'
                              % (subexpdir, jobfile))
            else:
                train_test.main(subexpdir)

        newb = int(np.floor((b + 1)*float(expconf['scale']) + int(expconf['increment']) - 1))
        newb = min(newb, len(blocks) - 2)
        if b == newb:
            break
        else:
            b = newb

if __name__ == '__main__':

    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', help='the experiments directory')
    parser.add_argument('recipe', help='the recipe directory')
    parser.add_argument('--computing', '-c',
                        help='the kind of computing you want to do')
    args = parser.parse_args()

    if args.computing and args.computing not in \
            ('condor', 'condor_gpu', 'local'):
        raise Exception('unknown computing mode %s' % args.computing)

    main(args.expdir, args.recipe, args.computing)
