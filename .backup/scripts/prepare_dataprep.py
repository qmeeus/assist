'''@file prepare_dataprep.py
do all the preparations for the data preparation'''

import os
import sys
sys.path.insert(0, os.getcwd())
import shutil
import argparse
from pathlib import Path
from configparser import ConfigParser
from assist.scripts import dataprep
from assist.tools import tools, logger


def main(options):

    expdir = Path(options["expdir"])
    recipe = Path(options["recipe"])
    backend = options["backend"]
    cuda = options["cuda"]
    njobs = options["njobs"]
    overwrite = options["overwrite"]

    if expdir.exists() and overwrite:
        shutil.rmtree(expdir)

    os.makedirs(expdir, exist_ok=True)

    #read the database configuration
    if not os.path.exists(os.path.join(recipe, 'database.cfg')):
        raise Exception('cannot find database.cfg in %s' % recipe)

    dataconf = ConfigParser()
    dataconf.read(os.path.join(recipe, 'database.cfg'))

    shutil.copyfile(os.path.join(recipe, 'features.cfg'),
                    os.path.join(expdir, 'features.cfg'))

    for speaker in dataconf.sections():

        logger.info('%s data preparation' % speaker)

        #create the experiments directory
        if not os.path.isdir(os.path.join(expdir, speaker)):
            os.makedirs(os.path.join(expdir, speaker))

        #create a database config for this speaker
        speakerconf = ConfigParser()
        speakerconf.add_section('database')
        for option, value in dict(dataconf.items(speaker)).items():
            speakerconf.set('database', option, value)
        with open(os.path.join(expdir, speaker, 'database.cfg'), 'w') as fid:
            speakerconf.write(fid)

        #put a link to the feature conf
        tools.symlink(
            os.path.join(expdir, 'features.cfg'),
            os.path.join(expdir, speaker, 'features.cfg')
        )

        if backend in ('condor', 'condor_gpu'):
            #create the outputs directory
            if not os.path.isdir(os.path.join(expdir, speaker, 'outputs')):
                os.makedirs(os.path.join(expdir, speaker, 'outputs'))

            if backend == 'condor_gpu':
                jobfile = 'run_script_GPU.job'
            else:
                jobfile = 'run_script.job'

            #submit the condor job
            os.system(
                'condor_submit expdir=%s script=dataprep'
                ' assist/condor/%s' % (os.path.join(expdir, speaker), jobfile))
        else:
            dataprep.main(os.path.join(expdir, speaker))



if __name__ == '__main__':

    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', type=Path, help='the experiments directory')
    parser.add_argument('recipe', type=Path, help='the recipe directory')
    parser.add_argument('--backend', default="local", choices=["local", "mp", "condor"])
    parser.add_argument("--njobs", type=int, default=12)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    main(vars(parser.parse_args()))
