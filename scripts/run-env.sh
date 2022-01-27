#!/bin/bash -e

CONDA_HOME=/users/spraak/qmeeus/spchtemp/bin/anaconda3
CONDA_INIT=$CONDA_HOME/etc/profile.d/conda.sh
CONDA_ENV=espnet-stable
source $CONDA_INIT
conda activate $CONDA_ENV

# python -m assist $@
python $@
