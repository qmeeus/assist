#!/bin/bash

set -eo pipefail

source /users/spraak/qmeeus/spchtemp/bin/anaconda3/etc/profile.d/conda.sh
conda activate espnet-stable

encoder=/esat/spchtemp/scratch/qmeeus/repos/espnet-stable/egs/cgn/asr1/exp/CGN_train_pytorch_train_pytorch_conformer_maskctc_specaug/results/model.val30.avg.best.pt
model_config=config/cfgs/lstm_vaccinbot_bn.json
config=config/cfgs/train_2opts.json
outdir=exp/debug/grabo/finetune_lstm_mlm
dataset=/esat/spchtemp/scratch/qmeeus/repos/datasets/config/grabo.json
input_key=fbank
output_key=tasks

python finetune_slu.py \
  --encoder $encoder \
  --outdir $outdir \
  --model-config $model_config \
  --dataset $dataset \
  --input-key $input_key \
  --output-key $output_key \
  --config $config \
  --kfold 10 \
  --split-method utterances \
  --device cuda
