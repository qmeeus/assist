#!/bin/bash

set -eo pipefail

source /users/spraak/qmeeus/spchtemp/bin/anaconda3/etc/profile.d/conda.sh
conda activate espnet-stable

set -u

encoder=bert-base-uncased
model_config=config/cfgs/gru_smartlights.json
configs="config/cfgs/train_smartlights_2opts.json"
outdir=exp/debug/smartlights/finetune_BERT+nlu_asr
dataset=/esat/spchtemp/scratch/qmeeus/repos/datasets/config/smartlights.json
input_key=asr
output_key=tasks
freeze_modules="encoder pooler"

python finetune_slu.py \
  --encoder $encoder \
  --outdir $outdir \
  --model-config $model_config \
  --nlu \
  --freeze-modules $freeze_modules \
  --dataset $dataset \
  --input-key $input_key \
  --output-key $output_key \
  --config $configs \
  --kfold 10 \
  --split-method utterances \
  --device cuda

python predict.py $outdir $dataset $input_key $output_key $encoder
