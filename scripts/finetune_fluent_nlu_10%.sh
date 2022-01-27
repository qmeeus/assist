#!/bin/bash

set -eo pipefail

source /users/spraak/qmeeus/spchtemp/bin/anaconda3/etc/profile.d/conda.sh
conda activate espnet-stable

encoder=bert-base-uncased
model_config=config/cfgs/lstm_fluent.json
configs="config/cfgs/train_fluent_2opts_1%.json config/cfgs/early_stopping.json"
outdir=exp/debug/fluent/train_nlu_asr_10%
dataset=/esat/spchtemp/scratch/qmeeus/repos/datasets/config/fluent.json
input_key=asr
output_key=tasks

python finetune_slu.py \
  --encoder $encoder \
  --outdir $outdir \
  --model-config $model_config \
  --nlu \
  --freeze-modules embeddings encoder pooler \
  --dataset $dataset \
  --dataset-size .1 \
  --input-key $input_key \
  --output-key $output_key \
  --config $configs \
  --device cuda
