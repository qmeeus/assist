#!/bin/bash

set -eo pipefail

source /users/spraak/qmeeus/spchtemp/bin/anaconda3/etc/profile.d/conda.sh
conda activate espnet-stable

set -u

encoder=bert-base-uncased
model_config=config/cfgs/gru_fluent_1%.json
configs="config/cfgs/train_fluent_2opts_1%.json"
outdir=exp/debug/fluent/train_nlu_asr_1%
dataset=/esat/spchtemp/scratch/qmeeus/repos/datasets/config/fluent.json
input_key=asr
output_key=tasks
freeze_modules="embeddings pooler encoder"

python finetune_slu.py \
  --encoder $encoder \
  --outdir $outdir \
  --model-config $model_config \
  --nlu \
  --freeze-modules $freeze_modules \
  --dataset $dataset \
  --dataset-size .01 \
  --input-key $input_key \
  --output-key $output_key \
  --config $configs \
  --device cuda

python predict.py $outdir $dataset $input_key $output_key $encoder
