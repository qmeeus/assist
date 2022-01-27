#!/bin/bash

USAGE="Usage: ./finetune.sh <OUTDIR> [<DATASET_SIZE>] [--freeze]"
if [ "$1" == "-h" ] || [ "$1" == "--help" ] || ! [ "$1" ]; then
  echo $USAGE && exit 0
fi
source /users/spraak/qmeeus/spchtemp/bin/anaconda3/etc/profile.d/conda.sh
conda activate espnet-stable

set -eo pipefail

outdir=$1
encoder=bert-base-uncased
model_config=config/cfgs/lstm_fluent.json
configs="config/cfgs/train_fluent_2opts_1%.json config/cfgs/early_stopping.json config/cfgs/sched_RLROP.json"
dataset=/esat/spchtemp/scratch/qmeeus/repos/datasets/config/fluent.json
dataset_size="${2:-1.}"
input_key=asr
output_key=tasks
freeze="embeddings pooler"
if [ "$3" == "--freeze" ]; then freeze="$freeze encoder"; fi

python finetune_slu.py \
  --encoder $encoder \
  --outdir $outdir \
  --model-config $model_config \
  --nlu \
  --freeze-modules $freeze \
  --dataset $dataset \
  --dataset-size $dataset_size \
  --input-key $input_key \
  --output-key $output_key \
  --config $configs \
  --device cuda

python predict.py $outdir $dataset $input_key $output_key $encoder
