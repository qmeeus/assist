#!/bin/bash

set -eo pipefail

USAGE="Usage: ./finetune.sh <OUTDIR> [<DATASET_SIZE>] [--freeze]"
if [ "$1" == "-h" ] || [ "$1" == "--help" ] || ! [ "$1" ]; then
  echo $USAGE && exit 0
fi
source /users/spraak/qmeeus/spchtemp/bin/anaconda3/etc/profile.d/conda.sh
conda activate espnet-stable

set -u

outdir=$1
dataset_size="${2:-1.}"
encoder=/esat/spchtemp/scratch/qmeeus/repos/espnet-stable/egs/librispeech/asr1/exp/train_960_pytorch_sti_transformer_lr10.0_ag8_p.5_mlm_wwm_specaug/results/model.val5.avg.best.pt
model_config=config/cfgs/lstm_fluent.json
configs="config/cfgs/train_fluent_2opts.json"
dataset=/esat/spchtemp/scratch/qmeeus/repos/datasets/config/fluent.json
input_key=fbank
output_key=tasks
freeze_modules="ctc encoder"

python finetune_slu.py \
  --encoder $encoder \
  --outdir $outdir \
  --model-config $model_config \
  --freeze-modules $freeze_modules \
  --dataset $dataset \
  --dataset-size $dataset_size \
  --input-key $input_key \
  --output-key $output_key \
  --config $configs \
  --device cuda

python predict.py $outdir $dataset $input_key $output_key
