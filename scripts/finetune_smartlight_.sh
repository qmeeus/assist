#!/bin/bash

set -eo pipefail

source /users/spraak/qmeeus/spchtemp/bin/anaconda3/etc/profile.d/conda.sh
conda activate espnet-stable

set -u

# encoder=/esat/spchtemp/scratch/qmeeus/repos/espnet-stable/egs/librispeech/asr1/exp/train_960_pytorch_train_pytorch_conformer_maskctc_specaug/results/model.val30.avg.best.pt
encoder=/esat/spchtemp/scratch/qmeeus/repos/espnet-stable/egs/librispeech/asr1/exp/train_960_pytorch_sti_transformer_lr10.0_ag8_p.5_mlm_wwm_specaug/results/model.val5.avg.best.pt
model_config=config/cfgs/gru_smartlights.json
# split_dir="exp/debug/smart-light-close-en/train_lstm_mlm_20210701_1805"
# classifier="$split_dir/split0/model.pt"
configs="config/cfgs/train_smartlights_2opts_.json"
outdir=exp/debug_/smartlights/finetune_mlm
dataset=/esat/spchtemp/scratch/qmeeus/repos/datasets/config/smartlights.json
input_key=fbank
output_key=tasks
freeze_modules="encoder ctc"

python finetune_slu.py \
  --encoder $encoder \
  --outdir $outdir \
  --model-config $model_config \
  --freeze-modules $freeze_modules \
  --dataset $dataset \
  --input-key $input_key \
  --output-key $output_key \
  --config $configs \
  --kfold 10 \
  --device cuda

python predict.py $outdir $dataset $input_key $output_key #$encoder
