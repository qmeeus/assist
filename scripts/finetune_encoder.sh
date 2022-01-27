#!/bin/bash

set -eo pipefail

{ [ "$1" ] && [ "$2" ] ;} || { echo "Usage: finetune_encoder.sh <OUTDIR> <IKEY> <OKEY>" && exit 1 ;}

source /users/spraak/qmeeus/spchtemp/bin/anaconda3/etc/profile.d/conda.sh
conda activate espnet-stable

set -u

outdir=$1
ikey=$2
kfold=${3:-0}
configs=config/cfgs/train_chatbot_2opts.json
okey=labels
model_config=config/cfgs/lstm_vaccinbot.json
dataset=/esat/spchtemp/scratch/qmeeus/repos/datasets/config/chatbot.json

options="--outdir $outdir --model-config $model_config --dataset $dataset --input-key $ikey --output-key $okey --config $configs"
if [ $ikey == "asr" ] || [ $ikey == "text" ]; then
  encoder=wietsedv/bert-base-dutch-cased
  options="$options --nlu"
    freeze_modules="encoder pooler"
else
  encoder=/esat/spchtemp/scratch/qmeeus/repos/espnet-stable/egs/cgn/asr1/exp/CGN_train_pytorch_train_pytorch_conformer_maskctc_specaug/results/model.val30.avg.best.pt
  freeze_modules="encoder decoder.decoders"
fi

[ "$freeze_modules" ] && options="$options --freeze-modules $freeze_modules"
[ "$kfold" -gt 0 ] && options="$options --kfold $kfold --split-method utterances"
options="$options --encoder $encoder --device cuda"

mkdir -p $outdir
echo "$(which python) finetune_slu.py $options" > $outdir/cmd.sh
python finetune_slu.py $options
python predict.py $outdir $dataset $ikey $okey $encoder

# outdir=exp/debug/vaccinbot/finetune_lstm_mlm
# input_key=fbank
# output_key=labels

# python finetune_slu.py \
#   --encoder $encoder \
#   --outdir $outdir \
#   --model-config $model_config \
#   --dataset $dataset \
#   --input-key $input_key \
#   --output-key $output_key \
#   --config $config \
#   --kfold 10 \
#   --split-method utterances \
#   --device cuda
