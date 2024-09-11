#!/usr/bin/bash

python scripts/finetune_slu.py \
    --outdir exp/debug/vaccinbot/test_20220217 \
    --model-config config/cfgs/lstm_vaccinbot_bn.json \
    --dataset /esat/spchtemp/scratch/qmeeus/repos/datasets/config/vaccinchat.json \
    --input-key asrtransformer \
    --output-key labels \
    --config config/cfgs/train_chatbot_2opts.json \
    --encoder wietsedv/bert-base-dutch-cased \
    --nlu \
    --freeze-modules encoder pooler \
    --kfold 10 \
    --split-method utterances \
    --device cuda
