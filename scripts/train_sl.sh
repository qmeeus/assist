#!/bin/bash -e

python train_lstm_fluent.py \
  exp/debug/smart-light-close-en/train_lstm_mlm \
  --expdir config/smart-light-close-en/lstm_128_mlm_wwm \
  --model-config config/cfgs/gru_smartlight.json \
  --config config/cfgs/train_smartlight.json \
  --method 10-fold

python prepare_for_evaluation.py exp/debug/smart-light-close-en/train_lstm_mlm

for expdir in exp/debug/smart-light-close-en/train_lstm_mlm/split*; do
  ln -sf model.pt $expdir/model
  python -m assist -vv evaluate $expdir config/smart-light-close-en/lstm_128_mlm_wwm
done

python collect_results.py exp/debug/smart-light-close-en/train_lstm_mlm
