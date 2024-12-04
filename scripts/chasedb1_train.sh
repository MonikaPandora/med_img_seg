#! /bin/sh
cd ..
python train.py \
--dataset 'chase' \
--data_root '/root/autodl-tmp/datas/CHASEDB1' \
--resize 960 960 \
--num-class 1 \
--batch-size 4 \
--epochs 130 \
--lr 0.0038 \
--lr-update 'poly' \
--use_stn \
--folds 5


