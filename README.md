
# Lighter Training Parameters
python test_train2.py --model resnet18 --data-dir ./combined_dataset --batch-size 64 --val-batch-size 128 --num-workers 12 --epochs 80 --lr 2e-4 --weight-decay 1e-5 --scheduler cosine --train-steps-per-epoch 1000 --val-steps-per-epoch 200 --verbose


# Heavier Training Parameter    
python test_train2.py --model resnet18 --data-dir ./combined_dataset --batch-size 128 --val-batch-size 256 --num-workers 16 --prefetch-factor 8 --epochs 80 --lr 4e-4 --weight-decay 1e-5 --scheduler cosine --train-steps-per-epoch 2000 --val-steps-per-epoch 400 --verbose
