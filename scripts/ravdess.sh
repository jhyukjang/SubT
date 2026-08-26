#!/bin/bash
DATASET="RAVDESS"
METHOD="coop"



# if [ "$METHOD" != "zeroshot" ] && [ "$METHOD" != "coop" ] && [ "$METHOD" != "cocoop" ] && [ "$METHOD" != "palm" ]; then
#     echo "Invalid METHOD=$METHOD . Please choose one of the following: ['zeroshot', 'coop', 'cocoop', 'palm']"
#     exit 1
# fi

if [ "$is_half_labels" = "1" ]; then
    is_half_labels="--is_half_labels"
else
    is_half_labels=""
fi

echo "Running METHOD=$METHOD on DATASET=$DATASET"

DATASET_ROOT="Audio-Datasets/$DATASET"

if [ -d "$DATASET_ROOT" ]; then
    echo "Dataset path exists: $DATASET_ROOT"
else
    echo "Dataset path does not exist. Please set the correct path to the dataset root directory in variable DATASET_ROOT"
fi


if [[ "$METHOD" = *"coop"* ]] || [[ "$METHOD" == *"cocoop"* ]] ; then
    CTX_DIM=512
else
    CTX_DIM=1024
fi


if [ "$METHOD" = "zeroshot" ]; then
    SEEDS=0
else
    SEEDS="0 1 2"
fi



for SEED in $SEEDS
    do
        python main.py \
            --model_name $METHOD \
            --dataset_root $DATASET_ROOT \
            --n_epochs 50 \
            --freq_test_model 10 \
            --ctx_dim $CTX_DIM \
            --batch_size 16 \
            --seed $SEED \
            --exp_name "$DATASET" \
            --do_logging \
            --train_csv "train.csv" \
            --test_csv "test.csv" \
            "$@"
    done
