#!/bin/bash
DATASET="ESC50-Actions"
METHOD="coop"


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




for FOLD in 1 2 3 4 5
# for FOLD in 4 5
    do
        for SEED in $SEEDS
            do
                echo "Running Fold-$FOLD with SEED=$SEED"
                # if [ -f "$DATASET_ROOT/train.csv" ]; then rm -rf "$DATASET_ROOT/train.csv"; fi
                # if [ -f "$DATASET_ROOT/test.csv" ]; then rm -rf "$DATASET_ROOT/test.csv"; fi
                # cp "$DATASET_ROOT/csv_files/train_$FOLD.csv" "$DATASET_ROOT/train.csv"
                # cp "$DATASET_ROOT/csv_files/test_$FOLD.csv" "$DATASET_ROOT/test.csv"

                python main.py \
                    --dataset_root $DATASET_ROOT \
                    --n_epochs 50 \
                    --freq_test_model 10 \
                    --ctx_dim $CTX_DIM \
                    --batch_size 16 \
                    --seed $SEED \
                    --exp_name "$DATASET-FOLD$FOLD" \
                    --do_logging \
                    --train_csv "csv_files/train_$FOLD.csv" \
                    --test_csv "csv_files/test_$FOLD.csv" \
                    "$@"
            done
    done
