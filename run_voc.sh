#!/bin/bash

export DETECTRON2_DATASETS="./dataset"
EXP_TAG=`[ -z "$1" ] && echo "$(date +%s)" || echo "$1"`
CKPT_DIR="checkpoints/voc/${EXP_TAG}"
NGPU=4

. ./hashmap.sh

MAX_ITER=$(create_hashmap)
STEPS=$(create_hashmap)

trap "exit" INT
trap "cleanup $MAX_ITER && cleanup $STEPS" exit

add_item "$MAX_ITER" 1 1000
add_item "$MAX_ITER" 2 1500
add_item "$MAX_ITER" 3 2000
add_item "$MAX_ITER" 5 2500
add_item "$MAX_ITER" 10 4000

add_item "$STEPS" 1 "(800, )"
add_item "$STEPS" 2 "(1200, )"
add_item "$STEPS" 3 "(1600, )"
add_item "$STEPS" 5 "(2000, )"
add_item "$STEPS" 10 "(3200, )"

for split in `seq 1 3`
do
    python3 main.py --num-gpus $NGPU --config-file configs/voc/base.yaml \
        DATASETS.TRAIN "('voc_2007+2012_trainval_base${split}', )" \
        DATASETS.TEST "('voc_2007_test_all${split}', )" \
        MODEL.WEIGHTS "./pretrain/R-101.pkl" \
        OUTPUT_DIR "${CKPT_DIR}/base${split}" \
        SEED 0
    python3 -m tools.ckpt_surgery -d voc -m init -s 0 \
        "${CKPT_DIR}/base${split}/model_final.pth"
    for seed in `seq 0 29`
    do
        for shot in 1 2 3 5 10
        do
            python3 main.py --num-gpus $NGPU --config-file configs/voc/fsod.yaml \
                DATASETS.TRAIN "('voc_2007+2012_trainval_all${split}_${shot}shot_seed${seed}', )" \
                DATASETS.TEST "('voc_2007_test_all${split}', )" \
                MODEL.WEIGHTS "${CKPT_DIR}/base${split}/model_final-fsod.pth" \
                OUTPUT_DIR "${CKPT_DIR}/fsod${split}/${shot}shot/seed${seed}" \
                SOLVER.MAX_ITER "$(get_item ${MAX_ITER} ${shot})" \
                SOLVER.STEPS "$(get_item ${STEPS} ${shot})" \
                SEED $seed
        done
    done
done

python -m tools.display_results "$CKPT_DIR"
