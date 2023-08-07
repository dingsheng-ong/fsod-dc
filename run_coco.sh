#!/bin/bash

export DETECTRON2_DATASETS="./dataset"
EXP_TAG=`[ -z "$1" ] && echo "$(date +%s)" || echo "$1"`
CKPT_DIR="checkpoints/coco/${EXP_TAG}"
NGPU=4

. ./hashmap.sh

MAX_ITER=$(create_hashmap)
STEPS=$(create_hashmap)

trap "exit" INT
trap "cleanup $MAX_ITER && cleanup $STEPS" exit

add_item "$MAX_ITER" 1 3200
add_item "$MAX_ITER" 2 4000
add_item "$MAX_ITER" 3 4800
add_item "$MAX_ITER" 5 6000
add_item "$MAX_ITER" 10 6000
add_item "$MAX_ITER" 30 12000

add_item "$STEPS" 1 "(2560, )"
add_item "$STEPS" 2 "(3200, )"
add_item "$STEPS" 3 "(3840, )"
add_item "$STEPS" 5 "(4800, )"
add_item "$STEPS" 10 "(4800, )"
add_item "$STEPS" 30 "(9600, )"

python3 main.py --num-gpus $NGPU --config-file configs/coco/base.yaml \
    MODEL.WEIGHTS "./pretrain/R-101.pkl" \
    OUTPUT_DIR "${CKPT_DIR}/base" \
    SEED 0
python3 -m tools.ckpt_surgery -d coco -m init -s 0 \
    "${CKPT_DIR}/base${split}/model_final.pth"
for seed in `seq 0 9`
do
    for shot in 1 2 3 5 10 30
    do
        python3 main.py --num-gpus $NGPU --config-file configs/coco/fsod.yaml \
            DATASETS.TRAIN "('coco_2014_trainval_all_${shot}shot_seed${seed}', )" \
            MODEL.WEIGHTS "${CKPT_DIR}/base/model_final-fsod.pth" \
            OUTPUT_DIR "${CKPT_DIR}/fsod/${shot}shot/seed${seed}" \
            SOLVER.MAX_ITER "$(get_item ${MAX_ITER} ${shot})" \
            SOLVER.STEPS "$(get_item ${STEPS} ${shot})" \
            SEED $seed
    done
done

python -m tools.display_results "$CKPT_DIR"
