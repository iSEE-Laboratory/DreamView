echo $WORLD_SIZE
echo $RANK
echo $MASTER_ADDR
echo $MASTER_PORT

export TORCH_DISTRIBUTED_DEBUG=INFO
export NODE_RANK=$RANK
unset RANK

python  main.py \
        -t \
        --base configs/dreamview-32gpus.yaml \
        --gpus 0,1,2,3,4,5,6,7 \
        --scale_lr False \
        --num_nodes 4 \
        --seed 42 \
        --check_val_every_n_epoch 10 \
        --finetune_from ckpts/sd-v2.1-base-4view.pt
    