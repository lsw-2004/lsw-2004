#!/bin/bash
# run_exp5_v4.sh - 运行简化版 exp5_v4

set -e

NO_GRAPHICS="--no-graphics"
UPDATES=""
RESUME=""
CUDA_DEVICES="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --graphics)
            NO_GRAPHICS=""
            shift
            ;;
        --updates)
            UPDATES="--updates $2"
            shift 2
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --cuda)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

echo "============================================"
echo "EXP5_V4 简化版训练"
echo "============================================"
echo "改进: 固定entropy系数, 更强碰撞惩罚, 简化模型"
echo "============================================"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ppo_nav

if [ -f "./Corriidor_linux/Corridor_linux.x86_64" ]; then
    chmod +x ./Corriidor_linux/Corridor_linux.x86_64
fi

mkdir -p ./checkpoints/cnn_gru_ppo_tb/exp5_v4
mkdir -p ./runs/cnn_gru_ppo_tb/exp5_v4

python train_ppo_exp5_v4.py $NO_GRAPHICS $UPDATES $RESUME
