#!/bin/bash
# run_exp5_v5.sh - 运行改进版 exp5_v5

set -e

NO_GRAPHICS="--no-graphics"
UPDATES=""
RESUME=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --graphics) NO_GRAPHICS="" ; shift ;;
        --updates) UPDATES="--updates $2" ; shift 2 ;;
        --resume) RESUME="--resume $2" ; shift 2 ;;
        --no-residual) EXTRA_ARGS="$EXTRA_ARGS --no-residual" ; shift ;;
        --no-trend) EXTRA_ARGS="$EXTRA_ARGS --no-trend" ; shift ;;
        *) shift ;;
    esac
done

echo "============================================"
echo "EXP5_V5 - 基于成功版本的改进"
echo "============================================"
echo "与 exp3/exp4 的区别:"
echo "  [特征] +4维趋势特征 (min_dist变化趋势)"
echo "  [架构] +残差连接 (GRU输出+输入投影)"
echo "  [训练] 自适应熵系数 (根据成功率调整)"
echo "  [奖励] progress_gain: 2.5->3.0"
echo "  [奖励] collision_penalty: -8.0->-10.0"
echo "  [奖励] success_bonus: 80.0->100.0"
echo "============================================"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ppo_nav

if [ -f "./Corriidor_linux/Corridor_linux.x86_64" ]; then
    chmod +x ./Corriidor_linux/Corridor_linux.x86_64
fi

mkdir -p ./checkpoints/cnn_gru_ppo_tb/exp5_v5
mkdir -p ./runs/cnn_gru_ppo_tb/exp5_v5

python train_ppo_exp5_v5.py $NO_GRAPHICS $UPDATES $RESUME $EXTRA_ARGS
