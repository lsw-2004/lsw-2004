#!/bin/bash
# run_exp6.sh - 运行行人运动预测版本 (支持静态+动态障碍物)

set -e

NO_GRAPHICS="--no-graphics"
UPDATES=""
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --graphics) NO_GRAPHICS="" ; shift ;;
        --updates) UPDATES="--updates $2" ; shift 2 ;;
        --resume) RESUME="--resume $2" ; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================"
echo "EXP6 - 行人运动预测 + 静态障碍物"
echo "============================================"
echo "观测空间结构 (共 233 维):"
echo "  - LiDAR 原始数据: 180 维"
echo "  - 原始低维状态: 7 维 (方向、距离、角度、速度等)"
echo "  - 静态障碍物特征: 6 维"
echo "  - 行人预测特征: 40 维 (5行人 × 8特征)"
echo ""
echo "静态障碍物特征 (6维):"
echo "  - 全局最小距离"
echo "  - 前/左/右区域最小距离"
echo "  - 最近障碍物角度"
echo "  - 危险程度分数"
echo ""
echo "行人预测特征 (每行人 8 维):"
echo "  - 位置 (x, y)"
echo "  - 速度 (vx, vy)"
echo "  - 距离、速度大小"
echo "  - 预测最近距离"
echo "  - 碰撞风险分数"
echo "============================================"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ppo_nav

if [ -f "./Corriidor_linux/Corridor_linux.x86_64" ]; then
    chmod +x ./Corriidor_linux/Corridor_linux.x86_64
fi

mkdir -p ./checkpoints/cnn_gru_ppo_tb/exp6
mkdir -p ./runs/cnn_gru_ppo_tb/exp6

python train_ppo_exp6.py $NO_GRAPHICS $UPDATES $RESUME
