#!/bin/bash
# run_exp7.sh - 运行 360° LiDAR 行人运动预测版本
# 基于 Exp6，LiDAR 维度升级为 360° 全向感知

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
echo "EXP7 - 360° LiDAR 行人运动预测"
echo "============================================"
echo "核心改动 (基于 Exp6):"
echo "  - LiDAR 维度: 180 -> 360 (全向感知)"
echo "  - 其他配置与 Exp6 完全一致"
echo ""
echo "观测空间: 413 维 (LiDAR 360 + low 7 + static 6 + ped 40)"
echo "============================================"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ppo_nav

# 检查 GPU 可用性
echo "Checking GPU..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

if [ -f "./Corridor_linux_360/Corridor_linux_360.x86_64" ]; then
    chmod +x ./Corridor_linux_360/Corridor_linux_360.x86_64
fi

mkdir -p ./checkpoints/cnn_gru_ppo_tb/exp7
mkdir -p ./runs/cnn_gru_ppo_tb/exp7

python train_ppo_exp7.py $NO_GRAPHICS $UPDATES $RESUME
