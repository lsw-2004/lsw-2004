#!/bin/bash
# exp5_v2 训练启动脚本
# 用法: bash run_exp5_v2.sh [options]

set -e

# 默认参数
NO_GRAPHICS="--no-graphics"
UPDATES=""
RESUME=""

# 解析命令行参数
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
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_exp5_v2.sh [--graphics] [--updates N] [--resume PATH]"
            exit 1
            ;;
    esac
done

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 检查虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用 GPU 0，可改为 1 或 0,1

# 检查 Unity 环境
if [ ! -f "./Corriidor_linux/Corridor_linux.x86_64" ]; then
    echo "Error: Unity environment not found at ./Corriidor_linux/Corridor_linux.x86_64"
    exit 1
fi

# 确保 Unity 可执行
chmod +x ./Corriidor_linux/Corridor_linux.x86_64

# 创建必要目录
mkdir -p ./checkpoints/cnn_gru_ppo_tb/exp5_v2
mkdir -p ./runs/cnn_gru_ppo_tb/exp5_v2

# 打印配置
echo "============================================"
echo "Starting exp5_v2 training"
echo "============================================"
echo "No graphics: $([ -n \"$NO_GRAPHICS\" ] && echo 'Yes' || echo 'No')"
echo "Updates: ${UPDATES:-default (3000)}"
echo "Resume: ${RESUME:-'No'}"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# 启动训练
python train_ppo_exp5_v2.py $NO_GRAPHICS $UPDATES $RESUME
