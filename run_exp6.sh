#!/bin/bash
# run_exp6.sh - 运行行人运动预测版本 (支持静态+动态障碍物)
# [优化版] 降低熵系数，增强奖励塑形

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
echo "EXP6 - 行人运动预测 + 静态障碍物 [优化版]"
echo "============================================"
echo "优化内容 (针对 6M 步平台期):"
echo "  - ent_coef: 0.008 → 0.002 (降低探索)"
echo "  - target_kl: 0.03 → 0.02 (更保守更新)"
echo "  - collision_penalty: -12 → -20 (更强避障)"
echo "  - success_bonus: 100 → 120 (更强成功激励)"
echo "  - progress_gain: 3.0 → 3.5 (更强前进激励)"
echo ""
echo "观测空间: 233 维 (LiDAR 180 + low 7 + static 6 + ped 40)"
echo "============================================"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ppo_nav

# 检查 GPU 可用性
echo "Checking GPU..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

if [ -f "./Corriidor_linux/Corridor_linux.x86_64" ]; then
    chmod +x ./Corriidor_linux/Corridor_linux.x86_64
fi

mkdir -p ./checkpoints/cnn_gru_ppo_tb/exp6
mkdir -p ./runs/cnn_gru_ppo_tb/exp6

# 使用随机 worker_id 避免端口冲突
WORKER_ID=$((RANDOM % 100 + 1))
echo "Using worker_id: $WORKER_ID"
python train_ppo_exp6.py $NO_GRAPHICS $UPDATES $RESUME --worker-id $WORKER_ID
