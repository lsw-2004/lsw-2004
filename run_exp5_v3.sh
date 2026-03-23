#!/bin/bash
# 实验5 v3 训练脚本
# 目标: 成功率 0.8

echo "=========================================="
echo "实验5 v3: 目标成功率 0.8"
echo "核心改进:"
echo "  - Kalman 滤波器动态追踪"
echo "  - 社交力模型奖励"
echo "  - 预测性安全层"
echo "  - 自动课程学习"
echo "  - 辅助任务学习"
echo "=========================================="

# 配置
PYTHON_SCRIPT="train_ppo_exp5_v3.py"
LOG_DIR="./runs/cnn_gru_ppo_tb/exp5_v3"
SAVE_DIR="./checkpoints/cnn_gru_ppo_tb/exp5_v3"

# 创建目录
mkdir -p $LOG_DIR
mkdir -p $SAVE_DIR

# 检查是否有恢复检查点
RESUME_FLAG=""
if [ -f "$SAVE_DIR/best_model.pt" ]; then
    read -p "Found existing checkpoint. Resume from best_model.pt? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        RESUME_FLAG="--resume $SAVE_DIR/best_model.pt"
        echo "Resuming from best_model.pt"
    fi
fi

# 运行训练
echo ""
echo "Starting training..."
echo "Logs will be saved to: $LOG_DIR"
echo "Checkpoints will be saved to: $SAVE_DIR"
echo ""

python $PYTHON_SCRIPT \
    --no-graphics \
    $RESUME_FLAG \
    2>&1 | tee -a "$LOG_DIR/training.log"

echo ""
echo "Training completed!"
echo "To view tensorboard: tensorboard --logdir=$LOG_DIR"
