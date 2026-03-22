#!/bin/bash
# TensorBoard 启动脚本
# 用法: bash view_tensorboard.sh [exp_name]

EXP_NAME=${1:-"exp5_v2"}
PORT=${2:-6006}

cd "$(dirname "$0")"

echo "============================================"
echo "Starting TensorBoard"
echo "============================================"
echo "Experiment: $EXP_NAME"
echo "Log dir: ./runs/cnn_gru_ppo_tb/$EXP_NAME"
echo "Port: $PORT"
echo ""
echo "Open in browser: http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo "============================================"

tensorboard --logdir="./runs/cnn_gru_ppo_tb/$EXP_NAME" --port=$PORT
