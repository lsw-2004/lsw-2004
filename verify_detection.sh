#!/bin/bash
# 动态检测验证脚本
# 用法: bash verify_detection.sh [options]

set -e

# 默认参数
NO_VIZ="--no-viz"
EPISODES=""
STEPS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --viz)
            NO_VIZ=""
            shift
            ;;
        --episodes)
            EPISODES="--episodes $2"
            shift 2
            ;;
        --steps)
            STEPS="--steps $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash verify_detection.sh [--viz] [--episodes N] [--steps N]"
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

# 设置环境变量（无 GPU 需求）
export CUDA_VISIBLE_DEVICES=""

# 检查 Unity 环境
if [ ! -f "./Corriidor_linux/Corridor_linux.x86_64" ]; then
    echo "Error: Unity environment not found at ./Corriidor_linux/Corridor_linux.x86_64"
    exit 1
fi

# 确保 Unity 可执行
chmod +x ./Corriidor_linux/Corridor_linux.x86_64

# 打印配置
echo "============================================"
echo "Starting dynamic detection verification"
echo "============================================"
echo "Visualization: $([ -z \"$NO_VIZ\" ] && echo 'Enabled' || echo 'Disabled')"
echo "Episodes: ${EPISODES:-default (5)}"
echo "Steps: ${STEPS:-default (200)}"
echo "============================================"

# 启动验证
python verify_dynamic_detection.py $NO_VIZ $EPISODES $STEPS
