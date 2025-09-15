#!/bin/bash
# CIFAR-10 智能优化训练测试脚本

set -e  # 出错时退出

echo "=== 🚀 CIFAR-10 智能优化训练测试 ==="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "[错误] 未找到Python"
    exit 1
fi

echo "[信息] Python版本: $(python --version)"

# 检查必要的包
echo "[信息] 检查依赖包..."
missing_packages=()

if ! python -c "import torch" 2>/dev/null; then
    missing_packages+=("torch")
fi

if ! python -c "import torchvision" 2>/dev/null; then
    missing_packages+=("torchvision")
fi

if ! python -c "import numpy" 2>/dev/null; then
    missing_packages+=("numpy")
fi

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "[错误] 缺少以下依赖包: ${missing_packages[*]}"
    echo "请运行: pip install torch torchvision numpy matplotlib"
    exit 1
fi

echo "[信息] 所有依赖包已安装"

# 检查GPU
if python -c "import torch; print('[信息] GPU可用:', torch.cuda.is_available())"; then
    if python -c "import torch; torch.cuda.is_available() and print('[信息] GPU数量:', torch.cuda.device_count())"; then
        python -c "import torch; torch.cuda.is_available() and print('[信息] GPU名称:', torch.cuda.get_device_name(0))"
    fi
fi

# 运行快速训练验证（2轮）
echo ""
echo "[信息] 🚀 开始智能优化模式验证（2轮）..."
python train.py --epochs 2 --out-dir runs/test

echo ""
echo "[完成] ✅ 测试通过！智能优化系统就绪"
echo ""
echo "🚀 推荐命令:"
echo "  智能优化模式: python train.py"
echo "  长时间训练:   python train.py --epochs 100"
echo "  手动控制:     python train.py --disable-auto --batch-size 128"
