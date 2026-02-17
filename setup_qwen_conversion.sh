#!/bin/bash
# ============================================================================
# setup_qwen_conversion.sh - Thiết lập môi trường chuyển đổi Qwen2.5-0.5B
# ============================================================================

set -e

echo "========================================"
echo "QWEN2.5-0.5B Conversion Setup"
echo "========================================"
echo ""

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 không được cài đặt!"
    echo "Vui lòng cài Python 3.8+ từ python.org hoặc package manager"
    exit 1
fi

echo "[1/4] Kiểm tra Python... OK"
echo ""

# Tạo virtual environment
if [ ! -d "venv" ]; then
    echo "[2/4] Tạo virtual environment..."
    python3 -m venv venv
else
    echo "[2/4] Virtual environment đã tồn tại"
fi
echo ""

# Activate venv
source venv/bin/activate

# Cài dependencies
echo "[3/4] Cài đặt dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo ""

# Kiểm tra HuggingFace CLI
echo "[4/4] Kiểm tra HuggingFace Hub..."
if ! command -v huggingface-cli &> /dev/null; then
    echo "[WARNING] HuggingFace CLI không được cài đặt"
    echo "Đang cài đặt..."
    pip install huggingface-hub[cli]
fi
echo ""

echo "========================================"
echo "Setup hoàn thành!"
echo "========================================"
echo ""
echo "NEXT STEPS:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Đăng nhập HuggingFace (nếu cần):"
echo "     huggingface-cli login"
echo ""
echo "  3. Chạy script chuyển đổi:"
echo "     python qwen_to_myai.py --model Qwen/Qwen2.5-0.5B-Instruct --output qwen_moe.myai"
echo ""
echo "  4. Hoặc tải model trước:"
echo "     huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct"
echo "     python qwen_to_myai.py --model ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct --output qwen_moe.myai"
echo ""
