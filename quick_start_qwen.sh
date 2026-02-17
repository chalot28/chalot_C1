#!/bin/bash
# ============================================================================
# quick_start_qwen.sh - Script tất-cả-trong-một để convert Qwen2.5-0.5B
# ============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  QWEN2.5-0.5B → AI_CHALOT_C1 MoE CONVERSION                       ║"
echo "║  Quick Start Script                                                ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
info "Kiểm tra môi trường..."

# Check Python
if ! command -v python3 &> /dev/null; then
    error "Python3 chưa được cài đặt!"
    echo "  Cài đặt: sudo apt install python3 python3-pip  # Ubuntu/Debian"
    echo "           brew install python3                  # macOS"
    exit 1
fi
success "Python3: $(python3 --version)"

# Check Rust
if ! command -v cargo &> /dev/null; then
    error "Rust chưa được cài đặt!"
    echo "  Cài đặt: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi
success "Rust: $(cargo --version | cut -d' ' -f2)"

# Check disk space
available_gb=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$available_gb" -lt 5 ]; then
    warn "Chỉ còn ${available_gb}GB dung lượng. Cần ít nhất 5GB!"
fi

echo ""
info "═══════════════════════════════════════════════════════════════"
info "BƯỚC 1: Thiết lập Python Environment"
info "═══════════════════════════════════════════════════════════════"

# Create venv if not exists
if [ ! -d "venv" ]; then
    info "Tạo virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate
success "Virtual environment activated"

# Install dependencies
info "Cài đặt dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
success "Dependencies installed"

echo ""
info "═══════════════════════════════════════════════════════════════"
info "BƯỚC 2: Tải Qwen2.5-0.5B từ HuggingFace"
info "═══════════════════════════════════════════════════════════════"

MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"

if [ -d "$MODEL_PATH" ]; then
    success "Model đã tồn tại: $MODEL_PATH"
else
    info "Đang tải model (~1GB, có thể mất 5-10 phút)..."
    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
    success "Model downloaded"
fi

echo ""
info "═══════════════════════════════════════════════════════════════"
info "BƯỚC 3: Chuyển đổi Qwen → .myai"
info "═══════════════════════════════════════════════════════════════"

OUTPUT_FILE="qwen_moe.myai"

if [ -f "$OUTPUT_FILE" ]; then
    warn "File $OUTPUT_FILE đã tồn tại. Xóa và tạo mới? [y/N]"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm "$OUTPUT_FILE"
        info "Đã xóa file cũ"
    else
        info "Giữ file cũ, bỏ qua conversion"
        skip_conversion=true
    fi
fi

if [ -z "$skip_conversion" ]; then
    info "Bắt đầu conversion (5-10 phút)..."
    python qwen_to_myai.py --model "$MODEL_PATH" --output "$OUTPUT_FILE"
    success "Conversion hoàn thành!"
fi

echo ""
info "═══════════════════════════════════════════════════════════════"
info "BƯỚC 4: Validate file .myai"
info "═══════════════════════════════════════════════════════════════"

python validate_myai.py "$OUTPUT_FILE"

echo ""
info "═══════════════════════════════════════════════════════════════"
info "BƯỚC 5: Build Rust Engine"
info "═══════════════════════════════════════════════════════════════"

info "Building release binary..."
cargo build --release
success "Build thành công!"

echo ""
info "═══════════════════════════════════════════════════════════════"
info "BƯỚC 6: Test trên ThinkPad"
info "═══════════════════════════════════════════════════════════════"

info "Chạy test inference..."
./target/release/AI_chalot_C1 "$OUTPUT_FILE" <<EOF
Xin chào! Bạn là ai?
exit
EOF

echo ""
success "╔════════════════════════════════════════════════════════════════════╗"
success "║  🎉 HOÀN THÀNH!                                                    ║"
success "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "File output: $OUTPUT_FILE ($(du -h $OUTPUT_FILE | cut -f1))"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Test thêm trên ThinkPad:"
echo "   ./target/release/AI_chalot_C1 $OUTPUT_FILE"
echo ""
echo "2. Deploy lên Pixel 5 (Android):"
echo "   # Build cho Android"
echo "   cargo build --release --target aarch64-linux-android"
echo ""
echo "   # Push lên device"
echo "   adb push $OUTPUT_FILE /sdcard/Download/"
echo "   adb push target/aarch64-linux-android/release/AI_chalot_C1 /data/local/tmp/"
echo "   adb shell chmod +x /data/local/tmp/AI_chalot_C1"
echo ""
echo "   # Run"
echo "   adb shell /data/local/tmp/AI_chalot_C1 /sdcard/Download/$OUTPUT_FILE"
echo ""
echo "3. Đọc thêm: QWEN_CONVERSION_GUIDE.md"
echo ""
