@echo off
REM ============================================================================
REM setup_qwen_conversion.bat - Thiết lập môi trường chuyển đổi Qwen2.5-0.5B
REM ============================================================================

echo ========================================
echo QWEN2.5-0.5B Conversion Setup
echo ========================================
echo.

REM Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python khong duoc cai dat!
    echo Vui long cai Python 3.8+ tu python.org
    pause
    exit /b 1
)

echo [1/4] Kiem tra Python... OK
echo.

REM Tạo virtual environment
if not exist "venv" (
    echo [2/4] Tao virtual environment...
    python -m venv venv
) else (
    echo [2/4] Virtual environment da ton tai
)
echo.

REM Activate venv
call venv\Scripts\activate.bat

REM Cài dependencies
echo [3/4] Cai dat dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.

REM Kiểm tra HuggingFace CLI
echo [4/4] Kiem tra HuggingFace Hub...
huggingface-cli --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] HuggingFace CLI khong duoc cai dat
    echo Dang cai dat...
    pip install huggingface-hub[cli]
)
echo.

echo ========================================
echo Setup hoan thanh!
echo ========================================
echo.
echo NEXT STEPS:
echo   1. Dang nhap HuggingFace (neu can):
echo      huggingface-cli login
echo.
echo   2. Chay script chuyen doi:
echo      python qwen_to_myai.py --model Qwen/Qwen2.5-0.5B-Instruct --output qwen_moe.myai
echo.
echo   3. Hoac tai model truoc:
echo      huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
echo      python qwen_to_myai.py --model models/Qwen2.5-0.5B-Instruct --output qwen_moe.myai
echo.

pause
