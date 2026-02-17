# 🚀 QWEN2.5-0.5B CONVERSION - QUICK REFERENCE CARD

## ⚡ TL;DR - Fastest Path to Running

### One-Line (Linux/Mac)
```bash
bash quick_start_qwen.sh
```

### Manual (All platforms)
```bash
# 1. Setup (chọn 1)
setup_qwen_conversion.bat              # Windows
bash setup_qwen_conversion.sh          # Linux/Mac

# 2. Convert
python qwen_to_myai.py --model Qwen/Qwen2.5-0.5B-Instruct --output qwen_moe.myai

# 3. Validate
python validate_myai.py qwen_moe.myai

# 4. Build & Run
cargo build --release
./target/release/AI_chalot_C1 qwen_moe.myai
```

---

## 📋 Command Reference

### Python Environment
```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate.bat         # Windows

# Install deps
pip install -r requirements.txt

# Verify
python --version
```

### HuggingFace
```bash
# Login (if needed)
huggingface-cli login

# Download model
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# Check cache location
ls ~/.cache/huggingface/hub/      # Linux/Mac
dir %USERPROFILE%\.cache\huggingface\hub\  # Windows
```

### Conversion
```bash
# Basic
python qwen_to_myai.py --model Qwen/Qwen2.5-0.5B-Instruct --output qwen_moe.myai

# From local path
python qwen_to_myai.py --model /path/to/downloaded/model --output qwen_moe.myai

# Custom output
python qwen_to_myai.py --model Qwen/Qwen2.5-0.5B-Instruct --output my_custom_name.myai
```

### Validation
```bash
# Check file
python validate_myai.py qwen_moe.myai

# Check multiple files
python validate_myai.py qwen_v1.myai
python validate_myai.py qwen_v2.myai
```

### Rust Build
```bash
# Debug (faster compile, slower runtime)
cargo build

# Release (slower compile, faster runtime)
cargo build --release

# Android
cargo build --release --target aarch64-linux-android

# Check build
file target/release/AI_chalot_C1                              # Linux/Mac
target\release\AI_chalot_C1.exe --help                        # Windows
```

### Run
```bash
# Interactive mode
./target/release/AI_chalot_C1 qwen_moe.myai

# With prompt
echo "Xin chào!" | ./target/release/AI_chalot_C1 qwen_moe.myai

# Verbose mode (see internals)
./target/release/AI_chalot_C1 qwen_moe.myai --verbose
```

### Android Deployment
```bash
# Build
cargo build --release --target aarch64-linux-android

# Check device
adb devices

# Push model
adb push qwen_moe.myai /sdcard/Download/

# Push binary
adb push target/aarch64-linux-android/release/AI_chalot_C1 /data/local/tmp/

# Make executable
adb shell chmod +x /data/local/tmp/AI_chalot_C1

# Run
adb shell /data/local/tmp/AI_chalot_C1 /sdcard/Download/qwen_moe.myai

# Monitor RAM
adb shell "ps -A | grep AI_chalot"

# Check CPU usage
adb shell top -n 1 | grep AI_chalot
```

---

## 🗂️ File Structure

```
AI_chalot_C1/
├── qwen_to_myai.py              ← Conversion script
├── validate_myai.py             ← Validation tool
├── requirements.txt             ← Python deps
├── setup_qwen_conversion.bat    ← Windows setup
├── setup_qwen_conversion.sh     ← Linux/Mac setup
├── quick_start_qwen.sh          ← All-in-one script
│
├── QWEN_CONVERSION_GUIDE.md     ← Full guide (600+ lines)
├── QWEN_IMPLEMENTATION_SUMMARY.md ← Tech details
├── QUICK_REFERENCE.md           ← This file
├── README.md                    ← Project overview
│
├── src/
│   ├── model/
│   │   ├── config.rs            ← Qwen config added
│   │   ├── constants.rs         ← Qwen constants
│   │   ├── brain_map.rs         ← 24-layer support
│   │   ├── header.rs
│   │   ├── engine.rs
│   │   └── ...
│   └── ...
│
└── target/                      ← Build outputs
    ├── release/
    │   └── AI_chalot_C1         ← Final binary
    └── aarch64-linux-android/
        └── release/
            └── AI_chalot_C1     ← Android binary
```

---

## 🎯 Key Parameters

### Qwen2.5-0.5B Specs
```
dim:            896
hidden_dim:     4864
n_layers:       24
n_heads:        14
vocab_size:     151936
max_seq_len:    2048

# After MoE up-cycling
n_experts:      8
top_k:          2
int4_group:     32
```

### Brain Map Layout
```
Shallow Reflex:  Layers 0-5   (6 layers)
Deep Logic:      Layers 6-17  (12 layers)
Hard Fact:       Layers 18-23 (6 layers)
```

### File Sizes
```
Original (HF):   ~1100 MB (Float32 safetensors)
Converted:       ~750 MB  (.myai format)
RAM usage:       ~250 MB  (runtime on Pixel 5)
```

---

## 🔧 Troubleshooting

### Issue: "Module not found"
```bash
# Ensure venv activated
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate.bat         # Windows

# Reinstall deps
pip install -r requirements.txt
```

### Issue: "Rust not found"
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Reload shell
source $HOME/.cargo/env
```

### Issue: "Android build fails"
```bash
# Install Android NDK
# Download from: https://developer.android.com/ndk/downloads

# Add target
rustup target add aarch64-linux-android

# Configure ~/.cargo/config.toml:
[target.aarch64-linux-android]
linker = "/path/to/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang"
```

### Issue: "File too small for header"
```bash
# Check conversion completed
ls -lh qwen_moe.myai

# Re-run conversion
python qwen_to_myai.py --model Qwen/Qwen2.5-0.5B-Instruct --output qwen_moe.myai
```

### Issue: "Out of memory on Pixel 5"
```bash
# Try reducing max_seq_len
# Edit qwen_to_myai.py line 26:
'max_seq_len': 1024,  # reduced from 2048

# Or enable paging (automatic in code)
```

### Issue: "Model outputs gibberish"
```bash
# Check validation first
python validate_myai.py qwen_moe.myai

# Verify conversion logs
# Look for "✓ Written" for all 24 layers

# Try re-downloading model
rm -rf ~/.cache/huggingface/hub/models--Qwen--*
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

---

## 📊 Performance Targets

### ThinkPad A485 (Ryzen 5 3500U)
```
Speed:    40-50 tokens/sec
RAM:      ~320 MB
Latency:  20-25 ms/token
Power:    ~8W
```

### Pixel 5 (Snapdragon 730G)
```
Speed:    20-25 tokens/sec
RAM:      ~280 MB
Latency:  40-50 ms/token
Power:    ~3W
Battery:  2-3 hours continuous
```

---

## 📚 Documentation Links

- **Full Guide**: [QWEN_CONVERSION_GUIDE.md](QWEN_CONVERSION_GUIDE.md)
- **Implementation**: [QWEN_IMPLEMENTATION_SUMMARY.md](QWEN_IMPLEMENTATION_SUMMARY.md)  
- **Architecture**: [NLLM_ARCHITECTURE.md](NLLM_ARCHITECTURE.md)
- **Optimization**: [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)

---

## ✅ Checklist

Before deployment:
- [ ] Python 3.8+ installed
- [ ] Rust 1.70+ installed
- [ ] HuggingFace CLI installed
- [ ] Virtual environment activated
- [ ] Dependencies installed (`requirements.txt`)
- [ ] Model downloaded (Qwen/Qwen2.5-0.5B-Instruct)
- [ ] Conversion completed (`qwen_moe.myai` ~750MB)
- [ ] Validation passed (`python validate_myai.py`)
- [ ] Rust build successful (`cargo build --release`)
- [ ] Test run completed (ThinkPad)
- [ ] Android build done (if deploying to mobile)
- [ ] Files pushed to device (`adb push`)

---

## 🎉 Success Indicators

You know it's working when:
- ✅ `validate_myai.py` shows "VALIDATION PASSED"
- ✅ File size is ~750MB
- ✅ Rust compilation has no errors
- ✅ Model loads in <10 seconds
- ✅ First token appears in <1 second
- ✅ Generation speed >15 tokens/sec
- ✅ RAM usage <400MB on ThinkPad, <350MB on Pixel
- ✅ Output is coherent text (not gibberish)

---

## 💡 Pro Tips

1. **Cache the model**: Download once, use many times
   ```bash
   huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
   ```

2. **Use release builds**: 3-5× faster than debug
   ```bash
   cargo build --release  # not just `cargo build`
   ```

3. **Monitor RAM**: Ensure you don't OOM
   ```bash
   watch -n 1 "ps aux | grep AI_chalot"
   ```

4. **Profile performance**: Find bottlenecks
   ```bash
   cargo build --release
   perf record ./target/release/AI_chalot_C1 qwen_moe.myai
   perf report
   ```

5. **Batch convert**: Multiple configs in parallel
   ```bash
   python qwen_to_myai.py --model Qwen/... --output qwen_v1.myai &
   python qwen_to_myai.py --model Qwen/... --output qwen_v2.myai &
   wait
   ```

---

**Keep this card handy! Bookmark it for quick command lookup! 🔖**
