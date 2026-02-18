"""Export BPE merges from Qwen tokenizer to .mytok format"""
from transformers import AutoTokenizer
import json
import struct
from pathlib import Path

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)

# Get BPE merges
bt = tok.backend_tokenizer
data = json.loads(bt.to_str())
merges_raw = data['model']['merges']
print(f"Found {len(merges_raw)} merges")

# Get vocab
vocab = tok.get_vocab()  # {string: id}
vocab_size = max(vocab.values()) + 1
print(f"Vocab size: {vocab_size}")

# Convert merges to token ID pairs
merges_ids = []
skipped = 0
for m in merges_raw:
    a, b = m  # Each merge is already [str, str]
    aid = vocab.get(a)
    bid = vocab.get(b)
    if aid is not None and bid is not None:
        merges_ids.append((aid, bid))
    else:
        skipped += 1

print(f"Converted: {len(merges_ids)} merges, skipped {skipped}")
print(f"Sample merges: {merges_ids[:5]}")

# Build reverse vocab: id -> bytes
id_to_bytes = {}
for token_str, token_id in vocab.items():
    try:
        token_bytes = token_str.encode('utf-8')
    except:
        token_bytes = b''
    id_to_bytes[token_id] = token_bytes

# Write .mytok file
output_path = 'qwen_moe.mytok'
with open(output_path, 'wb') as f:
    # Header
    f.write(struct.pack('<I', 0x544F4B45))  # Magic "TOKE"
    f.write(struct.pack('<I', 2))            # Version  
    f.write(struct.pack('<I', vocab_size))
    f.write(struct.pack('<I', len(merges_ids)))
    
    # Vocab entries
    for token_id in range(vocab_size):
        token_bytes = id_to_bytes.get(token_id, b'')
        if len(token_bytes) > 65535:
            token_bytes = token_bytes[:65535]
        f.write(struct.pack('<H', len(token_bytes)))
        f.write(token_bytes)
    
    # Merges
    for (a, b) in merges_ids:
        f.write(struct.pack('<II', a, b))

file_size = Path(output_path).stat().st_size
print(f"Written: {file_size / 1024:.1f} KB ({vocab_size} tokens, {len(merges_ids)} merges)")

# Verify: encode a test string
test = "The capital of France is"
correct_ids = tok.encode(test, add_special_tokens=False)
print(f"\nQwen encodes '{test}' -> {correct_ids}")
