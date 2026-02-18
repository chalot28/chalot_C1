"""Check BPE merges from Qwen tokenizer"""
from transformers import AutoTokenizer
import json

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)

# Explore the backend tokenizer
bt = tok.backend_tokenizer
model = bt.model
print('Model type:', type(model).__name__)

# Try to get merges from JSON representation
data = json.loads(bt.to_str())
model_data = data.get('model', {})
print('Model keys:', list(model_data.keys()))

if 'merges' in model_data:
    merges = model_data['merges']
    print(f'Found {len(merges)} merges!')
    print('Sample merges (first 10):')
    for m in merges[:10]:
        print(f'  {repr(m)}')
    
    # Convert merge strings to token pairs
    # Format is "tokenA tokenB" where tokenA/B are byte-level strings
    vocab = tok.get_vocab()
    print(f'\nVocab size: {len(vocab)}')
    
    # Check if we can resolve merge pairs to token IDs
    found = 0
    not_found = 0
    for m in merges[:20]:
        parts = m.split(' ', 1)
        if len(parts) == 2:
            a, b = parts
            aid = vocab.get(a)
            bid = vocab.get(b)
            if aid is not None and bid is not None:
                found += 1
            else:
                not_found += 1
                if not_found <= 5:
                    print(f'  Not found: {repr(a)} -> {aid}, {repr(b)} -> {bid}')
    
    print(f'\nMerge resolution: {found} found, {not_found} not found (out of 20)')
else:
    print('No merges found in model data')
    if 'vocab' in model_data:
        v = model_data['vocab']
        print(f'Vocab has {len(v)} entries')
        items = list(v.items())[:5]
        for k, v in items:
            print(f'  {repr(k)}: {v}')
