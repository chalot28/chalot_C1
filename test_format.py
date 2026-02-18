#!/usr/bin/env python3
"""Quick test: convert only 1 layer to verify format"""

import sys
sys.path.insert(0, '.')

from qwen_to_myai import *

# Override to only convert 1 layer
old_write = write_layers_moe

def test_write_layers(f, weights, config):
    # Backup  
    original_layers = config['n_layers']
    config['n_layers'] = 1  # Only 1 layer
    old_write(f, weights, config)
    config['n_layers'] = original_layers

# Monkey patch
import qwen_to_myai as qwen_mod
qwen_mod.write_layers_moe = test_write_layers

# Run conversion
if __name__ == '__main__':
    print("TESTING: Converting only 1 layer...")
    convert_qwen_to_myai('Qwen/Qwen2.5-0.5B-Instruct', 'test1layer.myai')
