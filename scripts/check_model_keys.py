#!/usr/bin/env python3
"""检查 GGUF 模型文件的关键键名"""

import struct
import sys
from pathlib import Path


def read_gguf_keys(path: str):
    """读取 GGUF 文件的所有张量键名和形状"""
    keys = []
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        version = struct.unpack("<I", f.read(4))[0]
        n_kv, n_tensors = struct.unpack("<QQ", f.read(16))

        # Skip key-value pairs
        for _ in range(n_kv):
            n_chars = struct.unpack("<I", f.read(4))[0]
            f.read(n_chars)  # skip key
            f.read(8)  # skip value type
            # Simplified: skip value data based on type
            value_type = struct.unpack("<I", f.read(8 - 4))[0] if False else 0
            # This is a simplified parser, just count tensors

        # Read tensor info
        for _ in range(n_tensors):
            n_chars = struct.unpack("<I", f.read(4))[0]
            name = f.read(n_chars).decode("utf-8")
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = list(struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims)))
            keys.append((name, shape))

    return keys


def analyze_model(path):
    keys = read_gguf_keys(path)

    # 分类统计
    audio_keys = [
        k
        for k in keys
        if any(x in k[0].lower() for x in ["audio", "injector", "wav2vec", "talk"])
    ]
    wan_keys = [k for k in keys if "model.diffusion_model" in k[0]]

    print(f"Total tensors: {len(keys)}")
    print(f"Wan model keys: {len(wan_keys)}")
    print(f"Audio-related keys: {len(audio_keys)}")

    # 检查关键组件
    has_causal_audio = any(
        "casual_audio_encoder" in k[0] or "causal_audio" in k[0] for k in keys
    )
    has_audio_injector = any("audio_injector" in k[0] for k in keys)
    has_wav2vec = any("wav2vec2" in k[0] for k in keys)
    has_multi_talk = any("multi_talk" in k[0].lower() for k in keys)

    print(f"\nComponent check:")
    print(f"  Causal audio encoder: {has_causal_audio}")
    print(f"  Audio injector: {has_audio_injector}")
    print(f"  Wav2Vec2: {has_wav2vec}")
    print(f"  MultiTalk specific: {has_multi_talk}")

    # 输出音频相关键
    if audio_keys:
        print(f"\nAudio keys (first 30):")
        for name, shape in sorted(audio_keys, key=lambda x: x[0])[:30]:
            print(f"  {name}: {shape}")

    # 判断兼容性
    print(f"\n{'='*50}")
    if has_causal_audio and has_audio_injector and has_wav2vec:
        print("✓ 模型兼容 S2V 架构，可直接使用现有管线")
        return "compatible"
    elif has_multi_talk:
        print("⚠ 模型有特殊组件，需要自定义处理")
        return "needs_custom"
    else:
        print("✗ 模型不兼容，需要全新实现")
        return "incompatible"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model_keys.py <model.gguf>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    result = analyze_model(str(path))
    sys.exit(0 if result == "compatible" else 1)
