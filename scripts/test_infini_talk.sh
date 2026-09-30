#!/bin/bash
# InfiniTalk 分块视频生成测试脚本

set -e

echo "=========================================="
echo "InfiniTalk Chunked Video Generation Test"
echo "=========================================="
echo ""

# 参数设置
MODEL="${1:-path/to/Wan2_1-InfiniteTalk_Single_Q8.gguf}"
AUDIO_ENCODER="${2:-path/to/wav2vec2_chinese_fp16.safetensors}"
IMAGE="${3:-test_data/reference.jpg}"
AUDIO="${4:-test_data/speech.wav}"
OUTPUT_DIR="${5:-test_output}"
BINARY="${6:-./build/bin/sd-cli.exe}"

mkdir -p "$OUTPUT_DIR"

echo "=== Configuration ==="
echo "Model: $MODEL"
echo "Audio Encoder: $AUDIO_ENCODER"
echo "Reference Image: $IMAGE"
echo "Audio: $AUDIO"
echo "Output Directory: $OUTPUT_DIR"
echo "Binary: $BINARY"
echo ""

# 检查文件是否存在
check_files() {
    local missing=0
    for f in "$MODEL" "$AUDIO_ENCODER" "$IMAGE" "$AUDIO"; do
        if [ ! -f "$f" ]; then
            echo "WARNING: File not found: $f"
            missing=1
        fi
    done
    if [ "$missing" -eq 1 ]; then
        echo ""
        echo "Please provide valid file paths as arguments:"
        echo "  $0 <model> <audio_encoder> <image> <audio> [output_dir] [binary]"
        exit 1
    fi
}

check_files

echo "=== Testing Standard S2V (no chunked) ==="
"$BINARY" -M vid_gen \
    --diffusion-model "$MODEL" \
    --audio-encoder "$AUDIO_ENCODER" \
    -i "$IMAGE" \
    --audio "$AUDIO" \
    --video-frames 81 \
    -o "$OUTPUT_DIR/s2v_test.mp4" \
    -d 0

if [ -f "$OUTPUT_DIR/s2v_test.mp4" ] && [ -s "$OUTPUT_DIR/s2v_test.mp4" ]; then
    echo "✓ S2V standard generation: PASS"
else
    echo "✗ S2V standard generation: FAIL"
fi

echo ""
echo "=== Testing Chunked Mode ==="
"$BINARY" -M vid_gen \
    --diffusion-model "$MODEL" \
    --audio-encoder "$AUDIO_ENCODER" \
    -i "$IMAGE" \
    --audio "$AUDIO" \
    --chunked \
    --chunk-duration 5.0 \
    --overlap 1.0 \
    --video-frames 81 \
    -o "$OUTPUT_DIR/multitalk_chunked.mp4" \
    -d 0

if [ -f "$OUTPUT_DIR/multitalk_chunked.mp4" ] && [ -s "$OUTPUT_DIR/multitalk_chunked.mp4" ]; then
    echo "✓ MultiTalk chunked generation: PASS"
else
    echo "✗ MultiTalk chunked generation: FAIL"
fi

echo ""
echo "=== Verification ==="
for f in "$OUTPUT_DIR/s2v_test.mp4" "$OUTPUT_DIR/multitalk_chunked.mp4"; do
    if [ -f "$f" ] && [ -s "$f" ]; then
        size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")
        echo "✓ $f ($size bytes)"
    else
        echo "✗ $f missing or empty"
    fi
done

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
