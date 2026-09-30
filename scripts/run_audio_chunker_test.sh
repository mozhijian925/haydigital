#!/bin/bash
# 编译和运行音频分块单元测试

set -e

echo "=== Building audio chunker test ==="

# 编译测试
g++ -std=c++17 \
    -I src \
    -I include \
    test/test_audio_chunker.cpp \
    src/audio/audio_chunker.cpp \
    -o test_audio_chunker

echo "✓ Build successful"
echo ""
echo "=== Running tests ==="

# 运行测试
./test_audio_chunker

echo ""
echo "=== Cleaning up ==="
rm -f test_audio_chunker

echo "✓ All tests completed"
