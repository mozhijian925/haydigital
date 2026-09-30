# 计划：完善 InfiniTalk 工作流 - 实现完整的分块视频生成

## Goal

基于 ComfyUI 工作流分析，完善 InfiniTalk 支持，实现完整的无限时长数字人口型视频生成功能。

---

## Current Context / Assumptions

### 已完成的工作

- ✅ 添加 `VERSION_WAN2_MULTI_TALK` 枚举
- ✅ 更新版本检测逻辑（`model_loader.cpp`）
- ✅ 更新模型构建器（`model_builders.cpp`）
- ✅ 更新 WanRunner 模型描述（`wan.hpp`）
- ✅ 更新视频管道条件处理（`video.cpp` 5 处）
- ✅ 创建 `AudioChunker` 类（音频分块）
- ✅ 创建 `ChunkedVideoGenerator` 存根
- ✅ 添加单元测试并通过
- ✅ 推送到 GitHub

### 缺失的功能（需要实现）

根据 ComfyUI 工作流分析，需要实现以下核心功能：

| 缺失项 | ComfyUI 节点 | 说明 |
|--------|-------------|------|
| 完整的 `generate()` 方法 | `WanVideoImageToVideoMultiTalk` | 无限时长视频生成 |
| 音频预处理 | `AudioCrop` | 裁剪、重采样、下混 |
| CLI 参数支持 | - | 启用分块模式的标志 |
| 完整测试 | `test_audio_chunker.cpp` | 已有基础测试 |

**注意**: 
- `src/runtime/audio_processing.{h,cpp}` 已存在，包含 `resample_audio()` 和 `downmix_to_mono()` 函数
- 可直接复用这些音频处理工具

---

## Architecture / Proposed Approach

**当前架构**:
```
src/pipeline/video.cpp (line 1081-1128)
  └── S2V 条件处理: 编码参考图片 + 音频 embedding + 构建 audio window
  
src/pipeline/video.cpp (line 1147-1215)
  └── prepare_video_generation_embeds: 构造 SDCondition
  └── 设置 c_ref_images 和 c_ref_audios

src/model/diffusion/wan.hpp (line 715-743)
  └── Wan forward: 处理 s2v 模式
  └── 使用 casual_audio_encoder + audio_injector
```

**方案**: 复用现有 S2V 代码，添加分块循环，实现 `ChunkedVideoGenerator::generate()`

---

## Step-by-Step Tasks

### Task 1: 实现 ChunkedVideoGenerator::generate()

**文件**: `src/pipeline/video_chunked.cpp`

**目标**: 实现完整的分块视频生成逻辑

**关键步骤**:
1. 编码参考图片为 latent（复用 `sd->encode_first_stage()`）
2. 音频重采样到 16kHz 单声道（复用 `resample_audio()` 和 `downmix_to_mono()`）
3. 音频分块（复用 `AudioChunker::chunk()`）
4. 逐块调用现有推理逻辑（需要提取 video.cpp 中的核心代码为辅助函数）
5. 重叠区域线性混合
6. 拼接所有 latent
7. 解码视频（复用 `sd->decode_first_stage()`）

**需要新增的辅助函数**（`src/pipeline/video_chunked.h` 或 `video.cpp`）:
```cpp
// 处理单个音频块的推理
std::optional<sd::Tensor<float>> process_chunk(
    StableDiffusionGGML* sd,
    const AudioChunk& chunk,
    const sd::Tensor<float>& ref_latent,
    int video_frames);
```

---

### Task 2: 提取 S2V 推理核心逻辑为可复用函数

**文件**: `src/pipeline/video.cpp`

**目标**: 将 line 1081-1128 的 S2V 处理逻辑提取为独立函数，便于 `ChunkedVideoGenerator` 复用

**需要修改**:
1. 在 `video.cpp` 开头添加辅助函数声明
2. 将现有内联代码改为调用辅助函数

**辅助函数签名**:
```cpp
static std::optional<ImageGenerationLatents> build_multitalk_latents(
    StableDiffusionGGML* sd,
    const sd_vid_gen_params_t* sd_vid_gen_params,
    const GenerationRequest& request,
    const sd::Tensor<float>& ref_image,
    const float* audio,
    int64_t audio_samples,
    int sample_rate);
```

---

### Task 3: 添加 CLI 参数支持

**文件**: `examples/cli/main.cpp`

**添加参数解析**:
```cpp
bool chunked_mode = false;
float chunk_duration = 5.0f;
float overlap_duration = 1.0f;

// 在参数解析循环中添加:
if (strcmp(argv[i], "--chunked") == 0) {
    chunked_mode = true;
} else if (strcmp(argv[i], "--chunk-duration") == 0 && i + 1 < argc) {
    chunk_duration = atof(argv[++i]);
} else if (strcmp(argv[i], "--overlap") == 0 && i + 1 < argc) {
    overlap_duration = atof(argv[++i]);
}
```

**添加生成逻辑**:
```cpp
if (chunked_mode && (sd->version == VERSION_WAN2_MULTI_TALK || 
                     sd->version == VERSION_WAN2_2_S2V)) {
    sd::ChunkedVideoGenerator::Params gen_params;
    gen_params.chunk_duration_sec = chunk_duration;
    gen_params.overlap_sec = overlap_duration;
    gen_params.fps = params.fps;
    gen_params.video_frames = params.frames;
    
    sd::ChunkedVideoGenerator generator(sd, gen_params);
    
    sd_image_t** frames = nullptr;
    int num_frames = 0;
    int fps = 0;
    
    if (generator.generate(params.init_image,
                           audio_data, audio_samples,
                           &frames, &num_frames, &fps)) {
        save_video(frames, num_frames, fps, output_path);
        // 释放内存...
    }
} else {
    // 原有逻辑
}
```

---

### Task 4: 创建完整测试脚本

**文件**: `scripts/test_infini_talk.sh`

```bash
#!/bin/bash
set -e

MODEL="${1:-path/to/Wan2_1-InfiniteTalk_Single_Q8.gguf}"
AUDIO_ENCODER="${2:-path/to/wav2vec2_chinese_fp16.safetensors}"
IMAGE="${3:-test_data/reference.jpg}"
AUDIO="${4:-test_data/speech.wav}"
OUTPUT="${5:-test_output/output.mp4}"

mkdir -p "$(dirname $OUTPUT)"

echo "=== Testing Standard S2V ==="
./build/bin/sd-cli.exe -M vid_gen \
    --diffusion-model "$MODEL" \
    --audio-encoder "$AUDIO_ENCODER" \
    -i "$IMAGE" \
    --audio "$AUDIO" \
    --video-frames 81 \
    -o "$OUTPUT" \
    -d 0

echo "=== Testing Chunked Mode ==="
./build/bin/sd-cli.exe -M vid_gen \
    --diffusion-model "$MODEL" \
    --audio-encoder "$AUDIO_ENCODER" \
    -i "$IMAGE" \
    --audio "$AUDIO" \
    --chunked \
    --chunk-duration 5.0 \
    --overlap 1.0 \
    --video-frames 81 \
    -o "${OUTPUT%.mp4}_chunked.mp4" \
    -d 0

echo "=== Verification ==="
for f in "$OUTPUT" "${OUTPUT%.mp4}_chunked.mp4"; do
    if [ -f "$f" ] && [ -s "$f" ]; then
        echo "✓ $f exists ($(stat -c%s "$f" 2>/dev/null || stat -f%z "$f") bytes)"
    else
        echo "✗ $f missing or empty"
    fi
done
```

---

## Tests / Validation

### 单元测试

```bash
# 运行音频分块测试
cd D:/workspace/stable-diffusion.cpp
bash scripts/run_audio_chunker_test.sh
# 预期：All audio chunker tests passed!
```

### 集成测试（需要 GPU 和模型）

```bash
# 测试标准 S2V
./build/bin/sd-cli.exe -M vid_gen \
    --diffusion-model path/to/Wan2.2-S2V-14B-Q8.gguf \
    --audio-encoder path/to/wav2vec2.safetensors \
    -i test_data/ref.jpg --audio test_data/audio.wav \
    -o /tmp/s2v_test.mp4

test -s /tmp/s2v_test.mp4 && echo "✓ S2V: PASS" || echo "✗ S2V: FAIL"

# 测试 MultiTalk
./build/bin/sd-cli.exe -M vid_gen \
    --diffusion-model path/to/Wan2_1-InfiniteTalk_Q8.gguf \
    --audio-encoder path/to/wav2vec2.safetensors \
    -i test_data/ref.jpg --audio test_data/audio.wav \
    --chunked --chunk-duration 5.0 \
    -o /tmp/multitalk_test.mp4

test -s /tmp/multitalk_test.mp4 && echo "✓ MultiTalk: PASS" || echo "✗ MultiTalk: FAIL"
```

---

## Risks, Tradeoffs, and Open Questions

### 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 分块接缝 visible | 中 | 高 | 实现重叠混合 |
| VRAM OOM | 高 | 高 | 流式解码 |
| 口型不同步 | 低 | 高 | 精确时间戳对齐 |

### 待解决问题

1. **重叠混合策略**: 线性混合还是扩散重采样？
2. **性能优化**: 是否支持 LightX2V LoRA 加速？
3. **音频预处理**: 是否需要支持实时音频流？

---

**计划文件**: `.hermes/plans/2026-09-30_091500-infini-talk-completion.md`
