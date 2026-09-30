# 计划：完善 InfiniTalk 工作流 - 实现完整的分块视频生成

## Goal

实现完整的 InfiniTalk 图片+音频驱动数字人口型视频生成功能，支持无限时长。

---

## Current Context / Assumptions

### 已完成的工作

| 项目 | 状态 | 文件位置 |
|------|------|----------|
| `VERSION_WAN2_MULTI_TALK` 枚举 | ✅ | `src/model.h:138` |
| 版本检测逻辑 | ✅ | `src/model_loader.cpp:500-530` |
| 模型构建器支持 | ✅ | `src/pipeline/model_builders.cpp:255` |
| Wan 模型描述 | ✅ | `src/model/diffusion/wan.hpp:933` |
| 视频管道条件处理 | ✅ | `src/pipeline/video.cpp:1081-1128` |
| `AudioChunker` 类 | ✅ | `src/audio/audio_chunker.{h,cpp}` |
| `ChunkedVideoGenerator` 框架 | ✅ | `src/pipeline/video_chunked.{h,cpp}` |
| 单元测试 | ✅ | `test/test_audio_chunker.cpp` (6 tests pass) |
| 音频处理工具 | ✅ | `src/runtime/audio_processing.{h,cpp}` |

### 缺失的功能（需要实现）

| 缺失项 | 严重程度 | 说明 |
|--------|----------|------|
| **完整的 `generate()` 实现** | 🔴 关键 | `video_chunked.cpp:45` 是空 stub |
| **S2V 推理逻辑复用** | 🔴 关键 | 需要提取为可复用函数 |
| **CLI 参数支持** | 🟡 重要 | 缺少 `--chunked` 标志 |
| **端到端测试脚本** | 🟡 重要 | 缺少集成测试 |

### 关键代码路径

```
examples/cli/main.cpp:920
  └── generate_video() → src/stable-diffusion.cpp:751
       └── sd::pipeline::generate_video() → src/pipeline/video.cpp:1537
            ├── prepare_video_generation_latents() (line 464-1146)
            │    ├── encode_first_stage()  # 编码参考图片 (line 1098)
            │    ├── get_audio_embedding()  # 音频 embedding (line 1114)
            │    └── build_s2v_audio_window()  # 构建音频窗口 (line 426-462)
            ├── prepare_video_generation_embeds() (line 1148-1215)
            │    └── 设置 SDCondition.c_ref_images / c_ref_audios
            └── sd->sample() (line 1681-1706)
                 └── decode_video_outputs() (line 1935)
```

---

## Architecture / Proposed Approach

**当前架构**:
- S2V 处理逻辑内联在 `video.cpp:1081-1128`
- 直接调用 `sd->sample()` 和 `sd->decode_first_stage()`
- 没有分块支持

**方案**:
1. 将 S2V 处理逻辑提取为独立辅助函数
2. 在 `ChunkedVideoGenerator` 中复用该逻辑
3. 添加循环和重叠混合支持

---

## Step-by-Step Tasks

### Task 1: 提取 S2V 条件处理为辅助函数

**文件**: `src/pipeline/video.cpp`

**目标**: 将 line 1081-1128 的内联代码提取为独立函数

**步骤**:
1. 在 `src/pipeline/video.cpp` 开头添加函数声明（约 line 420 附近）
2. 将现有内联代码替换为函数调用

**函数签名**:
```cpp
static bool prepare_multitalk_conditioning(
    StableDiffusionGGML* sd,
    const sd_vid_gen_params_t* sd_vid_gen_params,
    const GenerationRequest& request,
    const sd::Tensor<float>& ref_image,
    const float* audio_data,
    int64_t audio_samples,
    int sample_rate,
    ImageGenerationLatents& latents);
```

**预期输出**: 编译通过，现有测试仍通过。

---

### Task 2: 实现 ChunkedVideoGenerator::process_chunk()

**文件**: `src/pipeline/video_chunked.cpp`

**目标**: 实现单个音频块的推理逻辑

**关键步骤**:
1. 调用 `prepare_multitalk_conditioning()` 构建条件
2. 调用 `prepare_video_generation_embeds()` 构建 embeds
3. 调用 `sd_->sample()` 进行去噪
4. 返回 latent tensor

**代码框架**:
```cpp
std::optional<sd::Tensor<float>> ChunkedVideoGenerator::process_chunk(
    const AudioChunk& chunk,
    const sd::Tensor<float>& ref_latent) {
    
    // 1. 构建该块的条件
    ImageGenerationLatents latents;
    latents.ref_latents.push_back(ref_latent);
    
    // 2. 准备 conditioning（提取 video.cpp:1081-1128）
    GenerationRequest req = build_request_for_chunk(chunk);
    // 调用 prepare_multitalk_conditioning()
    
    // 3. 准备 embeds
    auto embeds = prepare_video_generation_embeds(sd_, &params_, req, latents);
    if (!embeds.has_value()) {
        return std::nullopt;
    }
    
    // 4. 调用采样
    auto result = sd_->sample(
        sd_->diffusion_model,
        true,
        latents.init_latent,
        sd::Tensor<float>::randn_like(latents.init_latent, sd_->rng),
        embeds->cond,
        SDCondition(),  // 简化：不使用 uncond
        embeds->img_uncond,
        sd::Tensor<float>(),
        0.f,
        params_->sample_params.guidance,
        0.f,  // eta
        0,    // shifted_timestep
        EULER_SAMPLE_METHOD,
        sd_->is_flow_denoiser(),
        nullptr,
        params_->sample_params.sigmas,
        latents.ref_latents,
        RefImageParams(),
        sd::Tensor<float>(),
        sd::Tensor<float>(),
        0.f,
        latents.audio_length,
        static_cast<float>(req.fps),
        nullptr,
        true,
        latents.video_positions);
    
    return result;
}
```

---

### Task 3: 实现 ChunkedVideoGenerator::generate()

**文件**: `src/pipeline/video_chunked.cpp`

**目标**: 实现完整的分块生成流程

**关键步骤**:
1. 编码参考图片为 latent
2. 音频预处理（重采样、下混）
3. 音频分块
4. 逐块推理
5. 重叠区域混合
6. 拼接所有 latent
7. 解码视频

**代码框架**:
```cpp
bool ChunkedVideoGenerator::generate(
    const sd_image_t& ref_image,
    const float* audio,
    int64_t audio_samples,
    sd_image_t*** output_frames,
    int* num_frames_out,
    int* fps_out) {
    
    // 1. 编码参考图片
    auto ref_tensor = sd_image_to_tensor(ref_image);
    ref_tensor = ensure_image_tensor_channels(ref_tensor, sd_->get_image_channels());
    auto ref_latent = sd_->encode_first_stage(ref_tensor);
    if (ref_latent.empty()) {
        LOG_ERROR("failed to encode reference image");
        return false;
    }
    
    // 2. 音频预处理
    auto mono = sd::audio::downmix_to_mono(audio, audio_samples, 1);
    mono = sd::audio::resample_audio(mono.data(), mono.size(), 44100, 16000);
    
    // 3. 音频分块
    auto chunks = chunker_.chunk(mono.data(), mono.size());
    if (chunks.empty()) {
        LOG_ERROR("no audio chunks generated");
        return false;
    }
    
    // 4. 逐块推理
    std::vector<sd::Tensor<float>> all_latents;
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto chunk_latent = process_chunk(chunks[i], ref_latent);
        if (!chunk_latent.has_value()) {
            LOG_ERROR("chunk %zu failed", i);
            return false;
        }
        all_latents.push_back(std::move(chunk_latent.value()));
    }
    
    // 5. 重叠混合
    // TODO: 实现 blend_chunks()
    
    // 6. 拼接 latent
    auto final_latent = all_latents[0];
    for (size_t i = 1; i < all_latents.size(); ++i) {
        final_latent = sd::ops::concat(final_latent, all_latents[i], 2);
    }
    
    // 7. 解码视频
    auto video = sd_->decode_first_stage(final_latent, true);
    if (video.empty()) {
        LOG_ERROR("failed to decode video");
        return false;
    }
    
    // 8. 转换为图像帧
    // TODO: 实现视频到帧的转换
    
    return true;
}
```

---

### Task 4: 添加 CLI 参数支持

**文件**: `examples/cli/main.cpp`

**目标**: 添加 `--chunked` 参数支持

**步骤**:
1. 在局部变量区域添加参数（约 line 45）
2. 在参数解析循环中添加处理（约 line 65）
3. 在 VID_GEN 分支中添加分块生成分支（约 line 917-925）

**代码修改**:
```cpp
// 在局部变量区域添加（line 45 附近）
bool chunked_mode = false;
float chunk_duration = 5.0f;
float overlap_duration = 1.0f;

// 在参数解析循环中添加（line 138 附近）
} else if (strcmp(argv[i], "--chunked") == 0) {
    chunked_mode = true;
} else if (strcmp(argv[i], "--chunk-duration") == 0 && i + 1 < argc) {
    chunk_duration = atof(argv[++i]);
} else if (strcmp(argv[i], "--overlap") == 0 && i + 1 < argc) {
    overlap_duration = atof(argv[++i]);

// 在 VID_GEN 分支中添加（line 917-925）
} else if (cli_params.mode == VID_GEN) {
    if (chunked_mode && sd->version == VERSION_WAN2_MULTI_TALK) {
        // 使用 ChunkedVideoGenerator
        sd::ChunkedVideoGenerator::Params gen_params;
        gen_params.chunk_duration_sec = chunk_duration;
        gen_params.overlap_sec = overlap_duration;
        gen_params.fps = cli_params.preview_fps;
        gen_params.video_frames = gen_params.video_frames;
        
        sd::ChunkedVideoGenerator generator(sd.get(), gen_params);
        sd_image_t** chunked_frames = nullptr;
        int num_frames = 0;
        int fps = 0;
        
        if (generator.generate(gen_params.init_image,
                               audio_data, audio_samples,
                               &chunked_frames, &num_frames, &fps)) {
            results.adopt(chunked_frames, num_frames);
            cli_params.preview_fps = fps;
        }
    } else {
        // 原有逻辑
        sd_vid_gen_params_t vid_gen_params = gen_params.to_sd_vid_gen_params_t();
        sd_image_t* generated_video = nullptr;
        if (!generate_video(sd_ctx.get(), &vid_gen_params, &generated_video, 
                           &num_results, &generated_audio, &cli_params.preview_fps)) {
            generated_video = nullptr;
        }
        gen_params.fps = cli_params.preview_fps;
        results.adopt(generated_video, num_results);
    }
}
```

---

### Task 5: 创建端到端测试脚本

**文件**: `scripts/test_infini_talk.sh`

```bash
#!/bin/bash
set -e

MODEL="${1:-path/to/Wan2_1-InfiniteTalk_Single_Q8.gguf}"
AUDIO_ENCODER="${2:-path/to/wav2vec2_chinese_fp16.safetensors}"
IMAGE="${3:-test_data/reference.jpg}"
AUDIO="${4:-test_data/speech.wav}"
OUTPUT_DIR="${5:-test_output}"

mkdir -p "$OUTPUT_DIR"

echo "=== Testing Standard S2V ==="
./build/bin/sd-cli.exe -M vid_gen \
    --diffusion-model "$MODEL" \
    --audio-encoder "$AUDIO_ENCODER" \
    -i "$IMAGE" \
    --audio "$AUDIO" \
    --video-frames 81 \
    -o "$OUTPUT_DIR/s2v_test.mp4" \
    -d 0

test -s "$OUTPUT_DIR/s2v_test.mp4" && echo "✓ S2V: PASS" || echo "✗ S2V: FAIL"

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
    -o "$OUTPUT_DIR/multitalk_test.mp4" \
    -d 0

test -s "$OUTPUT_DIR/multitalk_test.mp4" && echo "✓ MultiTalk: PASS" || echo "✗ MultiTalk: FAIL"
```

---

## Tests / Validation

### 单元测试（已有）

```bash
cd D:/workspace/stable-diffusion.cpp
bash scripts/run_audio_chunker_test.sh
# 预期：All audio chunker tests passed!
```

### 编译验证

```bash
cd D:/workspace/stable-diffusion.cpp
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
# 预期：Build succeeded
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

# 测试 MultiTalk 分块生成
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
| 分块接缝 visible | 高 | 高 | 实现重叠混合 + 过渡平滑 |
| VRAM OOM | 中 | 高 | 流式处理，及时释放 |
| 口型不同步 | 低 | 高 | 精确时间戳对齐 |
| 音频采样率不匹配 | 中 | 中 | 统一重采样到 16kHz |

### 待解决问题

1. **重叠混合策略**: 线性混合还是扩散重采样？
2. **性能优化**: 是否支持 LightX2V LoRA 加速？
3. **内存管理**: 大量块时如何避免内存累积？
4. **音频输入格式**: 是否需要支持 WAVE/MP3 多种格式？

---

**计划文件**: `.hermes/plans/2026-09-30_091500-infini-talk-completion-v2.md`
