# 计划：实现 InfiniTalk 完整工作流 - 完成 process_chunk() 推理逻辑

## Goal

实现 `ChunkedVideoGenerator::process_chunk()` 方法，使图片+音频能生成数字人口型视频。

---

## Current Context / Assumptions

### 已完成的工作

| 项目 | 状态 | 说明 |
|------|------|------|
| `VERSION_WAN2_MULTI_TALK` 枚举 | ✅ | `src/model.h:138` |
| 版本检测逻辑 | ✅ | `src/model_loader.cpp` |
| 模型构建器支持 | ✅ | `src/pipeline/model_builders.cpp` |
| Wan 模型描述 | ✅ | `src/model/diffusion/wan.hpp` |
| 视频管道条件处理 | ✅ | `src/pipeline/video.cpp:1081-1128` |
| `AudioChunker` 类 | ✅ | `src/audio/audio_chunker.{h,cpp}` |
| `ChunkedVideoGenerator` 框架 | ✅ | `src/pipeline/video_chunked.{h,cpp}` |
| `generate()` 方法 | ✅ | 编码/分块/解码流程已实现 |
| CLI 参数支持 | ✅ | `--chunked`, `--chunk-duration`, `--overlap` |

### 当前问题

**`process_chunk()` 是空 stub**，返回 `std::nullopt`，导致视频生成失败。

```cpp
// src/pipeline/video_chunked.cpp:167
LOG_WARN("process_chunk: simplified implementation - need to implement full latent processing");
return std::nullopt;
```

### 关键发现

1. **音频嵌入方法已存在**: `src/pipeline/diffusion_engine.cpp:1895`
   ```cpp
   sd::Tensor<float> StableDiffusionGGML::get_audio_embedding(const sd_audio_t& audio)
   ```

2. **S2V 条件处理逻辑在 video.cpp:1081-1128**
   - 编码参考图片
   - 计算音频 embedding
   - 构建 audio window

3. **采样方法在 diffusion_engine.h:430**
   ```cpp
   sd::Tensor<float> sample(const std::shared_ptr<DiffusionModelRunner>& work_diffusion_model, ...);
   ```

---

## Architecture / Proposed Approach

**当前架构**:
```
src/pipeline/video_chunked.cpp
  └── generate()           # ✅ 已实现
       ├── 编码参考图片    # ✅
       ├── 音频预处理      # ✅
       ├── 音频分块        # ✅
       └── process_chunk() # ❌ stub
            └── generate_video() # 调用错误！应直接调用 sampling
```

**正确架构**:
```
process_chunk()
  ├── 1. 计算音频 embedding (调用 get_audio_embedding)
  ├── 2. 构建 audio window (复用 build_s2v_audio_window 逻辑)
  ├── 3. 构建 latents (ref_latents + s2v_audio_embed)
  ├── 4. 构建 embeds (条件编码)
  └── 5. 调用 sd_->sample() 进行去噪
```

---

## Step-by-Step Tasks

### Task 1: 导出 build_s2v_audio_window 为可复用函数

**文件**: `src/pipeline/video.cpp`

**目标**: 将 `build_s2v_audio_window()` 从 static 函数改为可复用函数

**步骤**:
1. 在 `src/pipeline/generation.h` 中添加声明
2. 在 `src/pipeline/video.cpp` 中修改实现

**代码修改**:
```cpp
// src/pipeline/generation.h
namespace sd::pipeline {
    // ... existing code ...
    
    // Build S2V/MultiTalk audio window for conditioning
    sd::Tensor<float> build_s2v_audio_window(const sd::Tensor<float>& stacked, int64_t batch_frames);
}
```

**预期输出**: 编译通过。

---

### Task 2: 添加部分音频嵌入方法

**文件**: `src/pipeline/diffusion_engine.h`

**添加声明**:
```cpp
// 支持部分音频嵌入（用于分块生成）
sd::Tensor<float> get_audio_embedding_partial(
    const float* audio_data,
    int64_t sample_count,
    int sample_rate);
```

**文件**: `src/pipeline/diffusion_engine.cpp`

**添加实现**:
```cpp
sd::Tensor<float> StableDiffusionGGML::get_audio_embedding_partial(
    const float* audio_data,
    int64_t sample_count,
    int sample_rate) {
    
    if (audio_encoder == nullptr) {
        LOG_ERROR("audio encoder model is not loaded");
        return {};
    }
    if (audio_data == nullptr || sample_count == 0 || sample_rate == 0) {
        LOG_ERROR("invalid audio input");
        return {};
    }
    
    // 重采样到 16kHz
    auto mono = sd::audio::resample_audio(audio_data, sample_count, sample_rate, 16000);
    if (mono.empty()) {
        LOG_ERROR("audio resample failed");
        return {};
    }
    
    return audio_encoder->compute(n_threads, mono);
}
```

---

### Task 3: 实现 process_chunk() 完整逻辑

**文件**: `src/pipeline/video_chunked.cpp`

**目标**: 实现单个音频块的推理逻辑

**关键步骤**:
1. 计算音频 embedding
2. 构建 audio window
3. 构建 latents
4. 构建 embeds（条件编码）
5. 调用采样

**代码实现**:
```cpp
std::optional<sd::Tensor<float>> ChunkedVideoGenerator::process_chunk(
    const AudioChunk& chunk,
    const sd::Tensor<float>& ref_latent) {
    
    // 1. 计算音频 embedding
    sd_audio_t audio_data{};
    audio_data.sample_rate = 16000;
    audio_data.channels = 1;
    audio_data.sample_count = chunk.duration_samples;
    audio_data.data = const_cast<float*>(chunk.data);
    
    auto stacked = sd_->get_audio_embedding_partial(chunk.data, chunk.duration_samples, 16000);
    if (stacked.empty()) {
        LOG_ERROR("failed to compute audio embedding for chunk");
        return std::nullopt;
    }
    
    // 2. 构建 audio window
    int64_t latent_t = sd_->video_frames_to_latent_frames(params_.video_frames);
    int64_t batch_frames = latent_t * 4;
    auto audio_window = pipeline::build_s2v_audio_window(stacked, batch_frames);
    if (audio_window.empty()) {
        LOG_ERROR("failed to build s2v audio window");
        return std::nullopt;
    }
    
    // 3. 构建 latents
    ImageGenerationLatents latents;
    latents.ref_latents.push_back(ref_latent);
    latents.s2v_audio_embed = audio_window;
    latents.init_latent = sd_->generate_init_latent(params_.width, params_.height, params_.video_frames, true);
    if (latents.init_latent.empty()) {
        LOG_ERROR("failed to generate init latent");
        return std::nullopt;
    }
    
    // 4. 构建 embeds
    GenerationRequest req(sd_, params_.width, params_.height, params_.video_frames, params_.fps);
    // 需要实现条件编码逻辑...
    
    // 5. 调用采样
    auto result = sd_->sample(
        sd_->diffusion_model,
        true,
        latents.init_latent,
        sd::Tensor<float>::randn_like(latents.init_latent, sd_->rng),
        embeds.cond,
        SDCondition(),  // 简化：不使用 uncond
        embeds.img_uncond,
        sd::Tensor<float>(),
        0.f,
        sd_sample_params_t{},  // 使用默认参数
        0.f,  // eta
        0,    // shifted_timestep
        EULER_SAMPLE_METHOD,
        sd_->is_flow_denoiser(),
        nullptr,
        std::vector<float>{},  // sigmas
        latents.ref_latents,
        RefImageParams(),
        sd::Tensor<float>(),
        sd::Tensor<float>(),
        0.f,
        latents.audio_length,
        static_cast<float>(params_.fps),
        nullptr,
        true,
        latents.video_positions);
    
    return result;
}
```

---

### Task 4: 简化实现 - 直接调用 generate_video

**替代方案**: 如果完整实现太复杂，可以先使用简化方案：

**文件**: `src/pipeline/video_chunked.cpp`

**简化实现**:
```cpp
std::optional<sd::Tensor<float>> ChunkedVideoGenerator::process_chunk(
    const AudioChunk& chunk,
    const sd::Tensor<float>& ref_latent) {
    
    // 方案：直接调用 generate_video 并提取 latent
    // 注意：这需要修改 generate_video 返回值或添加新方法
    
    // 临时方案：返回 ref_latent，让 generate() 跳过此块
    // 等待完整实现
    
    LOG_WARN("process_chunk: using temporary fallback - returning ref_latent only");
    return ref_latent;
}
```

---

## Tests / Validation

### 单元测试

```bash
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
| 分块接缝 visible | 高 | 高 | 实现重叠混合 |
| VRAM OOM | 中 | 高 | 流式处理，及时释放 |
| 口型不同步 | 低 | 高 | 精确时间戳对齐 |
| 音频采样率不匹配 | 中 | 中 | 统一重采样到 16kHz |

### 待解决问题

1. **条件编码**: 是否需要文本 prompt？
2. **未条件化**: 是否需要 uncond 分支？
3. **采样参数**: 应该使用什么 guidance/sigma？
4. **重叠混合**: 如何实现平滑过渡？

---

## 实施建议

### 方案 A: 完整实现（推荐）

实现完整的 `process_chunk()` 逻辑，包括：
- 音频 embedding 计算
- Audio window 构建
- 条件编码
- 采样调用

**优点**: 功能完整，可生产使用
**缺点**: 需要更多代码和测试

### 方案 B: 简化实现（快速验证）

使用简化逻辑，先验证流程是否通：
- 跳过条件编码
- 直接返回 ref_latent
- 验证分块拼接逻辑

**优点**: 快速验证架构
**缺点**: 生成的视频可能不正确

---

**建议**: 先实施**方案 B** 验证架构，再逐步完善到**方案 A**。

**计划文件**: `.hermes/plans/2026-09-30_091500-process-chunk-implementation.md`
