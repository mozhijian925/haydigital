# 计划：实现 InfiniTalk 完整工作流 - 完成 process_chunk() 推理逻辑

## Goal

实现 `ChunkedVideoGenerator::process_chunk()` 方法，使图片+音频能生成数字人口型视频。

---

## Current Context / Assumptions

### 已完成的工作

| 项目 | 状态 | 文件位置 |
|------|------|----------|
| `VERSION_WAN2_MULTI_TALK` 枚举 | ✅ | `src/model.h:138` |
| 版本检测逻辑 | ✅ | `src/model_loader.cpp` |
| 模型构建器支持 | ✅ | `src/pipeline/model_builders.cpp:255` |
| Wan 模型描述 | ✅ | `src/model/diffusion/wan.hpp:933` |
| 视频管道条件处理 | ✅ | `src/pipeline/video.cpp:1081-1128` |
| `AudioChunker` 类 | ✅ | `src/audio/audio_chunker.{h,cpp}` |
| `ChunkedVideoGenerator` 框架 | ✅ | `src/pipeline/video_chunked.{h,cpp}` |
| `generate()` 方法 | ✅ | `src/pipeline/video_chunked.cpp:21-121` |
| CLI 参数支持 | ✅ | `examples/cli/main.cpp` |

### 当前问题

**`process_chunk()` 是空 stub**，返回 `std::nullopt`，导致视频生成失败。

```cpp
// src/pipeline/video_chunked.cpp:167 (旧版)
LOG_WARN("process_chunk: simplified implementation - need to implement full latent processing");
return std::nullopt;
```

### 关键依赖分析

| 组件 | 来源 | 签名 | 状态 |
|------|------|------|------|
| 音频嵌入 | `diffusion_engine.cpp:1895` | `sd::Tensor<float> get_audio_embedding(const sd_audio_t& audio)` | ✅ 已存在 |
| Audio Window 构建 | `video.cpp:426` | `static sd::Tensor<float> build_s2v_audio_window(...)` | ⚠️ static，需导出 |
| Latents 准备 | `video.cpp:464` | `static std::optional<ImageGenerationLatents> prepare_video_generation_latents(...)` | ⚠️ static，需导出 |
| Embeds 准备 | `video.cpp:1148` | `static std::optional<ImageGenerationEmbeds> prepare_video_generation_embeds(...)` | ⚠️ static，需导出 |
| 采样 | `diffusion_engine.h:430` | `sd::Tensor<float> sample(...)` | ✅ 已存在 |
| 生成初始 latent | `diffusion_engine.cpp:2821` | `sd::Tensor<float> generate_init_latent(int width, int height, int frames, bool video)` | ✅ 已存在 |

---

## Architecture / Proposed Approach

**正确实现路径**:

```
process_chunk()
  ├── 1. 构建 sd_vid_gen_params_t（配置音频、尺寸等）
  ├── 2. 构建 GenerationRequest（从 params_ 初始化）
  ├── 3. 调用 prepare_video_generation_latents()（已导出）
  ├── 4. 替换 ref_latents（使用传入的参考图片 latent）
  ├── 5. 计算音频 embedding → get_audio_embedding()
  ├── 6. 构建 audio window → build_s2v_audio_window()
  ├── 7. 设置 latents.s2v_audio_embed
  ├── 8. 构建条件 → prepare_video_generation_embeds()
  ├── 9. 设置 embeds.cond.c_ref_images 和 embeds.cond.c_ref_audios
  ├── 10. 初始化噪声
  └── 11. 调用 sd_->sample() 进行去噪推理
```

---

## Step-by-Step Tasks

### Task 1: 导出 build_s2v_audio_window

**文件**: `src/pipeline/generation.h`

**添加声明**:
```cpp
namespace sd::pipeline {
    // ... existing code ...
    
    // Build S2V/MultiTalk audio window for conditioning
    sd::Tensor<float> build_s2v_audio_window(const sd::Tensor<float>& stacked, int64_t batch_frames);
}
```

**文件**: `src/pipeline/video.cpp`

**修改 (line 426)**:
```cpp
// 从:
static sd::Tensor<float> build_s2v_audio_window(const sd::Tensor<float>& stacked, int64_t batch_frames) {

// 改为:
sd::Tensor<float> build_s2v_audio_window(const sd::Tensor<float>& stacked, int64_t batch_frames) {
```

**验证**:
```bash
cd D:/workspace/stable-diffusion.cpp
grep -n "build_s2v_audio_window" src/pipeline/generation.h
# 预期: 新函数声明
grep -n "^sd::Tensor<float> build_s2v_audio_window" src/pipeline/video.cpp
# 预期: static 已移除
```

---

### Task 2: 导出 prepare_video_generation_latents

**文件**: `src/pipeline/generation.h`

**添加声明**:
```cpp
namespace sd::pipeline {
    // ... existing code ...
    
    // Prepare video generation latents (S2V/MultiTalk support)
    std::optional<ImageGenerationLatents> prepare_video_generation_latents(
        StableDiffusionGGML* sd,
        const sd_vid_gen_params_t* sd_vid_gen_params,
        GenerationRequest* request);
}
```

**文件**: `src/pipeline/video.cpp`

**修改 (line 464)**:
```cpp
// 从:
static std::optional<ImageGenerationLatents> prepare_video_generation_latents(...)

// 改为:
std::optional<ImageGenerationLatents> prepare_video_generation_latents(...)
```

---

### Task 3: 导出 prepare_video_generation_embeds

**文件**: `src/pipeline/generation.h`

**添加声明**:
```cpp
namespace sd::pipeline {
    // ... existing code ...
    
    // Prepare video generation embeddings (S2V/MultiTalk support)
    std::optional<ImageGenerationEmbeds> prepare_video_generation_embeds(
        StableDiffusionGGML* sd,
        const sd_vid_gen_params_t* sd_vid_gen_params,
        const GenerationRequest& request,
        const ImageGenerationLatents& latents);
}
```

**文件**: `src/pipeline/video.cpp`

**修改 (line 1148)**:
```cpp
// 从:
static std::optional<ImageGenerationEmbeds> prepare_video_generation_embeds(...)

// 改为:
std::optional<ImageGenerationEmbeds> prepare_video_generation_embeds(...)
```

---

### Task 4: 更新 video_chunked.cpp includes

**文件**: `src/pipeline/video_chunked.cpp`

**添加 include**:
```cpp
#include "model/diffusion/model.hpp"  // 用于 RefImageParams
```

---

### Task 5: 实现 process_chunk() 完整逻辑

**文件**: `src/pipeline/video_chunked.cpp`

**完整实现**:
```cpp
std::optional<sd::Tensor<float>> ChunkedVideoGenerator::process_chunk(
    const AudioChunk& chunk,
    const sd::Tensor<float>& ref_latent) {
    
    LOG_INFO("Processing audio chunk: %lld samples @ 16kHz", (long long)chunk.duration_samples);
    
    // 1. 构建 Audio 数据
    sd_audio_t audio_data{};
    audio_data.sample_rate = 16000;
    audio_data.channels = 1;
    audio_data.sample_count = chunk.duration_samples;
    audio_data.data = const_cast<float*>(chunk.data);
    
    // 2. 构建 sd_vid_gen_params_t
    sd_vid_gen_params_t vid_params{};
    sd_vid_gen_params_init(&vid_params);
    vid_params.init_image = sd_image_t{0, 0, 0, nullptr};  // 不使用 init_image
    vid_params.ref_audios = &audio_data;
    vid_params.ref_audios_count = 1;
    vid_params.width = params_.width;
    vid_params.height = params_.height;
    vid_params.video_frames = params_.video_frames;
    vid_params.fps = params_.fps;
    vid_params.sample_params = sd_sample_params_t{};
    
    // 3. 构建 GenerationRequest
    GenerationRequest req(sd_, &vid_params);
    
    // 4. 准备 latents
    auto latents_opt = pipeline::prepare_video_generation_latents(sd_, &vid_params, &req);
    if (!latents_opt.has_value()) {
        LOG_ERROR("failed to prepare video generation latents");
        return std::nullopt;
    }
    auto latents = std::move(*latents_opt);
    
    // 5. 替换 ref_latent（使用传入的参考图片 latent）
    latents.ref_latents.clear();
    latents.ref_latents.push_back(ref_latent);
    
    // 6. 计算音频 embedding
    auto stacked = sd_->get_audio_embedding(audio_data);
    if (stacked.empty()) {
        LOG_ERROR("failed to compute audio embedding");
        return std::nullopt;
    }
    
    // 7. 构建 audio window
    int64_t latent_t = sd_->video_frames_to_latent_frames(params_.video_frames);
    int64_t batch_frames = latent_t * 4;
    auto audio_window = pipeline::build_s2v_audio_window(stacked, batch_frames);
    if (audio_window.empty()) {
        LOG_ERROR("failed to build s2v audio window");
        return std::nullopt;
    }
    latents.s2v_audio_embed = audio_window;
    
    // 8. 准备 embeds（条件编码）
    auto embeds_opt = pipeline::prepare_video_generation_embeds(sd_, &vid_params, req, latents);
    if (!embeds_opt.has_value()) {
        LOG_ERROR("failed to prepare video generation embeds");
        return std::nullopt;
    }
    auto embeds = std::move(*embeds_opt);
    
    // 9. 设置 audio conditioning（S2V/MultiTalk）
    embeds.cond.c_ref_images = latents.ref_latents;
    if (!latents.s2v_audio_embed.empty()) {
        embeds.cond.c_ref_audios = {latents.s2v_audio_embed};
    }
    
    // 10. 初始化噪声
    sd::Tensor<float> noise = sd::Tensor<float>::randn_like(latents.init_latent, sd_->rng);
    
    // 11. 采样推理
    sd_sample_params_t default_params = vid_params.sample_params;
    auto result = sd_->sample(
        sd_->diffusion_model,
        true,
        latents.init_latent,
        std::move(noise),
        embeds.cond,
        SDCondition(),  // S2V 不需要 uncond
        SDCondition(),  // img_uncond
        sd::Tensor<float>(),  // control_image
        0.f,  // control_strength
        default_params.guidance,
        default_params.eta,
        default_params.shifted_timestep,
        default_params.sample_method,
        sd_->is_flow_denoiser(),
        nullptr,  // extra_sample_args
        std::vector<float>{},  // sigmas
        std::vector<sd::Tensor<float>>{},  // ref_latents（已通过 embeds 传递）
        RefImageParams(),
        latents.denoise_mask,
        latents.vace_context,
        req.vace_strength,
        latents.audio_length,
        static_cast<float>(req.fps),
        req.cache_params,
        true,  // high_noise_sample_steps <= 0
        latents.video_positions);
    
    if (result.empty()) {
        LOG_ERROR("sampling failed");
        return std::nullopt;
    }
    
    LOG_INFO("Sampling completed for chunk");
    return result;
}
```

---

## Tests / Validation

### 编译验证

```bash
cd D:/workspace/stable-diffusion.cpp
cmake --build build --target sd-cli
# 预期：编译成功，无错误
```

### 单元测试

```bash
cd D:/workspace/stable-diffusion.cpp
bash scripts/run_audio_chunker_test.sh
# 预期：All audio chunker tests passed! (6/6)
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
| 分块接缝 visible | 高 | 高 | 实现重叠混合（blend_chunks） |
| VRAM OOM | 中 | 高 | 及时释放中间张量 |
| 音频时长不对齐 | 中 | 中 | 使用 latent 帧数对齐 |

### 待解决问题

1. **未条件化分支**: S2V/MultiTalk 是否需要 uncond？
   - 决策：不使用（参考 video.cpp:1189-1210，`use_uncond` 默认为 false）

2. **采样参数默认值**: 使用什么采样步骤和 guidance？
   - 决策：使用默认值（`sample_params_t{}`）

3. **重叠混合**: 何时实现？
   - 决策：本次先跳过，后续优化

---

## 实施顺序

1. **Task 1-3**: 导出函数（并行完成）
2. **Task 4**: 更新 includes
3. **Task 5**: 实现 process_chunk()（核心）
4. **验证**: 编译 + 单元测试 + 集成测试

---

**计划文件**: `.hermes/plans/2026-09-30_091500-process-chunk-implementation-v2.md`
