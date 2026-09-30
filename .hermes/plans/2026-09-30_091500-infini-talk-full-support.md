# 技术方案：完整支持 InfiniTetalk 无限时长数字人口型视频生成

## Goal

在 stable-diffusion.cpp 源码上进行二次开发，完整实现 InfiniTetalk 工作流，支持任意时长音频驱动的图片数字人口型同步视频生成（无 81 帧限制）。

---

## Current Context / Assumptions

### 现有架构分析

```
当前 S2V 实现（有长度限制）:
┌─────────────────────────────────────────────────────────────────┐
│  input: audio.wav (任意时长)                                     │
│       ↓                                                         │
│  get_audio_embedding() → wav2vec2 → [dim, frames@50Hz, layers]  │
│       ↓                                                         │
│  build_s2v_audio_window()                                       │
│    → build_audio_buckets() → [dim, batch_frames@16fps, layers]   │
│       ↓                                                         │
│  限制: batch_frames = latent_t * 4 = 81 * 4 = 324 帧            │
│       仅支持固定长度视频生成                                     │
└─────────────────────────────────────────────────────────────────┘

需要实现: 滑动窗口 + 缓存机制，支持任意时长
```

### 关键限制分析

| 限制来源 | 当前值 | 影响 |
|---------|--------|------|
| 视频帧数对齐 | `align_video_frames()` → 81 帧 | 固定输出长度 |
| 音频窗口构建 | `batch_frames = latent_t * 4` | 与视频长度绑定 |
| 推理速度 | ~30s/81帧 (RTX 4090) | 长视频耗时过长 |
| VRAM 占用 | ~28GB (14B Q8_0) | 需分块推理 |

### 假设

1. **InfiniteTalk 模型架构**：基于 Wan2.1 S2V，支持 causal 音频编码
2. **核心需求**：长音频 → 分块处理 → 拼接视频
3. **口型同步**：保持音频驱动的精确口型对齐
4. **视觉连续性**：相邻窗口间保持一致性（人物外观、背景）

---

## Architecture / Proposed Approach

**方案**: 分块推理 + 滑动窗口 + 上下文缓存

```
┌──────────────────────────────────────────────────────────────────────┐
│                    InfiniTetalk 无限时长生成架构                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入音频 (任意时长)                                                   │
│       ↓                                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Chunk 1: [0s - 5s]     Chunk 2: [5s - 10s]    Chunk 3: ...   │   │
│  │  → 推理 81 帧   →  推理 81 帧   →  推理 81 帧                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│       ↓              ↓                  ↓                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  重叠区域处理 (Overlap Handling)                               │   │
│  │  - 前 10 帧使用上一窗口的 latent 作为初始状态                    │   │
│  │  - 过渡区域使用线性混合或 diffusion 重采样                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│       ↓                                                              │
│  输出视频 (任意时长，口型同步)                                          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**核心组件**:
1. **音频分块处理器** (`src/audio/audio_chunker.hpp`)：将长音频分割为固定长度块
2. **分块推理器** (`src/pipeline/video_chunked.cpp`)：逐块生成视频 latent
3. **重叠拼接器** (`src/pipeline/video_blend.cpp`)：处理窗口间的平滑过渡
4. **流式解码器** (`src/pipeline/video_stream.cpp`)：边生成边解码，降低内存占用

---

## Step-by-Step Tasks

### Task 1: 验证 InfiniteTalk 模型兼容性

**目标**: 确认模型是否可直接使用现有 S2V 管线

**操作**: 使用 Python 脚本检查模型权重

```python
# scripts/check_multitalk_model.py
import struct
import sys
from pathlib import Path

def read_gguf_metadata(path: str):
    """读取 GGUF 文件的元数据和张量信息"""
    tensors = []
    with open(path, 'rb') as f:
        # GGUF magic
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Not GGUF: {path}")
        
        # Read header
        version = struct.unpack('<I', f.read(4))[0]
        n_kv, n_tensors_count = struct.unpack('<QQ', f.read(16))
        
        # Read KV pairs (simplified)
        for _ in range(n_kv):
            n_chars = struct.unpack('<I', f.read(4))[0]
            key = f.read(n_chars).decode('utf-8')
            # Skip value...
            f.read(8)  # type
        
        # Read tensors
        for _ in range(n_tensors_count):
            n_chars = struct.unpack('<I', f.read(4))[0]
            name = f.read(n_chars).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            shape = list(struct.unpack('<' + 'Q' * n_dims, f.read(8 * n_dims)))
            tensors.append((name, shape))
    
    return tensors

def analyze_model(path):
    tensors = read_gguf_metadata(path)
    
    # 分类统计
    audio_keys = [t for t in tensors if any(k in t[0] for k in ['audio', 'wav2vec', 'injector', 'encoder'])]
    wan_keys = [t for t in tensors if 'model.diffusion_model' in t[0]]
    transformer_keys = [t for t in tensors if 'blocks.' in t[0]]
    
    print(f"Total tensors: {len(tensors)}")
    print(f"Wan model keys: {len(wan_keys)}")
    print(f"Audio-related keys: {len(audio_keys)}")
    print(f"Transformer blocks: {len(transformer_keys)}")
    
    # 检查关键组件
    has_causal_audio = any('casual_audio_encoder' in t[0] or 'causal_audio' in t[0] for t in tensors)
    has_audio_injector = any('audio_injector' in t[0] for t in tensors)
    has_wav2vec = any('wav2vec2' in t[0] for t in tensors)
    has_multi_talk = any('multi_talk' in t[0].lower() for t in tensors)
    
    print(f"\nComponent check:")
    print(f"  Causal audio encoder: {has_causal_audio}")
    print(f"  Audio injector: {has_audio_injector}")
    print(f"  Wav2Vec2: {has_wav2vec}")
    print(f"  MultiTalk specific: {has_multi_talk}")
    
    # 检查 layer 数量
    block_counts = {}
    for name, shape in transformers:
        if 'blocks.' in name:
            parts = name.split('.')
            for p in parts:
                if p.isdigit():
                    block_counts[int(p)] = block_counts.get(int(p), 0) + 1
                    break
    
    if block_counts:
        max_layer = max(block_counts.keys())
        print(f"\nTransformer layers: {max_layer + 1}")
    
    return {
        'has_causal_audio': has_causal_audio,
        'has_audio_injector': has_audio_injector,
        'has_wav2vec': has_wav2vec,
        'has_multi_talk': has_multi_talk,
        'total_tensors': len(tensors)
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_multitalk_model.py <model.gguf>")
        sys.exit(1)
    
    result = analyze_model(sys.argv[1])
    
    # 判断兼容性
    if result['has_causal_audio'] and result['has_audio_injector']:
        print("\n✓ 模型兼容 S2V 架构，可直接使用现有管线")
    elif result['has_multi_talk']:
        print("\n⚠ 模型有特殊组件，需要自定义处理")
    else:
        print("\n✗ 模型不兼容，需要全新实现")
```

**验证命令**:
```bash
cd D:/workspace/stable-diffusion.cpp
python scripts/check_multitalk_model.py "C:/path/to/Wan2_1-InfiniteTalk_Single_Q8.gguf"
```

**预期输出**:
```
Total tensors: 587
Wan model keys: 512
Audio-related keys: 45
Transformer blocks: 320

Component check:
  Causal audio encoder: True
  Audio injector: True
  Wav2Vec2: True
  MultiTalk specific: False

✓ 模型兼容 S2V 架构，可直接使用现有管线
```

---

### Task 2: 添加音频分块处理模块

**新建文件**: `src/audio/audio_chunker.hpp`

```cpp
#ifndef __SD_AUDIO_AUDIO_CHUNKER_HPP__
#define __SD_AUDIO_AUDIO_CHUNKER_HPP__

#include <cstdint>
#include <vector>
#include "core/tensor.hpp"

namespace sd::audio {

/**
 * 音频分块处理器
 * 
 * 将长音频分割为固定长度的块，用于分块视频生成
 */
struct AudioChunk {
    int64_t start_sample;      // 块起始采样点（全局）
    int64_t end_sample;        // 块结束采样点（全局）
    int64_t duration_samples;  // 块长度（采样点数）
    float sample_rate;         // 采样率
    int channels;              // 声道数
    const float* data;         // 音频数据指针（原始音频的切片）
};

class AudioChunker {
public:
    /**
     * 构造函数
     * @param sample_rate 音频采样率（Hz）
     * @param channels 声道数
     * @param chunk_duration_sec 每个块的时长（秒）
     * @param overlap_sec 重叠区域时长（秒），用于平滑过渡
     */
    AudioChunker(int sample_rate, int channels, float chunk_duration_sec = 5.0f, float overlap_sec = 1.0f);
    
    /**
     * 将长音频分割为固定长度的块
     * @param audio 完整音频数据
     * @param samples 总采样数
     * @return 音频块列表
     */
    std::vector<AudioChunk> chunk(const float* audio, int64_t samples);
    
    /**
     * 获取块参数
     */
    int chunk_samples() const { return chunk_samples_; }
    int overlap_samples() const { return overlap_samples_; }
    int step_samples() const { return step_samples_; }
    
private:
    int sample_rate_;
    int channels_;
    int chunk_samples_;
    int overlap_samples_;
    int step_samples_;  // chunk_samples - overlap_samples
};

}  // namespace sd::audio

#endif  // __SD_AUDIO_AUDIO_CHUNKER_HPP__
```

**新建文件**: `src/audio/audio_chunker.cpp`

```cpp
#include "audio/audio_chunker.hpp"

namespace sd::audio {

AudioChunker::AudioChunker(int sample_rate, int channels, float chunk_duration_sec, float overlap_sec)
    : sample_rate_(sample_rate),
      channels_(channels),
      chunk_samples_(static_cast<int>(sample_rate_ * chunk_duration_sec)),
      overlap_samples_(static_cast<int>(sample_rate_ * overlap_sec)) {
    // 确保 chunk 大小是 4 的倍数（VAE 缩放因子）
    chunk_samples_ = (chunk_samples_ / 4) * 4;
    overlap_samples_ = (overlap_samples_ / 4) * 4;
    step_samples_ = chunk_samples_ - overlap_samples_;
}

std::vector<AudioChunk> AudioChunker::chunk(const float* audio, int64_t samples) {
    std::vector<AudioChunk> chunks;
    
    if (samples <= 0 || audio == nullptr) {
        return chunks;
    }
    
    int64_t remaining = samples;
    int64_t offset = 0;
    
    while (offset < samples) {
        AudioChunk chunk;
        chunk.start_sample = offset;
        chunk.duration_samples = std::min<int64_t>(chunk_samples_, samples - offset);
        chunk.end_sample = offset + chunk.duration_samples;
        chunk.sample_rate = sample_rate_;
        chunk.channels = channels_;
        chunk.data = &audio[offset * channels_];
        
        chunks.push_back(chunk);
        
        // 移动步长（考虑重叠）
        offset += step_samples_;
        
        // 如果是最后一个块且不足一个 chunk，直接添加
        if (offset >= samples) {
            break;
        }
    }
    
    return chunks;
}

}  // namespace sd::audio
```

**修改 CMakeLists.txt** 添加新源文件：

**文件**: `CMakeLists.txt`

在 `src/audio/` 相关的源文件列表中添加强制：

```cmake
# 查找音频相关源文件
file(GLOB AUDIO_SOURCES "src/audio/*.cpp")
list(APPEND SD_CLI_SOURCES ${AUDIO_SOURCES})
```

**验证**:
```bash
cd D:/workspace/stable-diffusion.cpp
cmake --build build --config Release -j8 2>&1 | grep -E "error:|AudioChunker"
# 预期：无错误，显示 AudioChunker 编译信息
```

---

### Task 3: 实现分块视频生成逻辑

**新建文件**: `src/pipeline/video_chunked.hpp`

```cpp
#ifndef __SD_PIPELINE_VIDEO_CHUNKED_HPP__
#define __SD_PIPELINE_VIDEO_CHUNKED_HPP__

#include <memory>
#include <vector>
#include "stable-diffusion.h"
#include "audio/audio_chunker.hpp"

namespace sd {

/**
 * 分块视频生成器
 * 
 * 支持长音频驱动的数字人口型视频生成
 */
class ChunkedVideoGenerator {
public:
    struct ChunkedVideoParams {
        float chunk_duration_sec = 5.0f;   // 每块时长
        float overlap_sec = 1.0f;          // 重叠区域
        int fps = 16;                      // 输出帧率
        int video_frames = 81;             // 每块帧数
        bool blend_overlaps = true;        // 是否混合重叠区域
        float blend_strength = 0.5f;       // 混合强度
    };
    
    ChunkedVideoGenerator(StableDiffusionGGML* sd, const ChunkedVideoParams& params);
    ~ChunkedVideoGenerator();
    
    /**
     * 生成分块视频
     * @param ref_image 参考图片
     * @param audio 驱动音频
     * @param audio_sample_rate 音频采样率
     * @param audio_channels 音频声道数
     * @param output_frames 输出帧数组（由调用者管理）
     * @param output_fps 输出帧率
     * @return 成功返回 true
     */
    bool generate_chunked_video(
        const sd_image_t& ref_image,
        const float* audio,
        int64_t audio_samples,
        int audio_sample_rate,
        int audio_channels,
        sd_image_t*** output_frames,
        int* num_frames_out,
        int* fps_out
    );
    
    /**
     * 获取生成进度（0.0 - 1.0）
     */
    float get_progress() const;
    
private:
    StableDiffusionGGML* sd_;
    ChunkedVideoParams params_;
    AudioChunker chunker_;
    
    std::vector<sd::Tensor<float>> processed_chunks_;
    
    /**
     * 处理单个音频块
     */
    std::optional<sd::Tensor<float>> process_audio_chunk(
        const AudioChunk& chunk,
        const sd::Tensor<float>& ref_latent,
        const std::vector<sd::Tensor<float>>* prev_latents = nullptr
    );
    
    /**
     * 混合两个相邻块的过渡区域
     */
    sd::Tensor<float> blend_chunks(
        const sd::Tensor<float>& chunk_a,
        const sd::Tensor<float>& chunk_b,
        int blend_frames
    );
    
    /**
     * 保存中间状态用于下一块推理
     */
    void save_context(const sd::Tensor<float>& latent, int chunk_index);
};

}  // namespace sd

#endif  // __SD_PIPELINE_VIDEO_CHUNKED_HPP__
```

**新建文件**: `src/pipeline/video_chunked.cpp`

```cpp
#include "pipeline/video_chunked.hpp"
#include "pipeline/diffusion_engine.h"
#include "pipeline/video.h"
#include "core/util.h"

namespace sd {

ChunkedVideoGenerator::ChunkedVideoGenerator(
    StableDiffusionGGML* sd,
    const ChunkedVideoParams& params)
    : sd_(sd),
      params_(params),
      chunker_(16000, 1, params.chunk_duration_sec, params.overlap_sec) {  // wav2vec2 使用 16kHz
}

ChunkedVideoGenerator::~ChunkedVideoGenerator() = default;

bool ChunkedVideoGenerator::generate_chunked_video(
    const sd_image_t& ref_image,
    const float* audio,
    int64_t audio_samples,
    int audio_sample_rate,
    int audio_channels,
    sd_image_t*** output_frames,
    int* num_frames_out,
    int* fps_out) {
    
    if (sd_ == nullptr || audio == nullptr || audio_samples <= 0) {
        LOG_ERROR("Invalid input for chunked video generation");
        return false;
    }
    
    LOG_INFO("Starting chunked video generation");
    LOG_INFO("  Audio: %lld samples @ %d Hz, %d channels", 
             (long long)audio_samples, audio_sample_rate, audio_channels);
    LOG_INFO("  Chunk duration: %.2fs, Overlap: %.2fs", 
             params_.chunk_duration_sec, params_.overlap_sec);
    
    // 1. 准备参考图片 latent
    auto ref_tensor = sd_image_to_tensor(ref_image);
    ref_tensor = ensure_image_tensor_channels(ref_tensor, sd_->get_image_channels());
    auto ref_latent = sd_->encode_first_stage(ref_tensor);
    if (ref_latent.empty()) {
        LOG_ERROR("Failed to encode reference image");
        return false;
    }
    LOG_INFO("Reference image encoded: %dx%dx%dx%d",
             (int)ref_latent.shape()[0], (int)ref_latent.shape()[1],
             (int)ref_latent.shape()[2], (int)ref_latent.shape()[3]);
    
    // 2. 音频分块
    // 注意：需要将音频重采样到 16kHz（wav2vec2 要求）
    auto mono = sd::audio::downmix_to_mono(audio, audio_samples, audio_channels);
    if (mono.empty()) {
        LOG_ERROR("Audio mono downmix failed");
        return false;
    }
    mono = sd::audio::resample_audio(mono.data(), mono.size(), audio_sample_rate, 16000);
    if (mono.empty()) {
        LOG_ERROR("Audio resample to 16kHz failed");
        return false;
    }
    
    auto chunks = chunker_.chunk(mono.data(), mono.size());
    LOG_INFO("Audio split into %zu chunks", chunks.size());
    
    if (chunks.empty()) {
        LOG_ERROR("No audio chunks generated");
        return false;
    }
    
    // 3. 逐块生成视频 latent
    processed_chunks_.clear();
    std::vector<sd::Tensor<float>> all_latents;
    
    for (size_t i = 0; i < chunks.size(); ++i) {
        LOG_INFO("Processing chunk %zu/%zu (%.2fs - %.2fs)",
                 i + 1, chunks.size(),
                 chunks[i].start_sample / 16000.0,
                 chunks[i].end_sample / 16000.0);
        
        // 准备该块的音频 embedding
        // 注意：这里需要修改 get_audio_embedding 以支持部分音频
        auto audio_embed = sd_->get_audio_embedding_partial(
            chunks[i].data, 
            chunks[i].duration_samples,
            16000
        );
        if (audio_embed.empty()) {
            LOG_ERROR("Failed to encode audio chunk %zu", i);
            return false;
        }
        
        // 构建音频窗口
        int64_t batch_frames = params_.video_frames * 4;  // VAE scale factor
        auto audio_window = build_s2v_audio_window(audio_embed, batch_frames);
        if (audio_window.empty()) {
            LOG_ERROR("Failed to build audio window for chunk %zu", i);
            return false;
        }
        
        // 调用现有推理逻辑
        auto latent = process_audio_chunk(chunks[i], ref_latent, 
                                          i > 0 ? &processed_chunks_ : nullptr);
        if (latent.has_value()) {
            processed_chunks_.push_back(latent.value());
            
            // 处理重叠区域
            if (i > 0 && params_.blend_overlaps && !processed_chunks_.empty()) {
                auto& prev = processed_chunks_[processed_chunks_.size() - 2];
                auto blended = blend_chunks(prev, latent.value(), params_.overlap_samples / 4);
                processed_chunks_.back() = std::move(blended);
            }
            
            // 累积所有 latent
            if (i == 0) {
                all_latents.push_back(processed_chunks_.back());
            } else {
                // 只添加非重叠部分
                int keep_frames = params_.video_frames - params_.overlap_samples / 4;
                if (keep_frames > 0) {
                    auto chunk = processed_chunks_.back();
                    chunk = sd::ops::slice(chunk, 2, 0, keep_frames);
                    all_latents.push_back(chunk);
                }
            }
        }
        
        LOG_INFO("Chunk %zu completed, progress: %.1f%%",
                 i + 1, (float)(i + 1) / chunks.size() * 100);
    }
    
    // 4. 拼接所有 latent
    if (all_latents.empty()) {
        LOG_ERROR("No video latents generated");
        return false;
    }
    
    sd::Tensor<float> final_latent = all_latents[0];
    for (size_t i = 1; i < all_latents.size(); ++i) {
        final_latent = sd::ops::concat(final_latent, all_latents[i], 2);
    }
    
    LOG_INFO("Final latent shape: %dx%dx%dx%d",
             (int)final_latent.shape()[0], (int)final_latent.shape()[1],
             (int)final_latent.shape()[2], (int)final_latent.shape()[3]);
    
    // 5. 解码视频
    auto video = sd_->decode_first_stage(final_latent, true);
    if (video.empty()) {
        LOG_ERROR("Failed to decode video");
        return false;
    }
    
    // 6. 转换为图像帧
    int total_frames = video.shape()[2];
    *output_frames = new sd_image_t*[total_frames];
    *num_frames_out = total_frames;
    *fps_out = params_.fps;
    
    for (int f = 0; f < total_frames; ++f) {
        (*output_frames)[f] = new sd_image_t;
        *(*output_frames)[f] = tensor_to_sd_image(video, f);
    }
    
    LOG_INFO("Chunked video generation completed: %d frames @ %d fps", total_frames, params_.fps);
    return true;
}

std::optional<sd::Tensor<float>> ChunkedVideoGenerator::process_audio_chunk(
    const AudioChunk& chunk,
    const sd::Tensor<float>& ref_latent,
    const std::vector<sd::Tensor<float>>* prev_latents) {
    
    // TODO: 调用现有的采样逻辑
    // 需要提取 video.cpp 中的核心采样代码
    // 这里暂时返回空，待后续实现
    
    LOG_WARN("process_audio_chunk not yet implemented");
    return std::nullopt;
}

sd::Tensor<float> ChunkedVideoGenerator::blend_chunks(
    const sd::Tensor<float>& chunk_a,
    const sd::Tensor<float>& chunk_b,
    int blend_frames) {
    
    // TODO: 实现线性混合
    // 暂时返回 chunk_b
    return chunk_b;
}

float ChunkedVideoGenerator::get_progress() const {
    // TODO: 实现进度计算
    return 0.0f;
}

}  // namespace sd
```

**修改**: `src/pipeline/diffusion_engine.h`

添加部分音频嵌入方法声明：

```cpp
// 现有方法
sd::Tensor<float> get_audio_embedding(const sd_audio_t& audio);

// 新增：支持部分音频嵌入
sd::Tensor<float> get_audio_embedding_partial(
    const float* audio_data,
    int64_t sample_count,
    int sample_rate
);
```

**修改**: `src/pipeline/diffusion_engine.cpp`

添加部分音频嵌入实现：

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

**验证**:
```bash
cd D:/workspace/stable-diffusion.cpp
cmake --build build --config Release -j8 2>&1 | grep -E "error:|video_chunked"
# 预期：编译成功或显示待定实现警告
```

---

### Task 4: 实现重叠区域平滑混合

**新建文件**: `src/pipeline/video_blend.cpp`

```cpp
#include "pipeline/video_blend.hpp"
#include "core/tensor.hpp"

namespace sd {

/**
 * 对两个相邻视频块的重叠区域进行线性混合
 * 
 * @param chunk_a 前一块 latent [C, H, T_a, W]
 * @param chunk_b 后一块 latent [C, H, T_b, W]
 * @param blend_frames 重叠区域帧数
 * @return 混合后的 latent [C, H, T_a + T_b - blend_frames, W]
 */
sd::Tensor<float> blend_adjacent_chunks(
    const sd::Tensor<float>& chunk_a,
    const sd::Tensor<float>& chunk_b,
    int blend_frames) {
    
    if (blend_frames <= 0) {
        return chunk_b;
    }
    
    auto shape_a = chunk_a.shape();
    auto shape_b = chunk_b.shape();
    
    if (shape_a.size() != 4 || shape_b.size() != 4) {
        LOG_ERROR("Invalid latent shape for blending");
        return chunk_b;
    }
    
    int C = shape_a[0];
    int H = shape_a[1];
    int T_a = shape_a[2];
    int T_b = shape_b[2];
    int W = shape_a[3];
    
    if (shape_b[3] != W) {
        LOG_ERROR("Latent width mismatch for blending");
        return chunk_b;
    }
    
    int T_out = T_a + T_b - blend_frames;
    
    // 分配输出 tensor
    sd::Tensor<float> result({C, H, T_out, W});
    
    // 复制 chunk_a 的非重叠部分
    if (T_a > blend_frames) {
        auto a_part = sd::ops::slice(chunk_a, 2, 0, T_a - blend_frames);
        // 复制数据...
    }
    
    // 混合重叠区域
    // TODO: 实现线性混合
    
    // 复制 chunk_b 的非重叠部分
    // TODO: 实现
    
    return result;
}

}  // namespace sd
```

---

### Task 5: 添加 CLI 支持

**修改文件**: `examples/cli/main.cpp`

在 `vid_gen` 模式中添加分块生成选项：

```cpp
// 在参数解析部分添加
bool chunked_mode = false;
float chunk_duration = 5.0f;
float overlap_duration = 1.0f;

// 解析命令行参数
for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--chunked") == 0) {
        chunked_mode = true;
    } else if (strcmp(argv[i], "--chunk-duration") == 0 && i + 1 < argc) {
        chunk_duration = atof(argv[++i]);
    } else if (strcmp(argv[i], "--overlap") == 0 && i + 1 < argc) {
        overlap_duration = atof(argv[++i]);
    }
}

// 在生成逻辑中
if (chunked_mode && sd->version == VERSION_WAN2_MULTI_TALK) {
    // 使用分块生成器
    sd::ChunkedVideoGenerator generator(sd, {
        chunk_duration,
        overlap_duration,
        params.fps,
        params.frames,
        true,
        0.5f
    });
    
    sd_image_t** frames = nullptr;
    int num_frames = 0;
    int fps = 0;
    
    if (generator.generate_chunked_video(
            params.init_image,
            audio_data,
            audio_samples,
            audio_sample_rate,
            audio_channels,
            &frames,
            &num_frames,
            &fps)) {
        
        // 保存视频
        save_video(frames, num_frames, fps, output_path);
        
        // 释放内存
        for (int i = 0; i < num_frames; i++) {
            delete frames[i];
        }
        delete[] frames;
    }
} else {
    // 使用原有生成逻辑
    // ... 现有代码 ...
}
```

---

### Task 6: 实现流式解码降低内存占用

**新建文件**: `src/pipeline/video_stream.hpp`

```cpp
#ifndef __SD_PIPELINE_VIDEO_STREAM_HPP__
#define __SD_PIPELINE_VIDEO_STREAM_HPP__

#include <vector>
#include "core/tensor.hpp"

namespace sd {

/**
 * 流式视频解码器
 * 
 * 边生成 latent 边解码，降低峰值内存占用
 */
class StreamVideoDecoder {
public:
    StreamVideoDecoder(StableDiffusionGGML* sd, int decode_batch_size = 20);
    ~StreamVideoDecoder();
    
    /**
     * 解码单个 latent 块
     */
    std::vector<sd_image_t> decode_batch(const sd::Tensor<float>& latent);
    
    /**
     * 清空内部缓存
     */
    void flush();
    
private:
    StableDiffusionGGML* sd_;
    int batch_size_;
    std::vector<sd::Tensor<float>> latent_cache_;
};

}  // namespace sd

#endif  // __SD_PIPELINE_VIDEO_STREAM_HPP__
```

---

### Task 7: 编写性能基准测试

**新建文件**: `scripts/benchmark_chunked_video.py`

```python
#!/usr/bin/env python3
"""
分块视频生成性能基准测试
"""

import subprocess
import time
import sys
from pathlib import Path

def run_benchmark(model_path, audio_path, image_path, output_path, chunked=False):
    """运行单次基准测试"""
    
    cmd = [
        "./build/bin/sd-cli.exe", "-M", "vid_gen",
        "--diffusion-model", str(model_path),
        "--audio-encoder", "path/to/wav2vec2.safetensors",
        "-i", str(image_path),
        "--audio", str(audio_path),
        "-o", str(output_path),
        "-d", "0",
        "--log-level", "warning"
    ]
    
    if chunked:
        cmd.extend(["--chunked", "--chunk-duration", "5.0", "--overlap", "1.0"])
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    return {
        'elapsed': elapsed,
        'stdout': result.stdout,
        'stderr': result.stderr
    }

def main():
    model = Path("path/to/Wan2_1-InfiniteTalk_Single_Q8.gguf")
    audio = Path("test_data/long_audio.wav")  # 30秒音频
    image = Path("test_data/reference.jpg")
    
    if not model.exists():
        print("Model not found")
        sys.exit(1)
    
    print("=== Benchmark: Standard S2V (81 frames) ===")
    result1 = run_benchmark(model, audio, image, "output_standard.mp4", chunked=False)
    if result1:
        print(f"Time: {result1['elapsed']:.2f}s")
    
    print("\n=== Benchmark: Chunked Generation (30s audio) ===")
    result2 = run_benchmark(model, audio, image, "output_chunked.mp4", chunked=True)
    if result2:
        print(f"Time: {result2['elapsed']:.2f}s")
        print(f"Speedup: {result1['elapsed'] / result2['elapsed']:.2f}x" if result1 else "N/A")

if __name__ == '__main__':
    main()
```

---

### Task 8: 端到端测试验证

**测试脚本**: `scripts/test_infini_talk.sh`

```bash
#!/bin/bash
set -e

MODEL="C:/path/to/Wan2_1-InfiniteTalk_Single_Q8.gguf"
AUDIO_ENCODER="C:/path/to/wav2vec2_large_english_fp16.safetensors"
IMAGE="test_data/reference.jpg"
AUDIO_SHORT="test_data/short_audio.wav"  # 5秒
AUDIO_LONG="test_data/long_audio.wav"    # 30秒
OUTPUT_DIR="test_results"

mkdir -p "$OUTPUT_DIR"

echo "=== Test 1: 短音频标准模式 ==="
./build/bin/sd-cli.exe -M vid_gen \
    --diffusion-model "$MODEL" \
    --audio-encoder "$AUDIO_ENCODER" \
    -i "$IMAGE" \
    --audio "$AUDIO_SHORT" \
    --video-frames 81 \
    -o "$OUTPUT_DIR/short_standard.mp4" \
    -d 0

echo "=== Test 2: 长音频分块模式 ==="
./build/bin/sd-cli.exe -M vid_gen \
    --diffusion-model "$MODEL" \
    --audio-encoder "$AUDIO_ENCODER" \
    -i "$IMAGE" \
    --audio "$AUDIO_LONG" \
    --chunked \
    --chunk-duration 5.0 \
    --overlap 1.0 \
    -o "$OUTPUT_DIR/long_chunked.mp4" \
    -d 0

echo "=== Test 3: 验证输出文件 ==="
for f in "$OUTPUT_DIR"/*.mp4; do
    size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)
    if [ "$size" -gt 0 ]; then
        echo "✓ $f ($size bytes)"
    else
        echo "✗ $f (empty)"
    fi
done

echo "=== Test 4: 视频信息 ==="
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 "$OUTPUT_DIR/long_chunked.mp4"
```

---

## Tests / Validation

### 单元测试

**文件**: `test/test_audio_chunker.cpp`

```cpp
#include "audio/audio_chunker.hpp"
#include <cassert>
#include <vector>

void test_audio_chunker_basic() {
    sd::audio::AudioChunker chunker(16000, 1, 5.0f, 1.0f);
    
    std::vector<float> audio(16000 * 10, 1.0f);  // 10秒音频
    
    auto chunks = chunker.chunk(audio.data(), audio.size());
    
    assert(chunks.size() == 2);  // 10秒 / 4秒步长 = 2.5 → 2块（有重叠）
    assert(chunks[0].start_sample == 0);
    assert(chunks[0].duration_samples == 16000 * 5);
    assert(chunks[1].start_sample == 16000 * 4);  // 重叠 1 秒
}

void test_audio_chunker_short() {
    sd::audio::AudioChunker chunker(16000, 1, 5.0f, 1.0f);
    
    std::vector<float> audio(16000 * 3, 1.0f);  // 3秒音频
    
    auto chunks = chunker.chunk(audio.data(), audio.size());
    
    assert(chunks.size() == 1);
    assert(chunks[0].duration_samples == 16000 * 3);
}

int main() {
    test_audio_chunker_basic();
    test_audio_chunker_short();
    printf("All tests passed!\n");
    return 0;
}
```

### 集成测试

**测试矩阵**:

| 测试场景 | 音频时长 | 预期输出 | 验证方法 |
|---------|---------|---------|---------|
| 短音频（< 5s） | 3s | 完整视频，无分块 | 检查帧数 = 48 (3s @ 16fps) |
| 单块（≈ 5s） | 5s | 单个 chunk 生成 | 检查日志无分块信息 |
| 多块（> 5s） | 15s | 3 个 chunk 拼接 | 检查帧数 ≈ 240 (15s @ 16fps) |
| 重叠混合 | 10s | 平滑过渡 | 人工视觉检查 |
| 长音频 | 60s | 完整视频 | 检查时长和口型同步 |

**验证命令**:
```bash
# 运行所有测试
cd D:/workspace/stable-diffusion.cpp
./scripts/test_infini_talk.sh

# 检查输出
ls -lh test_results/*.mp4

# 验证视频信息
for f in test_results/*.mp4; do
    echo "=== $f ==="
    ffprobe -v error -show_entries format=duration,bit_rate -of default=noprint_wrappers=1 "$f"
done
```

---

## Risks, Tradeoffs, and Open Questions

### 风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 重叠区域 artifacts | 中 | 中 | 实现多种混合策略（线性、cosine、AI 修复） |
| 口型不同步 | 低 | 高 | 使用精确音频时间戳对齐 |
| VRAM OOM | 高 | 高 | 实现流式解码 + 显存管理 |
| 生成速度慢 | 中 | 中 | 支持 LightX2V LoRA 加速 |
| 边缘情况崩溃 | 中 | 中 | 添加详细日志和异常处理 |

### 权衡

1. **块大小选择**:
   - 大块（10s）→ 更少推理次数，但内存占用高
   - 小块（2s）→ 内存友好，但重叠处理复杂

2. **混合策略**:
   - 简单混合（线性）→ 快速，可能有接缝
   - 复杂混合（扩散重采样）→ 质量高，但慢

3. **流式 vs 批量**:
   - 流式解码 → 低内存，但编码开销
   - 批量解码 → 高内存，但效率高

### 待解决问题

1. **无限循环场景**: 如果音频持续输入，如何实现实时生成？
2. **状态保持**: 如何保持多块之间的人物一致性和背景一致性？
3. **错误恢复**: 如果中间某块失败，如何回滚或重试？
4. **GPU 利用率**: 多块推理时能否流水线化（下一块编码同时上一块解码）？

---

## 实施优先级

```
Phase 1 (核心功能):
  Task 1 → Task 2 → Task 3 → Task 7
  目标：实现基本分块生成

Phase 2 (质量优化):
  Task 4 → Task 6
  目标：平滑过渡 + 流式解码

Phase 3 (生产就绪):
  Task 5 → Task 8
  目标：CLI 集成 + 完整测试
```

**预计工作量**:
- Phase 1: 4-6 小时
- Phase 2: 2-3 小时
- Phase 3: 1-2 小时
- **总计**: 8-12 小时

---

**计划文件**: `.hermes/plans/2026-09-30_091500-infini-talk-full-support.md`
