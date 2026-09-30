#include "audio/audio_chunker.hpp"

#include <algorithm>
#include <cstdio>

namespace sd::audio {

AudioChunker::AudioChunker(int sample_rate, int channels,
                           float chunk_duration_sec,
                           float overlap_sec)
    : sample_rate_(sample_rate),
      channels_(channels),
      chunk_samples_(static_cast<int>(sample_rate_ * chunk_duration_sec)),
      overlap_samples_(static_cast<int>(sample_rate_ * overlap_sec)) {
    // 确保 chunk 大小是 4 的倍数（VAE 缩放因子）
    chunk_samples_ = (chunk_samples_ / 4) * 4;
    overlap_samples_ = (overlap_samples_ / 4) * 4;
    step_samples_ = chunk_samples_ - overlap_samples_;

    if (step_samples_ <= 0) {
        // 防止除零或无限循环
        step_samples_ = chunk_samples_;
    }
}

std::vector<AudioChunk> AudioChunker::chunk(const float* audio, int64_t samples) {
    std::vector<AudioChunk> chunks;

    if (samples <= 0 || audio == nullptr) {
        return chunks;
    }

    // samples 是帧数（frame count），不是采样点数
    // 总采样点数 = samples * channels
    int64_t total_samples = samples * channels_;
    int64_t chunk_sample_count = chunk_samples_ * channels_;
    int64_t step_sample_count = step_samples_ * channels_;

    int64_t offset = 0;

    while (offset < total_samples) {
        AudioChunk chunk;
        chunk.start_sample = offset / channels_;
        int64_t remaining = total_samples - offset;
        chunk.duration_samples = std::min<int64_t>(chunk_samples_, remaining / channels_);
        chunk.end_sample = chunk.start_sample + chunk.duration_samples;
        chunk.sample_rate = static_cast<float>(sample_rate_);
        chunk.channels = channels_;
        chunk.data = &audio[offset];

        chunks.push_back(chunk);

        // 移动步长
        offset += step_sample_count;

        // 如果是最后一个块且不足一个 chunk，直接添加
        if (offset >= total_samples) {
            break;
        }
    }

    return chunks;
}

}  // namespace sd::audio
