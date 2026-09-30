#ifndef __SD_AUDIO_AUDIO_CHUNKER_HPP__
#define __SD_AUDIO_AUDIO_CHUNKER_HPP__

#include <cstdint>
#include <vector>

namespace sd::audio {

/**
 * 音频块结构
 */
struct AudioChunk {
    int64_t start_sample;      // 块起始采样点（全局）
    int64_t end_sample;        // 块结束采样点（全局）
    int64_t duration_samples;  // 块长度（采样点数）
    float sample_rate;         // 采样率
    int channels;              // 声道数
    const float* data;         // 音频数据指针（原始音频的切片）
};

/**
 * 音频分块处理器
 * 
 * 将长音频分割为固定长度的块，用于分块视频生成
 */
class AudioChunker {
public:
    /**
     * 构造函数
     * @param sample_rate 音频采样率（Hz）
     * @param channels 声道数
     * @param chunk_duration_sec 每个块的时长（秒）
     * @param overlap_sec 重叠区域时长（秒），用于平滑过渡
     */
    AudioChunker(int sample_rate, int channels,
                 float chunk_duration_sec = 5.0f,
                 float overlap_sec = 1.0f);

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
