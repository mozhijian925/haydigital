#ifndef __SD_PIPELINE_VIDEO_CHUNKED_HPP__
#define __SD_PIPELINE_VIDEO_CHUNKED_HPP__

#include <memory>
#include <optional>
#include <vector>

#include "audio/audio_chunker.hpp"
#include "stable-diffusion.h"

namespace sd {

class StableDiffusionGGML;

/**
 * 分块视频生成器
 * 
 * 支持长音频驱动的数字人口型视频生成
 */
class ChunkedVideoGenerator {
public:
    struct Params {
        float chunk_duration_sec = 5.0f;   // 每块时长
        float overlap_sec = 1.0f;          // 重叠区域
        int fps = 16;                      // 输出帧率
        int video_frames = 81;             // 每块帧数
        bool blend_overlaps = true;        // 是否混合重叠区域
        float blend_strength = 0.5f;       // 混合强度
    };

    ChunkedVideoGenerator(StableDiffusionGGML* sd, const Params& params);
    ~ChunkedVideoGenerator();

    /**
     * 生成分块视频
     * @param ref_image 参考图片
     * @param audio 驱动音频（单声道，16kHz）
     * @param audio_samples 音频采样数
     * @param output_frames 输出帧数组（由调用者管理）
     * @param num_frames_out 输出帧数
     * @param fps_out 输出帧率
     * @return 成功返回 true
     */
    bool generate(const sd_image_t& ref_image,
                  const float* audio,
                  int64_t audio_samples,
                  sd_image_t*** output_frames,
                  int* num_frames_out,
                  int* fps_out);

    /**
     * 获取生成进度（0.0 - 1.0）
     */
    float get_progress() const;

private:
    StableDiffusionGGML* sd_;
    Params params_;
    AudioChunker chunker_;
    std::vector<sd::Tensor<float>> processed_chunks_;

    /**
     * 处理单个音频块
     */
    std::optional<sd::Tensor<float>> process_chunk(const AudioChunk& chunk,
                                                    const sd::Tensor<float>& ref_latent);

    /**
     * 混合两个相邻块的过渡区域
     */
    sd::Tensor<float> blend_chunks(const sd::Tensor<float>& chunk_a,
                                    const sd::Tensor<float>& chunk_b,
                                    int blend_frames);
};

}  // namespace sd

#endif  // __SD_PIPELINE_VIDEO_CHUNKED_HPP__
