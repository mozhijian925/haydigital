#include "pipeline/video_chunked.hpp"

#include "pipeline/diffusion_engine.h"
#include "pipeline/generation.h"
#include "pipeline/request.h"
#include "model/diffusion/model.hpp"
#include "core/util.h"
#include "runtime/audio_processing.h"

namespace sd {

ChunkedVideoGenerator::ChunkedVideoGenerator(
    StableDiffusionGGML* sd,
    const Params& params)
    : sd_(sd),
      params_(params),
      // wav2vec2 使用 16kHz 采样率
      chunker_(16000, 1, params.chunk_duration_sec, params.overlap_sec) {
}

ChunkedVideoGenerator::~ChunkedVideoGenerator() = default;

bool ChunkedVideoGenerator::generate(
    const sd_image_t& ref_image,
    const float* audio,
    int64_t audio_samples,
    sd_image_t*** output_frames,
    int* num_frames_out,
    int* fps_out) {

    if (sd_ == nullptr || audio == nullptr || audio_samples <= 0) {
        LOG_ERROR("Invalid input for chunked video generation");
        return false;
    }

    LOG_INFO("Starting chunked video generation");
    LOG_INFO("  Audio: %lld samples @ 16kHz, mono", (long long)audio_samples);
    LOG_INFO("  Chunk duration: %.2fs, Overlap: %.2fs",
             params_.chunk_duration_sec, params_.overlap_sec);

    // 1. 编码参考图片为 latent
    auto ref_tensor = sd_image_to_tensor(ref_image);
    ref_tensor = ensure_image_tensor_channels(ref_tensor, sd_->get_image_channels());
    auto ref_latent = sd_->encode_first_stage(ref_tensor);
    if (ref_latent.empty()) {
        LOG_ERROR("failed to encode reference image");
        return false;
    }
    LOG_INFO("Reference image encoded to latent: [%lld, %lld, %lld, %lld]",
             (long long)ref_latent.shape()[0], (long long)ref_latent.shape()[1],
             (long long)ref_latent.shape()[2], (long long)ref_latent.shape()[3]);

    // 2. 音频预处理：下混到单声道 + 重采样到 16kHz
    // 注意：AudioChunker 期望输入已经是单声道 16kHz
    // 这里我们假设输入已经是单声道，只进行重采样
    auto chunked_audio = sd::audio::resample_audio(audio, audio_samples, 16000, 16000);
    if (chunked_audio.empty()) {
        LOG_ERROR("failed to resample audio");
        return false;
    }

    // 3. 音频分块
    auto chunks = chunker_.chunk(chunked_audio.data(), chunked_audio.size());
    if (chunks.empty()) {
        LOG_ERROR("no audio chunks generated");
        return false;
    }
    LOG_INFO("Generated %zu audio chunks", chunks.size());

    // 4. 逐块推理
    std::vector<sd::Tensor<float>> all_latents;
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto chunk_latent = process_chunk(chunks[i], ref_latent);
        if (!chunk_latent.has_value()) {
            LOG_ERROR("chunk %zu failed", i);
            return false;
        }
        all_latents.push_back(std::move(chunk_latent.value()));
        LOG_INFO("Processed chunk %zu/%zu", i + 1, chunks.size());
    }

    // 5. 拼接所有 latent（按时间维度 T 拼接）
    auto final_latent = all_latents[0];
    for (size_t i = 1; i < all_latents.size(); ++i) {
        final_latent = sd::ops::concat(final_latent, all_latents[i], 2);
    }
    LOG_INFO("Final latent shape: [%lld, %lld, %lld, %lld]",
             (long long)final_latent.shape()[0], (long long)final_latent.shape()[1],
             (long long)final_latent.shape()[2], (long long)final_latent.shape()[3]);

    // 6. 解码视频
    auto video = sd_->decode_first_stage(final_latent, true);
    if (video.empty()) {
        LOG_ERROR("failed to decode video");
        return false;
    }
    LOG_INFO("Video decoded: [%lld, %lld, %lld, %lld]",
             (long long)video.shape()[0], (long long)video.shape()[1],
             (long long)video.shape()[2], (long long)video.shape()[3]);

    // 7. 转换为图像帧数组
    int T = static_cast<int>(video.shape()[2]);  // 时间维度
    int H = static_cast<int>(video.shape()[1]);  // 高度
    int W = static_cast<int>(video.shape()[0]);  // 宽度

    *output_frames = (sd_image_t**)malloc(sizeof(sd_image_t*) * T);
    if (*output_frames == nullptr) {
        LOG_ERROR("failed to allocate output frames");
        return false;
    }

    for (int t = 0; t < T; ++t) {
        (*output_frames)[t] = new sd_image_t(tensor_to_sd_image(video, t));
    }

    if (num_frames_out) *num_frames_out = T;
    if (fps_out) *fps_out = params_.fps;

    LOG_INFO("Chunked video generation completed: %d frames @ %d fps", T, params_.fps);
    return true;
}

float ChunkedVideoGenerator::get_progress() const {
    // TODO: 实现进度计算
    return 0.0f;
}

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

sd::Tensor<float> ChunkedVideoGenerator::blend_chunks(
    const sd::Tensor<float>& chunk_a,
    const sd::Tensor<float>& chunk_b,
    int blend_frames) {
    // TODO: 实现线性混合
    return chunk_b;
}

}  // namespace sd
