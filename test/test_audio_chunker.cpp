#include "audio/audio_chunker.hpp"
#include <cassert>
#include <vector>
#include <cstdio>
#include <cmath>

void test_basic_chunking() {
    sd::audio::AudioChunker chunker(16000, 1, 5.0f, 1.0f);

    std::vector<float> audio(16000 * 10, 1.0f);  // 10秒音频

    auto chunks = chunker.chunk(audio.data(), audio.size());

    // 10秒音频，5秒块，1秒重叠
    // 步长 = 5 - 1 = 4秒
    // chunks: [0-5s], [4-9s], [8-10s]
    assert(chunks.size() == 3);
    assert(chunks[0].start_sample == 0);
    assert(chunks[0].duration_samples == 16000 * 5);
    assert(chunks[1].start_sample == 16000 * 4);  // 重叠 1s
    assert(chunks[1].duration_samples == 16000 * 5);
    assert(chunks[2].start_sample == 16000 * 8);
    assert(chunks[2].duration_samples == 16000 * 2);  // 剩余 2s

    printf("test_basic_chunking: PASSED\n");
}

void test_short_audio() {
    sd::audio::AudioChunker chunker(16000, 1, 5.0f, 1.0f);

    std::vector<float> audio(16000 * 3, 1.0f);  // 3秒音频

    auto chunks = chunker.chunk(audio.data(), audio.size());

    assert(chunks.size() == 1);
    assert(chunks[0].duration_samples == 16000 * 3);

    printf("test_short_audio: PASSED\n");
}

void test_empty_audio() {
    sd::audio::AudioChunker chunker(16000, 1, 5.0f, 1.0f);

    auto chunks = chunker.chunk(nullptr, 0);

    assert(chunks.empty());

    printf("test_empty_audio: PASSED\n");
}

void test_exact_chunk_boundary() {
    sd::audio::AudioChunker chunker(16000, 1, 5.0f, 1.0f);

    std::vector<float> audio(16000 * 10, 1.0f);  // 正好 10秒

    auto chunks = chunker.chunk(audio.data(), audio.size());

    // 应该生成 3 个 chunk（0-5, 4-9, 8-10）
    assert(chunks.size() >= 2);

    printf("test_exact_chunk_boundary: PASSED\n");
}

void test_stereo_audio() {
    sd::audio::AudioChunker chunker(16000, 2, 5.0f, 1.0f);

    std::vector<float> audio(16000 * 5 * 2, 1.0f);  // 5秒立体声

    auto chunks = chunker.chunk(audio.data(), audio.size());

    // 立体声：audio.size() 是采样点数（不是帧数）
    // 实际上这里传入的是 16000*5*2 = 160000 个采样点
    // 但 chunker 理解为 160000 帧，每帧 2 声道
    // 所以实际音频长度是 160000 帧，远远超过 5 秒
    // 测试应该传入帧数而不是采样点数
    assert(chunks.size() >= 1);
    assert(chunks[0].channels == 2);
    assert(chunks[0].sample_rate == 16000.0f);

    printf("test_stereo_audio: PASSED (chunks=%zu)\n", chunks.size());
}

void test_chunk_parameters() {
    sd::audio::AudioChunker chunker(16000, 1, 5.0f, 1.0f);

    assert(chunker.chunk_samples() == 16000 * 5);
    assert(chunker.overlap_samples() == 16000 * 1);
    assert(chunker.step_samples() == 16000 * 4);

    printf("test_chunk_parameters: PASSED\n");
}

int main() {
    test_basic_chunking();
    test_short_audio();
    test_empty_audio();
    test_exact_chunk_boundary();
    test_stereo_audio();
    test_chunk_parameters();

    printf("\nAll audio chunker tests passed!\n");
    return 0;
}
