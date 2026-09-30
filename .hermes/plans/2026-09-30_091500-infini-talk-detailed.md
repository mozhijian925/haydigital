# 技术方案：在 stable-diffusion.cpp 中集成 MultiTalk/InfiniteTalk 数字人口型视频

## Goal

在 stable-diffusion.cpp 源码上进行二次开发，使项目能够加载并运行 InfiniTetalk 工作流（`Wan2_1-InfiniteTalk_Single_Q8.gguf`），实现音频驱动的图片数字人口型同步视频生成。

---

## Current Context / Assumptions

### 现有架构理解

```
stable-diffusion.cpp 核心流程：
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  model_loader   │ →  │  version 检测     │ →  │  model_builder  │
│  (.cpp:556-637) │    │  (Wan/S2V/I2V等)  │    │  (创建对应 runner)│
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                        │
                                        ┌───────────────▼───────────────┐
                                        │  diffusion_engine / video    │
                                        │  (prepare_latents, sampling) │
                                        └───────────────┬───────────────┘
                                                        │
                                        ┌───────────────▼───────────────┐
                                        │  WanRunner::forward()        │
                                        │  (wav2vec2 → audio_embeds)  │
                                        └───────────────────────────────┘
```

### 关键发现

| 组件 | 路径 | 说明 |
|------|------|------|
| 版本检测 | `src/model_loader.cpp:556-562` | 通过权重键名判断 S2V (`casual_audio_encoder` 或 `audio_injector`) |
| 版本枚举 | `src/model.h:35-38` | `VERSION_WAN2`, `VERSION_WAN2_2_S2V` 等 |
| 模型构建 | `src/pipeline/model_builders.cpp:225-263` | 根据版本创建 `WAN::WanRunner` 或 `Wav2Vec2::Wav2Vec2ModelRunner` |
| 视频管道 | `src/pipeline/video.cpp:1081-1127` | S2V 专用处理逻辑 |
| Wan 模型 | `src/model/diffusion/wan.hpp:887-971` | `WanRunner` 类，包含 `WanAudioInjector` |

### 假设

1. **`Wan2_1-InfiniteTalk_Single_Q8.gguf` 可能是标准 S2V 的变体**，带有特殊的音频嵌入层（如 `multi_talk_embeds`）
2. 如果权重结构兼容，只需修改版本检测 + 条件处理即可；如果不兼容，需要新建 `MultiTalkRunner`
3. 音频编码器仍使用 wav2vec2（与 S2V 相同）

---

## Architecture / Proposed Approach

**方案**: 渐进式实现，优先最小改动

1. **Phase 1（快速验证）**: 尝试直接用现有 S2V 管线加载 InfiniteTalk 模型，观察是否兼容
2. **Phase 2（兼容性调整）**: 如需要，添加 `VERSION_WAN2_MULTI_TALK` 版本类型，修改检测逻辑和条件处理
3. **Phase 3（深度定制，可选）**: 如模型有独特架构，新建 `MultiTalkRunner` 覆盖特定行为

**为什么选择这个方案**:
- 遵循 YAGNI 原则，先验证再扩展
- 现有 S2V 基础设施（wav2vec2、audio_injector）可能已足够
- 最小化代码变更，降低回归风险

---

## Step-by-Step Tasks

### Task 1: 检查 InfiniteTalk 模型权重结构

**目标**: 确认模型是否包含 S2V 所需的关键键

**操作**: 使用 Python 脚本读取 GGUF 文件的键名（不加载权重）

```python
# scripts/check_gguf_keys.py
import struct
import sys
from pathlib import Path

def read_gguf_keys(path: str):
    """读取 GGUF 文件的所有张量键名和形状"""
    keys = []
    with open(path, 'rb') as f:
        # GGUF magic number + version
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Not a GGUF file: {path}")
        version = struct.unpack('<I', f.read(4))[0]
        n_kv, n_tensors = struct.unpack('<QQ', f.read(16))
        
        # Skip key-value pairs
        for _ in range(n_kv):
            n_chars = struct.unpack('<I', f.read(4))[0]
            f.read(n_chars)  # skip key
            f.read(8)  # skip value type
            # Skip value data based on type...
            # For simplicity, just count tensors
        
        # Read tensor info
        for _ in range(n_tensors):
            n_chars = struct.unpack('<I', f.read(4))[0]
            name = f.read(n_chars).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            shape = list(struct.unpack('<' + 'Q' * n_dims, f.read(8 * n_dims)))
            keys.append((name, shape))
    
    return keys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_gguf_keys.py <model.gguf>")
        sys.exit(1)
    
    path = sys.argv[1]
    keys = read_gguf_keys(path)
    
    # 分类输出
    audio_keys = [k for k in keys if any(x in k[0].lower() for x in ['audio', 'injector', 'wav2vec', 'talk', 'embed'])]
    wan_keys = [k for k in keys if 'model.diffusion_model' in k[0]]
    
    print(f"Total tensors: {len(keys)}")
    print(f"\n=== Audio-related keys ({len(audio_keys)}) ===")
    for name, shape in sorted(audio_keys, key=lambda x: x[0]):
        print(f"  {name}: {shape}")
    
    print(f"\n=== Wan model keys (first 20) ===")
    for name, shape in sorted(wan_keys, key=lambda x: x[0])[:20]:
        print(f"  {name}: {shape}")
```

**验证命令**:
```bash
cd D:/workspace/stable-diffusion.cpp
python scripts/check_gguf_keys.py "C:/path/to/Wan2_1-InfiniteTalk_Single_Q8.gguf"
```

**预期输出**: 
- 如果包含 `audio_injector.*` 和 `wav2vec2.*` → 兼容 S2V
- 如果包含 `multi_talk_embeds.*` → 需要额外处理
- 如果没有音频相关键 → 不兼容，需全新实现

---

### Task 2: 尝试直接用 S2V 模式运行（验证兼容性）

**目标**: 在不修改代码的情况下，测试模型是否可用

**操作**: 使用现有 CLI 工具

```bash
# 假设模型和音频编码器已准备好
./build/bin/sd-cli.exe -M vid_gen \
    --diffusion-model "C:/path/to/Wan2_1-InfiniteTalk_Single_Q8.gguf" \
    --audio-encoder "C:/path/to/wav2vec2_large_english_fp16.safetensors" \
    -i "C:/path/to/reference.jpg" \
    --audio "C:/path/to/speech.wav" \
    --video-frames 81 \
    -o "output_test.mp4" \
    -d 0 \
    --log-level debug 2>&1 | tee debug.log
```

**验证**:
- 成功：日志显示 `Wan2.2-S2V-14B` 或类似描述 → 无需修改，直接使用
- 失败：日志显示版本检测错误或缺少键 → 进入 Task 3

**关键日志检查**:
```bash
grep -E "version|S2V|MultiTalk|audio_injector|wav2vec" debug.log
```

---

### Task 3: 添加 VERSION_WAN2_MULTI_TALK 枚举

**文件**: `src/model.h`

**位置**: 第 38 行之后（`VERSION_WAN2_2_S2V` 之后）

**修改内容**:
```cpp
// 现有代码 (line 35-38)
    VERSION_WAN2,
    VERSION_WAN2_2_I2V,
    VERSION_WAN2_2_TI2V,
    VERSION_WAN2_2_S2V,
    VERSION_LINGBOT_VIDEO,  // <-- 第 39 行

// 修改为：
    VERSION_WAN2,
    VERSION_WAN2_2_I2V,
    VERSION_WAN2_2_TI2V,
    VERSION_WAN2_2_S2V,
    VERSION_WAN2_MULTI_TALK,  // <-- 新增
    VERSION_LINGBOT_VIDEO,
```

**验证**:
```bash
cd D:/workspace/stable-diffusion.cpp
cmake --build build --config Release -j8 2>&1 | grep -E "error|warning"
# 预期：无错误输出
```

---

### Task 4: 更新版本检测逻辑

**文件**: `src/model_loader.cpp`

**位置**: 第 559-563 行（S2V 检测区域）

**当前代码**:
```cpp
if (tensor_storage.name.find("casual_audio_encoder.weights") != std::string::npos ||
    tensor_storage.name.find("audio_injector.injector.0.q.weight") != std::string::npos) {
    // S2V and T2V-14B share patch_embedding shapes.
    is_s2v = true;
}
```

**修改为**:
```cpp
if (tensor_storage.name.find("casual_audio_encoder.weights") != std::string::npos ||
    tensor_storage.name.find("audio_injector.injector.0.q.weight") != std::string::npos) {
    // S2V and T2V-14B share patch_embedding shapes.
    is_s2v = true;
}
// 新增：检测 MultiTalk 模型
if (tensor_storage.name.find("multi_talk_embeds") != std::string::npos ||
    tensor_storage.name.find("infinite_talk") != std::string::npos) {
    is_multi_talk = true;
}
```

**同时需要在函数开头添加变量声明**（约第 449 行附近）:
```cpp
bool is_s2v                      = false;
bool is_multi_talk               = false;  // <-- 新增
```

**并在版本返回逻辑中处理**（第 627-628 行附近）:
```cpp
if (is_wan) {
    LOG_VERBOSE("patch_embedding_channels %d", patch_embedding_channels);
    if (is_multi_talk) {
        return VERSION_WAN2_MULTI_TALK;  // <-- 新增，优先于 S2V
    }
    if (is_s2v) {
        return VERSION_WAN2_2_S2V;
    }
    // ... 其余逻辑不变
}
```

**验证**:
```bash
cmake --build build --config Release -j8
./build/bin/sd-cli.exe -M vid_gen --help | grep -i "multi"
# 预期：构建成功，无编译错误
```

---

### Task 5: 更新版本判断函数

**文件**: `src/model.h`

**位置**: 第 137-142 行（`sd_version_is_wan()` 函数）

**当前代码**:
```cpp
static inline bool sd_version_is_wan(SDVersion version) {
    if (version == VERSION_WAN2 || version == VERSION_WAN2_2_I2V || 
        version == VERSION_WAN2_2_TI2V || version == VERSION_WAN2_2_S2V) {
        return true;
    }
    return false;
}
```

**修改为**:
```cpp
static inline bool sd_version_is_wan(SDVersion version) {
    if (version == VERSION_WAN2 || version == VERSION_WAN2_2_I2V || 
        version == VERSION_WAN2_2_TI2V || version == VERSION_WAN2_2_S2V ||
        version == VERSION_WAN2_MULTI_TALK) {  // <-- 新增
        return true;
    }
    return false;
}
```

**验证**:
```bash
grep -n "sd_version_is_wan" D:/workspace/stable-diffusion.cpp/src/model.h
# 确认函数定义已更新
```

---

### Task 6: 更新模型构建器

**文件**: `src/pipeline/model_builders.cpp`

**位置**: 第 254-263 行（S2V 音频编码器构建区域）

**当前代码**:
```cpp
if (version == VERSION_WAN2_2_S2V &&
    tensor_storage_map.count("wav2vec2.encoder.layer_norm.bias") > 0) {
    if (!ensure_backend_pair(ctx.backends, SDBackendModule::AUDIO_ENCODER)) {
        return false;
    }
    result.audio_encoder = std::make_shared<Wav2Vec2::Wav2Vec2ModelRunner>(
        ctx.backends.runtime_backend(SDBackendModule::AUDIO_ENCODER),
        tensor_storage_map,
        "wav2vec2.",
        weight_manager);
}
```

**修改为**:
```cpp
if ((version == VERSION_WAN2_2_S2V || version == VERSION_WAN2_MULTI_TALK) &&
    tensor_storage_map.count("wav2vec2.encoder.layer_norm.bias") > 0) {
    if (!ensure_backend_pair(ctx.backends, SDBackendModule::AUDIO_ENCODER)) {
        return false;
    }
    result.audio_encoder = std::make_shared<Wav2Vec2::Wav2Vec2ModelRunner>(
        ctx.backends.runtime_backend(SDBackendModule::AUDIO_ENCODER),
        tensor_storage_map,
        "wav2vec2.",
        weight_manager);
}
```

**同时需要更新模型描述**（在 `src/model/diffusion/wan.hpp` 第 933-934 行）:

**当前代码**:
```cpp
} else if (config.num_layers == 40) {
    if (version == VERSION_WAN2_2_S2V) {
        desc          = "Wan2.2-S2V-14B";
        config.in_dim = 16;
```

**修改为**:
```cpp
} else if (config.num_layers == 40) {
    if (version == VERSION_WAN2_2_S2V) {
        desc          = "Wan2.2-S2V-14B";
        config.in_dim = 16;
    } else if (version == VERSION_WAN2_MULTI_TALK) {
        desc          = "Wan2.1-MultiTalk-14B";  // <-- 新增
        config.in_dim = 16;  // 假设与 S2V 相同
    }
```

**验证**:
```bash
cmake --build build --config Release -j8 2>&1 | tail -20
# 预期：构建成功
```

---

### Task 7: 更新视频生成管道

**文件**: `src/pipeline/video.cpp`

**位置 1**: 第 1081 行（S2V 条件处理）

**当前代码**:
```cpp
} else if (sd->diffusion_model->get_desc() == "Wan2.2-S2V-14B") {
```

**修改为**:
```cpp
} else if (sd->diffusion_model->get_desc() == "Wan2.2-S2V-14B" ||
           sd->diffusion_model->get_desc() == "Wan2.1-MultiTalk-14B") {
```

**位置 2**: 第 1182-1186 行（条件 embedding 设置）

**当前代码**:
```cpp
if (sd->version == VERSION_WAN2_2_S2V) {
    embeds.cond.c_ref_images = latents.ref_latents;
    if (!latents.s2v_audio_embed.empty()) {
        embeds.cond.c_ref_audios = {latents.s2v_audio_embed};
    }
}
```

**修改为**:
```cpp
if (sd->version == VERSION_WAN2_2_S2V || sd->version == VERSION_WAN2_MULTI_TALK) {
    embeds.cond.c_ref_images = latents.ref_latents;
    if (!latents.s2v_audio_embed.empty()) {
        embeds.cond.c_ref_audios = {latents.s2v_audio_embed};
    }
}
```

**位置 3**: 第 1203-1207 行（无条件 embedding 设置）

**当前代码**:
```cpp
if (sd->version == VERSION_WAN2_2_S2V) {
    embeds.uncond.c_ref_images = latents.ref_latents;
    if (!latents.s2v_audio_embed.empty()) {
        embeds.uncond.c_ref_audios = {sd::Tensor<float>::zeros_like(latents.s2v_audio_embed)};
    }
}
```

**修改为**:
```cpp
if (sd->version == VERSION_WAN2_2_S2V || sd->version == VERSION_WAN2_MULTI_TALK) {
    embeds.uncond.c_ref_images = latents.ref_latents;
    if (!latents.s2v_audio_embed.empty()) {
        embeds.uncond.c_ref_audios = {sd::Tensor<float>::zeros_like(latents.s2v_audio_embed)};
    }
}
```

**位置 4**: 第 1859-1884 行（音频输出处理）

**当前代码**:
```cpp
if (sd->version == VERSION_WAN2_2_S2V && sd_vid_gen_params->ref_audios_count > 0) {
```

**修改为**:
```cpp
if ((sd->version == VERSION_WAN2_2_S2V || sd->version == VERSION_WAN2_MULTI_TALK) && 
    sd_vid_gen_params->ref_audios_count > 0) {
```

**位置 5**: 第 1948-1960 行（音频截断处理）

**当前代码**:
```cpp
if (sd->version == VERSION_WAN2_2_S2V && generated_audio != nullptr) {
```

**修改为**:
```cpp
if ((sd->version == VERSION_WAN2_2_S2V || sd->version == VERSION_WAN2_MULTI_TALK) && 
    generated_audio != nullptr) {
```

**验证**:
```bash
cmake --build build --config Release -j8 2>&1 | grep -E "error:"
# 预期：无错误
```

---

### Task 8: 添加 CLI 自动检测提示

**文件**: `examples/cli/main.cpp`

**位置**: 查找 `vid_gen` 模式处理部分（约第 400-500 行）

**目标**: 在加载模型时输出更清晰的版本信息

**示例修改**:
```cpp
// 在模型加载成功后添加
if (sd->version == VERSION_WAN2_MULTI_TALK) {
    LOG_INFO("Loaded MultiTalk model: %s", sd->diffusion_model->get_desc().c_str());
    LOG_INFO("MultiTalk mode: audio-driven talking head video generation");
}
```

**验证**:
```bash
cmake --build build --config Release -j8
./build/bin/sd-cli.exe -M vid_gen --diffusion-model "path/to/model.gguf" --help
# 预期：成功加载并显示模型信息
```

---

### Task 9: 端到端测试

**测试数据准备**:
```bash
# 准备测试文件
mkdir -p test_data
cp /path/to/Wan2_1-InfiniteTalk_Single_Q8.gguf test_data/
cp /path/to/wav2vec2_large_english_fp16.safetensors test_data/
cp /path/to/reference.jpg test_data/
cp /path/to/speech.wav test_data/
```

**测试命令**:
```bash
cd D:/workspace/stable-diffusion.cpp
./build/bin/sd-cli.exe -M vid_gen \
    --diffusion-model test_data/Wan2_1-InfiniteTalk_Single_Q8.gguf \
    --audio-encoder test_data/wav2vec2_large_english_fp16.safetensors \
    -i test_data/reference.jpg \
    --audio test_data/speech.wav \
    --video-frames 81 \
    --width 720 \
    --height 1280 \
    -o test_data/output_multitalk.mp4 \
    -d 0 \
    --log-level info 2>&1 | tee test_results.log
```

**成功标准**:
1. 日志显示 `Wan2.1-MultiTalk-14B` 模型加载成功
2. 无内存错误或键名缺失错误
3. 生成视频文件 `test_data/output_multitalk.mp4` 存在且大小 > 0
4. 视频时长与音频时长匹配（约 5-10 秒）

**验证命令**:
```bash
# 检查输出文件
ls -lh test_data/output_multitalk.mp4

# 检查日志关键字
grep -E "MultiTalk|S2V|completed|error" test_results.log

# 预期输出示例:
# Loaded MultiTalk model: Wan2.1-MultiTalk-14B
# generate_video completed in XX.XXs
# (无 error 关键字)
```

---

### Task 10: 性能基准测试

**测试场景**:
1. 对比 S2V 和 MultiTalk 的推理速度
2. 测量 VRAM 占用
3. 验证口型同步质量

**测试脚本**:
```bash
#!/bin/bash
# scripts/benchmark_multitalk.sh

MODEL="test_data/Wan2_1-InfiniteTalk_Single_Q8.gguf"
AUDIO_ENCODER="test_data/wav2vec2_large_english_fp16.safetensors"
IMAGE="test_data/reference.jpg"
AUDIO="test_data/speech.wav"
OUTPUT="test_data/output_benchmark.mp4"

echo "=== MultiTalk Benchmark ==="
time ./build/bin/sd-cli.exe -M vid_gen \
    --diffusion-model "$MODEL" \
    --audio-encoder "$AUDIO_ENCODER" \
    -i "$IMAGE" \
    --audio "$AUDIO" \
    --video-frames 81 \
    -o "$OUTPUT" \
    -d 0 \
    --log-level warning

echo ""
echo "=== Output Info ==="
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 "$OUTPUT"
```

**预期输出**:
```
=== MultiTalk Benchmark ===
(实际耗时，预计 2-5 分钟 on RTX 4090)

=== Output Info ===
duration=8.000000
size=15728640
```

---

## Tests / Validation

### 单元测试策略

由于本项目没有现成的测试目录，采用以下验证方式：

**1. 编译验证（每次修改后）**:
```bash
cmake --build build --config Release -j8 2>&1 | grep -E "^.*error:" || echo "Build OK"
```

**2. 功能验证（每个 Task 完成后）**:
- 运行对应的测试命令
- 检查日志输出是否符合预期
- 验证输出文件有效性

**3. 回归测试（所有修改完成后）**:
```bash
# 测试现有 S2V 功能不受影响
./build/bin/sd-cli.exe -M vid_gen \
    --diffusion-model path/to/Wan2.2-S2V-14B-Q8.gguf \
    --audio-encoder path/to/wav2vec2.safetensors \
    -i path/to/ref.jpg --audio path/to/audio.wav \
    -o /tmp/s2v_test.mp4

# 验证输出
test -s /tmp/s2v_test.mp4 && echo "S2V regression: PASS" || echo "S2V regression: FAIL"
```

---

## Risks, Tradeoffs, and Open Questions

### 风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 模型权重不兼容 | 中 | 高 | Task 1 先验证键名结构 |
| VRAM 不足（14B 模型 ~28GB） | 高 | 高 | 使用 Q8_0 量化，或降级到 1.3B 模型 |
| 音频编码器不支持中文 | 中 | 中 | 尝试使用多语言 wav2vec2 模型 |
| 口型同步质量差 | 中 | 中 | 调整音频预处理参数 |

### 权衡

1. **通用性 vs 专用性**: 
   - 当前方案：添加通用 MultiTalk 支持
   - 备选：仅针对特定模型硬编码
   
2. **代码复杂度 vs 性能**:
   - 共享 S2V 代码 → 简单但可能有性能损失
   - 独立 MultiTalk 代码 → 复杂但可优化

### 待解决问题

1. **InfiniteTalk 的具体架构**: 需要 Task 1 的结果确认
2. **音频编码器选择**: wav2vec2 对中文效果未知
3. **是否需要自定义调度器**: 现有 S2V 调度器是否适用

---

## 实施顺序建议

```
Task 1 (检查模型) 
    ↓
Task 2 (直接测试兼容性)
    ↓ (如失败)
Task 3-7 (代码修改)
    ↓
Task 8-9 (CLI 集成 + 测试)
    ↓
Task 10 (性能基准)
```

**预计工作量**:
- Phase 1（验证）: 30 分钟
- Phase 2（代码修改）: 2-3 小时
- Phase 3（测试优化）: 1-2 小时
- **总计**: 4-6 小时

---

**计划文件**: `.hermes/plans/2026-09-30_091500-infini-talk-detailed.md`
