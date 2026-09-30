# 计划：在 stable-diffusion.cpp 中实现 MultiTalk/InfiniteTalk 数字人口型视频生成

## Goal

在 stable-diffusion.cpp 源码上二次开发，支持 InfiniTetalk 工作流（音频驱动的图片数字人口型同步视频生成），对应 ComfyUI 节点 `WanVideoImageToVideoMultiTalk`。

## Current Context / Assumptions

**现有状态**:
- 项目已支持 Wan2.1/2.2 S2V（Speech-to-Video）：使用 wav2vec2 音频编码器驱动口型
- 项目已有 `WanAudioInjector`（`src/model/diffusion/wan.hpp:625`）用于 S2V 的音频注入
- **MultiTalk 模型**（`Wan2_1-InfiniteTalk_Single_Q8.gguf`）是一个变体，使用不同的音频编码路径（可能包含额外的 embedding 层）
- ComfyUI 工作流包含：音频下载 → Whisper 转写 → Wav2Vec2 编码 → InfiniteTalk 生成视频

**关键假设**:
1. `Wan2_1-InfiniteTalk_Single_Q8.gguf` 是 Wan2.1-I2V 的变体，带有特殊的音频处理头
2. 需要验证模型权重结构（查看是否包含 `multi_talk_embeds` 或类似键）
3. 如果模型权重结构与标准 S2V 兼容，可能只需调整版本检测逻辑

## Architecture / Proposed Approach

stable-diffusion.cpp 采用分层架构：
- `src/model_loader.cpp` → 检测模型版本（基于权重键名）
- `src/pipeline/model_builders.cpp` → 根据版本创建对应的 runner
- `src/model/diffusion/wan.hpp` → Wan 模型核心逻辑（包括 `WanAudioInjector`）
- `src/pipeline/video.cpp` → 视频生成管道（音频编码、latents 构造、采样）
- `include/stable-diffusion.h` → 公共 C API

**方案**: 分两阶段进行
1. **快速验证**: 尝试直接运行现有 S2V 管线，检查是否能加载 InfiniteTalk 模型
2. **深度集成**: 如需要，添加新的版本类型 `VERSION_WAN2_MULTI_TALK` 和专用 runner

## Step-by-Step Tasks

### Task 1: 检查 InfiniteTalk 模型权重结构

**目的**: 确认模型与 S2V 的兼容性

```bash
# 先找到模型文件位置
find C:/Users/user/AppData/Local/hermes/attachments/ -name "*InfiniteTalk*" -o -name "*infinite_talk*" 2>/dev/null
find D:/workspace/ -name "*InfiniteTalk*" -o -name "*MULTITALK*" 2>/dev/null

# 使用 Python 读取 GGUF 模型的键（仅读，不加载）
python3 -c "
import struct
def read_gguf_keys(path):
    keys = []
    with open(path, 'rb') as f:
        header = f.read(12)
        n_kv, n_tensors = struct.unpack('<QQ', header)
        for _ in range(n_tensors):
            n_chars = struct.unpack('<I', f.read(4))[0]
            name = f.read(n_chars).decode('utf-8')
            n_dims, shape_len = struct.unpack('<II', f.read(8))
            shape = struct.unpack('<' + 'Q'*shape_len, f.read(8*shape_len))
            keys.append((name, shape))
    return keys

# 尝试从常见路径查找
import glob
for p in glob.glob('C:/Users/user/AppData/Local/**/*.gguf', recursive=True):
    if 'InfiniteTalk' in p or 'MULTITALK' in p.upper():
        print(f'Found: {p}')
        keys = read_gguf_keys(p)
        # 检查关键键
        audio_keys = [k for k in keys if 'audio' in k[0].lower() or 'injector' in k[0].lower() or 'talk' in k[0].lower()]
        for k, s in audio_keys[:20]:
            print(f'  {k}: {s}')
" 2>/dev/null
```

**预期输出**: 显示模型中的权重键，特别是 `audio_injector`、`wav2vec2`、`multi_talk` 相关键

### Task 2: 修改版本检测逻辑以识别 MultiTalk 模型

**文件**: `src/model_loader.cpp`

在 `detect_model_version()` 函数中添加 MultiTalk 检测。参考 S2V 检测逻辑（行 627-628）：

```cpp
// 在 is_s2v 检测附近添加
if (is_wan) {
    // 检查是否为 MultiTalk 模型
    bool has_multi_talk_embeds = false;
    bool has_casual_audio_encoder = false;
    for (const auto& [name, storage] : tensor_storage_map) {
        if (name.find("multi_talk_embeds") != std::string::npos) {
            has_multi_talk_embeds = true;
        }
        if (name.find("casual_audio_encoder") != std::string::npos) {
            has_casual_audio_encoder = true;
        }
    }
    if (has_multi_talk_embeds || has_casual_audio_encoder) {
        return VERSION_WAN2_MULTI_TALK;
    }
    // ... 原有 S2V/I2V/TI2V 检测逻辑
}
```

**文件**: `src/model.h`

在 `SDVersion` 枚举中添加新类型：

```cpp
// 在 VERSION_WAN2_2_S2V 后添加
VERSION_WAN2_MULTI_TALK,
```

**文件**: `src/model.h`

更新 `sd_version_is_wan()` 函数以包含新类型：

```cpp
static inline bool sd_version_is_wan(SDVersion version) {
    if (version == VERSION_WAN2 || version == VERSION_WAN2_2_I2V || 
        version == VERSION_WAN2_2_TI2V || version == VERSION_WAN2_2_S2V ||
        version == VERSION_WAN2_MULTI_TALK) {  // 添加这行
        return true;
    }
    return false;
}
```

### Task 3: 创建 MultiTalkRunner（如需要专用处理）

**文件**: `src/model/diffusion/wan_multi_talk.hpp`（新建）

如果 InfiniteTalk 模型有特殊的音频处理路径，创建专用 runner：

```cpp
#ifndef __SD_MODEL_DIFFUSION_WAN_MULTI_TALK_HPP__
#define __SD_MODEL_DIFFUSION_WAN_MULTI_TALK_HPP__

#include "model/diffusion/wan.hpp"

namespace WAN {

class MultiTalkRunner : public WanRunner {
public:
    MultiTalkRunner(ggml_backend_t backend,
                    const String2TensorStorage& tensor_storage_map,
                    const std::string prefix,
                    std::shared_ptr<RunnerWeightManager> weight_manager)
        : WanRunner(backend, tensor_storage_map, prefix, VERSION_WAN2_MULTI_TALK, weight_manager) {
        desc = "Wan2.1-MultiTalk";
    }
    
    std::string get_desc() override { return desc; }
    
    // 如果有特殊的 forward 逻辑，在此覆盖
};

} // namespace WAN

#endif // __SD_MODEL_DIFFUSION_WAN_MULTI_TALK_HPP__
```

### Task 4: 修改模型构建器以支持 MultiTalk

**文件**: `src/pipeline/model_builders.cpp`

在 Wan 检测分支中添加 MultiTalk 支持：

```cpp
} else if (sd_version_is_wan(version)) {
    result.conditioner = std::make_shared<T5CLIPEmbedder>(...);
    
    // 根据版本创建不同的 runner
    if (version == VERSION_WAN2_MULTI_TALK) {
        result.diffusion = std::make_shared<WAN::MultiTalkRunner>(...);
    } else {
        result.diffusion = std::make_shared<WAN::WanRunner>(...);
    }
    
    // ... 其余 Wan 处理逻辑保持不变
}
```

### Task 5: 修改视频生成管道以支持 MultiTalk 音频条件

**文件**: `src/pipeline/video.cpp`

在 `prepare_latents_video()` 函数中添加 MultiTalk 处理。参考 S2V 的处理（行 1182-1208）：

```cpp
// 在 S2V 处理之后添加 MultiTalk 处理
if (sd->version == VERSION_WAN2_MULTI_TALK) {
    embeds.cond.c_ref_images = latents.ref_latents;
    // MultiTalk 可能使用不同的音频 embedding 方式
    if (!latents.s2v_audio_embed.empty()) {
        embeds.cond.c_ref_audios = {latents.s2v_audio_embed};
    }
}
```

### Task 6: 修改 CLI 示例以支持 MultiTalk 参数

**文件**: `examples/cli/main.cpp`

添加 `--multi-talk` 或自动检测参数。或者在生成参数中添加：

```cpp
// 在参数解析部分
bool multi_talk_mode = false;
// ... 解析 --multi-talk 标志
if (multi_talk_mode) {
    ctx.params.model_args["multi_talk"] = "1";
}
```

### Task 7: 构建并测试

**命令**:
```bash
cd D:/workspace/stable-diffusion.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8
```

**测试命令**:
```bash
./build/bin/sd-cli.exe -M vid_gen \
    --diffusion-model path/to/Wan2_1-InfiniteTalk_Single_Q8.gguf \
    --audio-encoder path/to/wav2vec2_large_english_fp16.safetensors \
    -i path/to/reference.jpg \
    --audio path/to/speech.wav \
    --video-frames 81 \
    -o output.mp4
```

**预期输出**: 成功生成视频，日志显示 `Wan2.1-MultiTalk` 模型加载成功

### Task 8: 集成音频预处理（可选）

如果用户需要从 ComfyUI 工作流直接调用，可以封装一个 Python 脚本：

**文件**: `scripts/infini_talk.py`（新建）

```python
#!/usr/bin/env python3
"""
InfiniTetalk 工作流封装脚本
将 ComfyUI 工作流的核心步骤迁移到 stable-diffusion.cpp
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='InfiniTetalk 数字人口型视频生成')
    parser.add_argument('--model', required=True, help='MultiTalk GGUF 模型路径')
    parser.add_argument('--audio-encoder', required=True, help='wav2vec2 音频编码器路径')
    parser.add_argument('--image', required=True, help='参考图片路径')
    parser.add_argument('--audio', required=True, help='驱动音频路径')
    parser.add_argument('--output', default='output.mp4', help='输出视频路径')
    parser.add_argument('--frames', type=int, default=81, help='视频帧数')
    parser.add_argument('--sd-cli', default='./build/bin/sd-cli.exe', help='sd-cli 路径')
    
    args = parser.parse_args()
    
    cmd = [
        args.sd_cli, '-M', 'vid_gen',
        '--diffusion-model', args.model,
        '--audio-encoder', args.audio_encoder,
        '-i', args.image,
        '--audio', args.audio,
        '--video-frames', str(args.frames),
        '-o', args.output
    ]
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
```

## Tests / Validation

**测试策略**: TDD（测试驱动开发）

1. **单元测试**: 验证版本检测逻辑
   - 创建 `test/test_multi_talk_detection.cpp`
   - 测试不同模型权重键组合的版本检测

2. **集成测试**: 端到端视频生成
   - 准备测试数据（图片 + 音频）
   - 运行完整生成流程
   - 验证输出视频有效（格式、时长）

3. **回归测试**: 确保不影响现有功能
   - 运行现有的 S2V 测试
   - 运行 I2V/T2V 测试
   - 确认无性能退化

**验证命令**:
```bash
# 构建测试
cmake --build build --target tests --config Release

# 运行测试
./build/bin/tests
```

## Risks, Tradeoffs, and Open Questions

**风险**:
1. **模型不兼容**: InfiniteTalk 可能使用完全不同的架构，无法通过简单扩展支持
2. **性能问题**: 14B 模型需要大量 VRAM（~28GB for Q8_0），可能需要量化或分布式推理
3. **音频质量**: wav2vec2 编码器可能对中文语音识别效果有限

**权衡**:
1. **快速验证 vs 深度集成**: 先尝试现有 S2V 管线能否直接工作，再决定是否需要深度修改
2. **通用性 vs 专用性**: 是否设计为通用 MultiTalk 框架，还是仅支持特定模型

**待解决问题**:
1. `Wan2_1-InfiniteTalk_Single_Q8.gguf` 的具体权重结构未知
2. MultiTalk 模型是否需要额外的训练数据或配置文件
3. 是否支持实时流式生成（ComfyUI 工作流似乎是离线批量生成）

**下一步行动**:
1. 首先运行 Task 1 检查模型文件
2. 根据模型结构决定实施路径
3. 如果兼容 S2V，只需修改版本检测（Task 2-3）
4. 如果不兼容，需要评估重写多少代码

---

**计划文件路径**: `.hermes/plans/2026-09-30_091500-infini-talk-migration.md`
