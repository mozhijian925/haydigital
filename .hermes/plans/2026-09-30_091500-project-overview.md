# stable-diffusion.cpp 项目概览与导航计划

## Goal

阅读并理解 stable-diffusion.cpp 项目架构，为后续可能的代码修改、功能扩展或问题排查建立完整的上下文认知。

## Current Context

- **项目**: stable-diffusion.cpp - 纯 C++ 实现的扩散模型推理引擎
- **技术栈**: C++17, CMake 构建系统, ggml 张量计算库
- **规模**: ~200 个源文件，~50,000+ 行 C++ 代码
- **目标用户**: AI 编程代理（需要零上下文也能理解项目）

## Architecture / Proposed Approach

项目采用分层架构：C API → 核心引擎（StableDiffusionGGML）→ 模型组件 → ggml 计算层。核心流程是：加载模型权重 → 构建推理图 → 执行扩散采样 → 解码输出。采用模块化设计，支持 SD1.x/SD2.x/SDXL/SD3/FLUX/Wan/Qwen 等多种模型架构。

## Step-by-Step Tasks

### 任务 1: 理解顶层入口与 C API

**文件**: `include/stable-diffusion.h`, `src/stable-diffusion.cpp`

**目标**: 理解公共 API 接口和数据结构

**关键实体**:
- `sd_ctx_t` - 上下文句柄，封装 `StableDiffusionGGML*`
- `sd_ctx_params_t` - 模型加载配置（40+ 字段）
- `sd_img_gen_params_t` / `sd_vid_gen_params_t` - 图像/视频生成参数
- `generate_image()` / `generate_video()` - 主要生成函数

**阅读重点**:
```cpp
// src/stable-diffusion.cpp:632-654
struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    // 创建上下文，初始化模型
}
```

**验证命令**:
```bash
grep -n "SD_API" include/stable-diffusion.h | head -30
# 应输出约 30 个公共 API 函数声明
```

---

### 任务 2: 理解核心引擎 StableDiffusionGGML

**文件**: `src/pipeline/diffusion_engine.h`, `src/pipeline/diffusion_engine.cpp`

**目标**: 掌握模型生命周期管理

**关键方法** (按初始化顺序):
1. `init(const sd_ctx_params_t*)` - 整体初始化入口
2. `init_backend()` - 初始化 ggml 后端（CPU/CUDA/Metal/Vulkan等）
3. `init_model_loader()` - 初始化模型加载器
4. `build_core_runners()` - 构建核心推理组件（Clip/Diffusion/Vae）
5. `validate_and_load_runners()` - 验证并加载权重

**关键成员**:
```cpp
class StableDiffusionGGML {
    SDBackendManager backend_manager;  // 后端管理
    SDVersion version;                 // 模型版本
    std::shared_ptr<Conditioner> cond_stage_model;  // 文本编码器
    std::shared_ptr<DiffusionModelRunner> diffusion_model;  // 扩散模型
    std::shared_ptr<VAE> first_stage_model;         // VAE 编解码
    std::shared_ptr<ModelManager> model_manager;    // 权重管理
};
```

**验证**:
```bash
wc -l src/pipeline/diffusion_engine.cpp
# 约 140KB, 1400+ 行，核心实现
```

---

### 任务 3: 理解模型版本体系

**文件**: `src/model.h`

**目标**: 掌握支持的模型类型和版本判断

**版本枚举** (重要分类):
```cpp
enum SDVersion {
    // SD1/SD2/SDXL 系列
    VERSION_SD1, VERSION_SD2, VERSION_SDXL, ...
    // SD3 系列
    VERSION_SD3, ...
    // FLUX 系列
    VERSION_FLUX, VERSION_FLUX_FILL, VERSION_FLUX2, ...
    // 视频模型
    VERSION_WAN2, VERSION_LTXAV, VERSION_MINIMAX_H3, ...
    // 其他新模型
    VERSION_QWEN_IMAGE, VERSION_LENS, VERSION_IDEOGRAM4, ...
};
```

**辅助函数** (用于条件编译):
- `sd_version_is_sd1()` / `sd_version_is_flux()` / `sd_version_is_wan()` 等
- `sd_version_supports_image_generation()` / `sd_version_supports_video_generation()`

**验证**:
```bash
grep -c "VERSION_" src/model.h
# 约 60+ 个版本常量
```

---

### 任务 4: 理解模型加载与权重管理

**文件**: `src/model_loader.h/cpp`, `src/model_manager.h/cpp`

**目标**: 掌握权重加载、格式支持、内存管理

**ModelLoader** (`src/model_loader.h`):
- 支持格式：GGUF、Safetensors、PyTorch (.ckpt/.pth/.pt)
- 文件哈希与缓存机制
- 张量名称转换（name conversion）

**ModelManager** (`src/model_manager.h`):
- 权重驻留管理（DeviceResidencyManager）
- 支持 mmap 映射
- LoRA 应用管理
- 多设备分割支持

**关键流程**:
```
add_file() → parse_file() → load_tensors() → mmap_tensors()
```

**验证**:
```bash
ls -lh src/model_loader.cpp src/model_manager.cpp
# model_loader.cpp ~83KB, model_manager.cpp ~79KB
```

---

### 任务 5: 理解模型组件架构

**目录**: `src/model/diffusion/`

**目标**: 了解各模型架构实现

**核心组件**:
```
src/model/diffusion/
├── unet.hpp        # SD1/SD2/SDXL UNet
├── flux.hpp        # FLUX DiT
├── mmdit.hpp       # SD3 MM-DiT
├── wan.hpp         # Wan 视频模型
├── qwen_image.hpp  # Qwen Image
└── ltxv.hpp        # LTX-Video
```

**每个组件包含**:
- 模型定义类（继承 `DiffusionModelRunner`）
- 前向传播方法 `forward()`
- 权重注册方法

**验证**:
```bash
ls src/model/diffusion/*.hpp | wc -l
# 约 30 个模型架构实现
```

---

### 任务 6: 理解推理流程

**文件**: `src/pipeline/image.cpp`, `src/pipeline/video.cpp`

**目标**: 掌握图像/视频生成流程

**图像生成流程** (`image.cpp` ~53KB):
```
generate_image() 
  → get_learned_condition()           # 文本编码
  → generate_init_latent()            # 生成初始噪声
  → sample() [循环 steps 次]          # 扩散去噪
    → diffusion_model->forward()     # 模型推理
    → denoiser->denoise()            # 去噪器更新
  → decode_first_stage()              # VAE 解码
  → 返回 sd_image_t
```

**视频生成流程** (`video.cpp` ~108KB):
- 类似图像流程，增加时间维度
- 支持条件参考图像/视频
- 支持音频条件输入

**验证**:
```bash
grep -n "^bool generate_" src/pipeline/image.cpp src/pipeline/video.cpp
# image.cpp:228 bool generate_image(...)
# video.cpp:567 bool generate_video(...)
```

---

### 任务 7: 理解核心计算层

**目录**: `src/core/`

**关键文件**:
- `tensor.hpp` (~76KB) - 张量抽象层，封装 ggml_tensor
- `ggml_runner.h/cpp` (~42KB) - 计算图执行器
- `ggml_extend.h/cpp` (~34KB) - ggml 扩展操作
- `backend_fit.h/cpp` (~9KB) - 后端自动适配
- `util.h/cpp` (~76KB) - 工具函数（图像处理、矩阵运算等）

**张量布局注意** (来自 AGENTS.md):
```cpp
// shape()[0] 是最内层连续维度（与 PyTorch 相反）
// 广播对齐从低维到高维
// 内部 API 用空 sd::Tensor 表示 null/失败
```

**验证**:
```bash
ls -lh src/core/*.hpp src/core/*.h src/core/*.cpp | wc -l
# 约 30 个核心实现文件
```

---

### 任务 8: 理解条件编码

**目录**: `src/conditioning/`
**文件**: `src/tokenizers/*.cpp`

**目标**: 掌握文本编码流程

**流程**:
```
prompt string 
  → Tokenizer (BPE/Unigram/HF) 
  → Conditioner (CLIP/T5/Gemma) 
  → SDCondition (cond + uncond tensors)
```

**Tokenizers** (`src/tokenizers/`):
- `bpe_tokenizer.cpp` - SD1/SD2/SDXL CLIP
- `t5_unigram_tokenizer.cpp` - SD3/Wan T5
- `hf_tokenizer.cpp` - HuggingFace 兼容
- `gemma_tokenizer.cpp` - FLUX Gemma

**验证**:
```bash
ls src/tokenizers/*.cpp
# 约 12 个 tokenizer 实现
```

---

### 任务 9: 理解采样器与调度器

**文件**: `src/runtime/denoiser.hpp` (~120KB)

**目标**: 掌握扩散采样算法

**采样方法** (来自 `stable-diffusion.h`):
```cpp
enum sample_method_t {
    EULER_SAMPLE_METHOD,       // 欧拉
    EULER_A_SAMPLE_METHOD,     // 欧拉各向异性
    HEUN_SAMPLE_METHOD,        // Heun
    DPMPP2M_SAMPLE_METHOD,     // DPM++ 2M
    LCM_SAMPLE_METHOD,         // LCM
    // ... 约 20 种采样方法
};
```

**调度器**:
```cpp
enum scheduler_t {
    DISCRETE_SCHEDULER,        // 离散
    KARRAS_SCHEDULER,          // Karras (SDXL推荐)
    EXPONENTIAL_SCHEDULER,     // 指数
    // ... 约 17 种调度器
};
```

**验证**:
```bash
grep "enum sample_method_t" src/stable-diffusion.cpp
# 检查 sample_method_to_str 数组初始化
```

---

### 任务 10: 理解示例程序

**目录**: `examples/cli/main.cpp` (~40KB)

**目标**: 理解命令行工具用法

**关键功能**:
- 参数解析（使用 `argparse` 或类似库）
- 批量生成支持
- 预览回调集成
- 元数据嵌入

**验证**:
```bash
head -50 examples/cli/main.cpp
tail -100 examples/cli/main.cpp
```

---

## Tests / Validation

本项目没有传统单元测试，验证方式：

1. **构建验证**:
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
# 应无编译错误
```

2. **运行示例**:
```bash
./bin/sd-cli -m model.safetensors -p "a cat" --steps 20
# 应生成 output.png
```

3. **API 兼容性检查**:
```bash
grep "SD_API" include/stable-diffusion.h | wc -l
# 记录当前 API 数量，修改后验证未减少
```

---

## Risks, Tradeoffs, and Open Questions

### 已知风险
1. **版本兼容**: `src/model.h` 中 `SDVersion` 枚举增长迅速，新模型需新增枚举值
2. **内存管理**: `ModelManager` 的权重驻留逻辑复杂，多设备场景易出错
3. **张量布局**: `sd::Tensor` 维度顺序与 PyTorch 相反，容易混淆

### 设计权衡
1. **C API 优先**: 公共接口保持 C 兼容，但内部使用 C++17
2. **模块化**: 模型组件完全解耦，但增加了代码复杂度
3. **后端抽象**: `SDBackendManager` 统一了多设备支持，但调试困难

### 开放问题
1. 新模型添加标准流程？
2. `src/name_conversion.cpp` (~82KB) 的转换规则如何维护？
3. 测试覆盖率如何评估？

---

## Quick Reference: 关键文件速查表

| 功能 | 文件路径 | 大小 |
|------|----------|------|
| 公共 API | `include/stable-diffusion.h` | 20KB |
| 核心引擎 | `src/pipeline/diffusion_engine.h/cpp` | 20KB / 141KB |
| 模型版本 | `src/model.h` | 10KB |
| 模型加载 | `src/model_loader.h/cpp` | 7KB / 67KB |
| 权重管理 | `src/model_manager.h/cpp` | 15KB / 79KB |
| 图像生成 | `src/pipeline/image.cpp` | 53KB |
| 视频生成 | `src/pipeline/video.cpp` | 108KB |
| 张量抽象 | `src/core/tensor.hpp` | 76KB |
| 采样器 | `src/runtime/denoiser.hpp` | 120KB |
| 模型组件 | `src/model/diffusion/*.hpp` | ~30 个文件 |
| 命令行示例 | `examples/cli/main.cpp` | 40KB |

---

## Next Steps for Implementation

如果需要添加新功能：

1. **新模型支持**: 在 `src/model/diffusion/` 添加新 `.hpp`，实现 `DiffusionModelRunner` 接口
2. **新功能参数**: 修改 `sd_ctx_params_t` 或 `sd_img_gen_params_t`，更新 `src/stable-diffusion.cpp`
3. **后端适配**: 修改 `src/core/ggml_extend_backend.cpp`
4. **文档更新**: 更新 `docs/` 下对应模型文档
