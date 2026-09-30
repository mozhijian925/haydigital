# 计划：为 stable-diffusion.cpp InfiniTetalk 二次开发建立版本管理体系

## Goal

在 stable-diffusion.cpp 项目上建立完整的版本管理流程，支持 InfiniTetalk 数字人功能的二次开发，确保代码可追溯、可回滚、可协作。

---

## Current Context / Assumptions

### 当前状态

- **项目根目录**: `D:/workspace/stable-diffusion.cpp`
- **Git 仓库**: ❌ 尚未初始化（`fatal: not a git repository`）
- **已有文件**:
  - `.gitignore` ✅ 已存在（排除 build/、models/、*.exe 等）
  - `.hermes/plans/` ✅ 已有计划文件
  - `src/`、`include/`、`examples/`、`docs/` 等标准结构
- **外部依赖**:
  - `ggml/` — git submodule（需单独初始化）
  - `thirdparty/` — vendored 依赖
  - `examples/server/frontend/` — git submodule

### 假设

1. 用户希望在本地建立完整的 Git 版本管理
2. 需要支持多人协作（未来可能 push 到 GitHub）
3. 需要分支策略支持功能开发（InfiniTalk）与主线的分离
4. 遵循项目现有的 [Conventional Commit](CONTRIBUTING.md) 规范

---

## Architecture / Proposed Approach

**方案**: 初始化 Git 仓库 + 建立分支策略 + 配置钩子

```
stable-diffusion.cpp/
├── .git/                  # 新建：版本控制
├── .gitignore            # 已有：排除构建产物
├── .github/              # 新建：CI/PR 模板
│   ├── pull_request_template.md
│   └── workflows/
│       └── build.yml      # 自动构建检查
├── .husky/               # 新建：pre-commit 钩子
│   └── pre-commit         # 自动格式化
├── src/                   # 源代码
├── include/              # 公共头文件
├── examples/             # 示例代码
├── docs/                 # 文档
├── scripts/              # 工具脚本
├── .hermes/plans/        # AI 助手计划
└── models/               # 模型文件（.gitignore 排除）
```

**分支策略**:
- `main` — 稳定主线（上游同步）
- `feature/infini-talk` — InfiniTalk 功能开发分支
- `hotfix/*` — 紧急修复

---

## Step-by-Step Tasks

### Task 1: 初始化 Git 仓库

**目标**: 将项目纳入版本控制

**操作**:

```bash
# 进入项目目录
cd D:/workspace/stable-diffusion.cpp

# 初始化 Git 仓库
git init

# 添加远程仓库（可选，后续配置）
# git remote add origin https://github.com/your-org/stable-diffusion.cpp.git

# 创建主分支
git checkout -b main

# 添加所有文件
git add .

# 初始提交
git commit -m "chore: initial commit of stable-diffusion.cpp with InfiniTalk support"

# 验证
git log --oneline -5
git status
```

**预期输出**:
```
[main (root-commit) abc1234] chore: initial commit of stable-diffusion.cpp with InfiniTalk support
 199 files changed, 45678 insertions(+)
 create mode 100644 CMakeLists.txt
 create mode 100644 README.md
 ...

abc1234 chore: initial commit of stable-diffusion.cpp with InfiniTalk support
HEAD@{0}: commit (initial): chore: initial commit of stable-diffusion.cpp with InfiniTalk support

On branch main
nothing to commit, working tree clean
```

---

### Task 2: 初始化 Submodule

**目标**: 管理外部依赖（ggml、frontend）

**操作**:

```bash
# 初始化 ggml submodule
git submodule update --init --recursive ggml

# 初始化 frontend submodule（如果存在）
git submodule update --init --recursive examples/server/frontend

# 验证 submodule 状态
git submodule status
```

**预期输出**:
```
 abc1234567890abcdef1234567890abcdef12 (v0.1.0)
  subdir examples/server/frontend (v1.0.0)
```

---

### Task 3: 创建功能分支

**目标**: 隔离 InfiniTalk 开发工作

**操作**:

```bash
# 从 main 创建功能分支
git checkout -b feature/infini-talk

# 验证分支创建
git branch
git branch -v
```

**预期输出**:
```
  main
* feature/infini-talk
  main abc1234 chore: initial commit...
* feature/infini-talk abc1234 chore: initial commit...
```

---

### Task 4: 配置 Git 钩子（自动格式化）

**目标**: 提交前自动格式化代码

**新建文件**: `.husky/pre-commit`

```bash
#!/bin/sh
# Pre-commit hook: format code before commit

echo "Running code formatter..."

# 运行项目的格式化脚本
if [ -f "scripts/format-code.sh" ]; then
    bash scripts/format-code.sh
elif [ -f "scripts/format-code.ps1" ]; then
    powershell -ExecutionPolicy Bypass -File scripts/format-code.ps1
fi

# 添加格式化后的文件
git add -u

echo "Formatting complete."
```

**设置执行权限**:
```bash
chmod +x .husky/pre-commit
```

**验证**:
```bash
# 测试钩子
.husky/pre-commit
# 预期：无错误输出
```

---

### Task 5: 创建 GitHub 工作流（CI/CD）

**新建文件**: `.github/workflows/build.yml`

```yaml
name: Build and Test

on:
  push:
    branches: [main, feature/*]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: windows-latest

    steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive

    - name: Configure CMake
      run: cmake -B ${{github.workspace}}/build -DCMAKE_BUILD_TYPE=Release

    - name: Build
      run: cmake --build ${{github.workspace}}/build --config Release -j8

    - name: Run tests
      run: cd ${{github.workspace}}/build && ctest --output-on-failure
```

**验证**:
```bash
# 检查 YAML 语法
python -c "import yaml; yaml.safe_load(open('.github/workflows/build.yml'))"
echo "✓ YAML syntax valid"
```

---

### Task 6: 创建 PR 模板

**新建文件**: `.github/pull_request_template.md`

```markdown
## Description

<!-- 描述变更内容 -->

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

<!-- 描述测试方法和结果 -->

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated (if needed)
- [ ] Tests added/updated
```

---

### Task 7: 创建版本标签策略

**目标**: 定义版本号规范

**新建文件**: `VERSION`

```
0.1.0-infini-talk-dev
```

**版本号规范** (Semantic Versioning):
- `MAJOR.MINOR.PATCH`
- `MAJOR`: 不兼容的 API 修改
- `MINOR`: 向后兼容的功能新增
- `PATCH`: 向后兼容的问题修正

**标签创建示例**:
```bash
# 创建版本标签
git tag -a v0.1.0 -m "Initial InfiniTalk support"

# 查看标签
git tag -l
git show v0.1.0
```

---

### Task 8: 创建开发环境配置

**新建文件**: `.vscode/settings.json`

```json
{
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "cmake.configureOnOpen": true,
    "C_Cpp.default.compilerPath": "C:/Program Files/mingw-w64/x86_64-13.1.0-posix-seh-ucrt-rt_v10-rev0/mingw64/bin/g++.exe",
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-vscode.cpptools"
}
```

**新建文件**: `.vscode/tasks.json`

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "CMake: Configure",
            "type": "shell",
            "command": "cmake",
            "args": ["-B", "${workspaceFolder}/build", "-DCMAKE_BUILD_TYPE=Release"],
            "group": "build"
        },
        {
            "label": "CMake: Build",
            "type": "shell",
            "command": "cmake",
            "args": ["--build", "${workspaceFolder}/build", "--config", "Release", "-j8"],
            "group": "build"
        },
        {
            "label": "Run sd-cli",
            "type": "shell",
            "command": "${workspaceFolder}/build/bin/sd-cli.exe",
            "problemMatcher": []
        }
    ]
}
```

---

### Task 9: 创建开发工作流文档

**新建文件**: `docs/DEVELOPMENT_WORKFLOW.md`

```markdown
# 开发工作流

## 分支策略

```
main ← feature/infini-talk
       ↑
       hotfix/urgent-fix
```

- `main`: 稳定版本，仅接受 PR
- `feature/*`: 功能开发分支
- `hotfix/*`: 紧急修复分支

## 提交规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add MultiTalk model support
fix: correct audio timestamp alignment
docs: update InfiniteTalk integration guide
refactor: extract audio chunking logic
test: add chunked video generation tests
```

## 开发流程

1. 创建功能分支: `git checkout -b feature/your-feature`
2. 频繁提交: `git commit -m "feat: small incremental change"`
3. 定期同步: `git pull origin main`
4. 提交 PR: 描述变更 + 测试验证
5. 代码审查: 响应 review 意见
6. 合并: squash merge 到 main

## 版本发布

```bash
# 1. 切换到 main 并拉取
git checkout main
git pull origin main

# 2. 创建版本标签
git tag -a v0.2.0 -m "InfiniteTalk support with chunked generation"

# 3. 推送标签
git push origin v0.2.0
```
```

---

### Task 10: 首次功能提交

**目标**: 提交第一个 InfiniTalk 相关代码

**操作流程**:

```bash
# 确保在功能分支
git branch  # 应显示 * feature/infini-talk

# 查看状态
git status

# 添加新文件
git add src/audio/audio_chunker.hpp
git add src/audio/audio_chunker.cpp
git add src/pipeline/video_chunked.hpp
git add src/pipeline/video_chunked.cpp

# 提交
git commit -m "feat: add chunked video generation for infinite-length talking head"

# 查看提交历史
git log --oneline -3
```

**预期输出**:
```
abc1234 feat: add chunked video generation for infinite-length talking head
def5678 chore: initial commit...
```

---

## Tests / Validation

### Git 仓库验证

```bash
# 1. 验证仓库状态
cd D:/workspace/stable-diffusion.cpp
git status
# 预期：no commits yet 或 clean working tree

# 2. 验证分支
git branch -a
# 预期：* main

# 3. 验证日志
git log --oneline
# 预期：显示初始提交

# 4. 验证 submodule
git submodule status
# 预期：显示 ggml 和 frontend 状态
```

### 工作流验证

```bash
# 测试分支创建和切换
git checkout -b test-branch
git checkout main
git branch -d test-branch

# 测试提交流程
echo "test" > test_file.txt
git add test_file.txt
git commit -m "test: verify workflow"
git reset --hard HEAD~1
rm test_file.txt
```

### CI 验证（可选）

```bash
# 本地测试 CI 配置
docker run --rm -v ${PWD}:/repo -w /repo actions/checkout:main git status
# 预期：显示仓库文件列表
```

---

## Risks, Tradeoffs, and Open Questions

### 风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 大文件提交导致仓库膨胀 | 中 | 高 | 严格使用 .gitignore，大模型文件用 LFS |
| Submodule 同步问题 | 中 | 中 | 文档化 submodule 更新流程 |
| 分支策略混乱 | 低 | 中 | 明确的分支命名规范 + 文档 |
| 历史重写冲突 | 低 | 高 | 避免 force push，使用 merge |

### 权衡

1. **Git LFS vs 排除大文件**:
   - LFS: 支持大文件，但需要额外配置
   - 排除: 简单，但无法追踪模型变更
   
   **建议**: 模型文件放入 `.gitignore`，不提交到仓库

2. **Monorepo vs 多仓库**:
   - Monorepo: 单一版本，易于管理
   - 多仓库: 独立发布，但同步复杂
   
   **建议**: 使用 monorepo，InfiniTalk 作为功能分支

3. **Squash Merge vs Merge Commit**:
   - Squash: 干净历史，但丢失细节
   - Merge: 保留完整历史，但历史复杂
   
   **建议**: 功能分支使用 squash merge 到 main

### 待解决问题

1. **远程仓库**: 是否需要创建 GitHub 仓库？用户组织/个人账号？
2. **CI/CD**: 是否需要自动化测试和部署？
3. **协作流程**: 是否需要 code review 流程？
4. **Issue 追踪**: 是否集成 GitHub Issues？

---

## 实施顺序

```
Task 1 (初始化 Git) 
    ↓
Task 2 (Submodule)
    ↓
Task 3 (功能分支)
    ↓
Task 4-6 (配置钩子 + CI)
    ↓
Task 7-9 (版本策略 + 文档)
    ↓
Task 10 (首次提交)
```

**预计工作量**: 30-60 分钟

---

## 后续步骤

完成版本管理后，可以：

1. 开始 InfiniTalk 功能开发（参考 `2026-09-30_091500-infini-talk-full-support.md`）
2. 每完成一个功能模块提交一次
3. 定期同步上游 main 分支
4. 发布测试版本标签

---

**计划文件**: `.hermes/plans/2026-09-30_091500-version-management.md`
