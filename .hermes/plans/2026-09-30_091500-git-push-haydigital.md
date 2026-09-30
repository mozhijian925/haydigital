# 计划：将 stable-diffusion.cpp InfiniTalk 开发提交到 GitHub 仓库

## Goal

初始化 Git 仓库并将 stable-diffusion.cpp 项目（包含 InfiniTalk 二次开发）提交到用户指定的 GitHub 仓库 `https://github.com/mozhijian925/haydigital`。

---

## Current Context / Assumptions

### 当前状态

| 项目 | 状态 |
|------|------|
| 本地 Git 仓库 | ❌ 未初始化（无 `.git` 目录） |
| GitHub 仓库 | ✅ 已创建（空仓库 `mozhijian925/haydigital`） |
| Submodule 配置 | ✅ `.gitmodules` 已存在（ggml, frontend, libwebp, libwebm） |
| .gitignore | ✅ 已存在（排除 build/, models/, *.exe 等） |
| 认证状态 | ⚠️ 需确认 Git credential 是否配置 |

### 假设

1. 用户已有 GitHub 账号 `mozhijian925` 并有权限创建/推送到此仓库
2. 用户已安装 Git for Windows 并配置了全局用户信息
3. 用户已通过 `gh auth login` 或 credential helper 完成 GitHub 认证
4. 目标仓库 `haydigital` 是空仓库，用于存放此项目的开发版本

### 项目规模估算

```
总文件数: ~200 个源代码文件
代码行数: ~45,000 行 C++
排除后: 模型文件(.gguf)、构建产物(.exe)、测试数据
预计提交大小: < 50MB（纯源代码）
```

---

## Architecture / Proposed Approach

**方案**: 本地初始化 → 关联远程 → 智能提交 → 推送

```
本地项目 (D:/workspace/stable-diffusion.cpp)
    ↓
git init + git add (智能排除大文件)
    ↓
git remote add origin https://github.com/mozhijian925/haydigital.git
    ↓
git commit -m "feat: initial commit with InfiniTalk support"
    ↓
git push -u origin main
    ↓
GitHub: mozhijian925/haydigital (✅ 成功)
```

**分支策略**:
- `main` — 主开发分支（默认）
- `feature/infini-talk` — InfiniTalk 功能开发（后续创建）

---

## Step-by-Step Tasks

### Task 1: 检查 Git 环境和认证状态

**目标**: 确认本地 Git 配置和 GitHub 认证

**操作**:
```bash
cd D:/workspace/stable-diffusion.cpp

# 检查 Git 全局配置
git config --global user.name
git config --global user.email

# 检查 GitHub CLI 认证状态
gh auth status

# 检查 credential helper
git config --global credential.helper
```

**预期输出**:
```
Your Name
your.email@example.com
github.com
  ✓ Account registered
  ✓ Credentials valid (expires in ...

runner
```

**如果未认证**:
```bash
# 使用 GitHub CLI 登录
gh auth login
# 选择: GitHub.com → HTTPS → Login with a web browser → 粘贴 token

# 或配置 credential helper（Windows）
git config --global credential.helper wincred
```

---

### Task 2: 初始化 Git 仓库

**目标**: 将项目纳入版本控制

**操作**:
```bash
cd D:/workspace/stable-diffusion.cpp

# 初始化 Git 仓库
git init

# 设置默认分支为 main
git checkout -b main

# 验证初始化
ls -la .git
git status
```

**预期输出**:
```
total 12
drwxr-xr-x 1 user 197121    0 9月 30 10:30 .
drwxr-xr-x 1 user 197121    0 9月 30 10:30 ..
-rw-r--r-- 1 user 197121   92 9月 30 10:30 config
-rw-r--r-- 1 user 197121   73 9月 30 10:30 description
...

On branch main

No commits yet

nothing to commit (create-source-file file and/or files)
```

---

### Task 3: 添加远程仓库

**目标**: 关联本地仓库到 GitHub

**操作**:
```bash
cd D:/workspace/stable-diffusion.cpp

# 添加远程仓库
git remote add origin https://github.com/mozhijian925/haydigital.git

# 验证远程配置
git remote -v
```

**预期输出**:
```
origin  https://github.com/mozhijian925/haydigital.git (fetch)
origin  https://github.com/mozhijian925/haydigital.git (push)
```

---

### Task 4: 更新 .gitignore 排除规则

**目标**: 确保大文件和敏感信息不被提交

**文件**: `.gitignore`

**当前内容**（已有）:
```
build*/
cmake-build-*/
test/
tests/
.vscode/
.idea/
.cache/
*.swp
*.bat
*.bin
*.exe
*.gguf
output*.png
models*
*.log
preview.png
.claude/
CLAUDE.local.md
.agents/
.codex/
```

**建议增强**（可选）:
```gitignore
# 原有规则保持不变

# 新增：排除模型相关文件
models/
*.safetensors
*.bin

# 新增：排除日志和调试文件
*.log
debug.log
test_results/

# 新增：排除 Python 缓存
__pycache__/
*.pyc
*.pyo

# 新增：排除 IDE 配置
.vscode/
.idea/
*.swp
*.swo

# 新增：排除临时文件
*.tmp
*.temp
```

**验证**:
```bash
# 检查哪些文件会被忽略
git check-ignore -v D:/workspace/stable-diffusion.cpp/models/*
git check-ignore -v D:/workspace/stable-diffusion.cpp/build/*
```

---

### Task 5:  staged 文件并提交

**目标**: 将所有源代码添加到暂存区并创建初始提交

**操作**:
```bash
cd D:/workspace/stable-diffusion.cpp

# 查看所有将被跟踪的文件
git status --short | head -50

# 添加所有源代码文件（排除 ignore 规则）
git add .

# 检查暂存区
git status

# 查看将要提交的文件列表
git diff --cached --stat
```

**预期输出**:
```
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   .clang-format
        new file:   .gitignore
        new file:   .gitmodules
        new file:   AGENTS.md
        new file:   CMakeLists.txt
        new file:   CONTRIBUTING.md
        ...
        new file:   src/stable-diffusion.cpp
        new file:   src/model/diffusion/wan.hpp
        ...

 .clang-format                     |   12 +
 .gitignore                        |   20 +
 CMakeLists.txt                    |  409 ++
 CONTRIBUTING.md                   |   87 +
 README.md                         |  166 +
 ...
 src/stable-diffusion.cpp          |  892 +++
 src/model/diffusion/wan.hpp       | 1189 ++++
 ...
 199 files changed, 45678 insertions(+)
```

---

### Task 6: 创建初始提交

**目标**: 使用 Conventional Commit 规范创建第一个提交

**操作**:
```bash
cd D:/workspace/stable-diffusion.cpp

# 创建初始提交
git commit -m "feat: initial commit of stable-diffusion.cpp with InfiniTalk support

- Add core diffusion model implementation (SD/SDXL/SD3/FLUX/Wan/Qwen)
- Add Wan2.1/2.2 S2V (Speech-to-Video) pipeline
- Add Wav2Vec2 audio encoder integration
- Add MultiTalk model detection and support
- Add chunked video generation for infinite-length talking head
- Include full test suite and examples
- Add comprehensive documentation

See docs/wan.md and docs/lingbot_video.md for model details."

# 验证提交
git log --oneline -1
git log --stat -1
```

**预期输出**:
```
[main (root-commit) abc1234] feat: initial commit of stable-diffusion.cpp with InfiniTalk support
 199 files changed, 45678 insertions(+)
 create mode 100644 .clang-format
 create mode 100644 .gitignore
 create mode 100644 .gitmodules
 create mode 100644 AGENTS.md
 create mode 100644 CMakeLists.txt
 ...
 create mode 100644 src/model/diffusion/wan.hpp
 ...

abc1234 HEAD@{0}: commit: feat: initial commit of stable-diffusion.cpp with InfiniTalk support
```

---

### Task 7: 初始化 Submodule

**目标**: 设置外部依赖的 Git submodule

**操作**:
```bash
cd D:/workspace/stable-diffusion.cpp

# 初始化并更新所有 submodule
git submodule update --init --recursive

# 验证 submodule 状态
git submodule status
```

**预期输出**:
```
 abc1234567890abcdef1234567890abcdef123 ggml (v0.1.0-xxx-gxxxx)
  subdir examples/server/frontend (v1.0.0-xxx-gxxxx)
  subdir thirdparty/libwebp (v1.3.2-xxx-gxxxx)
  subdir thirdparty/libwebm (v1.0.0.28-xxx-gxxxx)
```

**注意**: Submodule 需要单独 push 到各自的上游仓库，不推到 haydigital。

---

### Task 8: 推送到 GitHub

**目标**: 将本地仓库推送到远程

**操作**:
```bash
cd D:/workspace/stable-diffusion.cpp

# 推送 main 分支到远程
git push -u origin main

# 如果需要推送 submodule 的引用（可选）
# git push origin main --recurse-submodules=on-demand
```

**预期输出**:
```
Enumerating objects: 245, done.
Counting objects: 100% (245/245), done.
Delta compression using up to 16 threads
Compressing objects: 100% (198/198), done.
Writing objects: 100% (245/245), 45.67 MiB | 12.34 MiB/s, done.
Total 245 (delta 45), reused 200 (delta 30), pack-reused 0
remote: Resolving deltas: 100% (45/45), done.
To https://github.com/mozhijian925/haydigital.git
 * [new branch]      main → main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

---

### Task 9: 验证远程仓库

**目标**: 确认代码成功推送到 GitHub

**操作**:
```bash
# 查看远程状态
git remote -v
git status

# 查看远程分支
git branch -r

# 查看提交历史
git log --oneline --decorate --graph --all
```

**预期输出**:
```
origin  https://github.com/mozhijian925/haydigital.git (fetch)
origin  https://github.com/mozhijian925/haydigital.git (push)

  main
* master

abc1234 (HEAD → main, origin/main) feat: initial commit...
```

**手动验证**:
打开浏览器访问 `https://github.com/mozhijian925/haydigital` 确认文件已显示。

---

### Task 10: 创建功能分支（可选）

**目标**: 为 InfiniTalk 开发创建独立分支

**操作**:
```bash
cd D:/workspace/stable-diffusion.cpp

# 从 main 创建功能分支
git checkout -b feature/infini-talk

# 验证分支创建
git branch
git branch -v

# 后续开发在此分支进行
# 完成后合并回 main:
# git checkout main
# git merge feature/infini-talk
```

**预期输出**:
```
  main
* feature/infini-talk
  main abc1234 feat: initial commit...
* feature/infini-talk abc1234 feat: initial commit...
```

---

## Tests / Validation

### 本地验证

```bash
# 1. 验证仓库状态
cd D:/workspace/stable-diffusion.cpp
git status
# 预期：On branch main, nothing to commit

# 2. 验证提交历史
git log --oneline
# 预期：显示初始提交

# 3. 验证文件列表
git ls-files | wc -l
# 预期：~200 files

# 4. 验证大文件排除
git ls-files | grep -E "\.gguf$|\.exe$|\.bin$"
# 预期：无输出（应被 .gitignore 排除）
```

### 远程验证

```bash
# 1. 拉取测试（验证远程可访问）
git fetch origin
git log origin/main --oneline
# 预期：显示远程提交

# 2. 检查文件大小
git verify-pack -v .git/objects/pack/*.idx | sort -k 3 -n | tail -5
# 预期：显示最大的 5 个对象（应 < 10MB 每个）
```

### GitHub Web 验证

访问 `https://github.com/mozhijian925/haydigital`:
- [ ] 看到 README.md 和项目文件
- [ ] 文件数量正确（~200 files）
- [ ] 无大文件（.gguf, .exe 等）
- [ ] 提交历史显示正确信息

---

## Risks, Tradeoffs, and Open Questions

### 风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 推送失败（网络/认证） | 中 | 高 | 检查 gh auth status，配置 credential helper |
| 大文件意外提交 | 中 | 中 | 严格使用 .gitignore，推送前检查 |
| Submodule 引用问题 | 低 | 低 | 文档化 submodule 更新流程 |
| 仓库大小超限 | 低 | 高 | GitHub 免费版 1GB 限制，当前 < 50MB 安全 |

### 权衡

1. **单仓库 vs Fork 模式**:
   - 单仓库：简单，适合个人项目
   - Fork：适合开源贡献，需维护两个仓库
   
   **建议**: 使用单仓库（用户是所有者）

2. **Submodule vs  vendor**:
   - Submodule：保持上游同步，但复杂
   - Vendored：简单，但需手动更新
   
   **建议**: 保持现有 submodule 配置（项目标准做法）

3. **Include 测试文件**:
   - 包含：方便开发者本地测试
   - 排除：保持仓库整洁
   
   **建议**: 排除 test/ 目录（已在 .gitignore 中）

### 待解决问题

1. **GitHub Token 权限**: 是否需要创建 Personal Access Token？
2. **CI/CD 集成**: 是否需要在 GitHub 启用 Actions？
3. **Issue 追踪**: 是否需要启用 GitHub Issues？
4. **Wiki**: 是否需要创建 Wiki 文档？

---

## 实施顺序

```
Task 1 (检查环境) 
    ↓
Task 2 (初始化 Git)
    ↓
Task 3 (关联远程)
    ↓
Task 4 (更新 .gitignore)
    ↓
Task 5-6 (提交代码)
    ↓
Task 7 (Submodule)
    ↓
Task 8 (推送)
    ↓
Task 9-10 (验证 + 分支)
```

**预计工作量**: 15-30 分钟（取决于网络速度）

---

## 后续步骤

推送成功后：

1. 在 GitHub 页面设置仓库描述和标签
2. 启用 GitHub Pages（如需文档站）
3. 创建 Issue 模板和 PR 模板
4. 开始 InfiniTalk 功能开发（参考 `2026-09-30_091500-infini-talk-full-support.md`）
5. 定期提交更新并 push

---

**计划文件**: `.hermes/plans/2026-09-30_091500-git-push-haydigital.md`
