# Commit and Push Staged Changes

## Overview

为已 `git add` 暂存的文件生成 commit，并 push 到当前分支。假设变更已全部 staged，不再执行 `git add`。

## Steps

1. **检查暂存状态**
   - 运行 `git status` 确认有已暂存文件
   - 若无 staged 文件，提示用户先执行 `git add` 再运行此命令

2. **查看变更内容**
   - 运行 `git diff --cached` 查看 staged 的 diff
   - 根据实际变更生成简洁、准确的 commit message

3. **生成并执行 commit**
   - 基于 diff 内容生成 commit message
   - 格式：`git commit -s -m "type(scope): description"`（`-s` 添加 Signed-off-by）
   - 示例：`git commit -s -m "fix(pa): correct sliding window mask in decode kernel"`
   - 规则：≤72 字符、祈使语气（fix/add/update）、首字母大写、句末无句号

4. **Push 到当前分支**
   - 运行 `git push -u origin HEAD` 或 `git push -u origin $(git branch --show-current)`
   - 若 push 被拒绝（远程有新提交），执行 `git pull --rebase && git push`

## Rules

- **禁止执行 `git add`**：只提交用户已经 `git add` 暂存的文件。绝对不要自行执行 `git add` 添加任何文件，即使发现有 untracked 或 modified 但未 staged 的文件，也不得擅自 stage。如果没有已 staged 的文件，停止操作并提示用户先手动 `git add`
- **Commit message**：基于实际 diff，描述做了什么以及原因
- **Push**：推送到当前分支对应的远程分支
