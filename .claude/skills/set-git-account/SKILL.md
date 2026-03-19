# 设置 Git 帐号

## Overview

将 Git 全局配置设置为用户名 `<USER>`，邮箱 `<EMAIL>`。适用于配置本机 Git 提交身份。

## Steps

1. **设置 Git 用户名**
   - 执行：`git config --global user.name "<USER>"`

2. **设置 Git 邮箱**
   - 执行：`git config --global user.email "<EMAIL>"`

3. **验证配置**
   - 执行：`git config --global --list | grep user` 确认配置已生效

## Rules

- **全局配置**：使用 `--global`，作用于当前用户所有仓库
- **若需仅对当前仓库生效**：去掉 `--global`，在目标仓库目录下执行