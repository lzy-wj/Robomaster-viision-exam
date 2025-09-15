# 第二部分：工作站环境配置与部署

## 📋 项目概述

机器人开发环境的自动化配置与部署，涵盖开发工具安装、版本控制初始化、容器化封装的完整 DevOps 流程。

## 🏗️ 项目结构

```
第二部分/
├── README.md                           # 项目文档
├── test_all.sh                         # 一键测试脚本
├── 2.1_env/                            # 开发环境自动化
│   ├── setup_env.sh                    # 环境配置脚本
│   ├── test.sh                         # 自动化测试
│   └── README.md                       # 模块文档
├── 2.2_repo/                           # 工程版本控制
│   ├── init_repo.sh                    # Git 项目初始化
│   ├── test.sh                         # 自动化测试
│   ├── README.md                       # 模块文档
│   └── sample_repo/                    # 生成的示例项目
└── 2.3_container/                      # 应用容器化
    ├── Dockerfile                      # 多阶段构建配置
    ├── app/main.py                     # 示例应用程序
    ├── requirements.txt                # Python 依赖
    ├── test.sh                         # 自动化测试
    └── README.md                       # 模块文档
```

## 🚀 快速开始

### ⚡ 一键测试（推荐）

```bash
cd 第二部分
bash test_all.sh
```

### 🔧 单模块测试

```bash
# 环境配置测试
bash 2.1_env/test.sh

# 版本控制测试  
bash 2.2_repo/test.sh

# 容器化测试
bash 2.3_container/test.sh
```

## 📦 模块功能

### 🛠️ 2.1 开发环境自动化

- **功能**: 自动安装 git、cmake、docker 等核心工具
- **特性**: 架构检测、权限配置、多发行版支持
- **使用**: `bash setup_env.sh --check|--install`

### 🎯 2.2 工程版本控制

- **功能**: 创建标准 Git 项目结构（Hello, RoboMaster!）
- **特性**: 自动 .gitignore 配置、幂等性操作
- **使用**: `bash init_repo.sh [目录名]`

### 🐳 2.3 应用容器化

- **功能**: 多阶段 Docker 构建与轻量化部署
- **特性**: 虚拟环境隔离、最小化镜像
- **使用**: `docker build -t rm-hello . && docker run --rm rm-hello`

## 📋 系统要求

- **操作系统**: Linux (推荐 Ubuntu/Debian)
- **必需工具**: bash, git (2.1 自动安装)
- **可选工具**: docker (容器化测试需要)
