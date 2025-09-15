### 2.3 应用容器化（多阶段 Docker 构建）

包含内容：
- `Dockerfile`：builder 阶段安装依赖，runtime 阶段仅携带虚拟环境与应用代码。
- `app/main.py`：最小可运行应用。
- `requirements.txt`：依赖清单（示例为空）。

测试：
- bash: `bash test.sh`
  - 构建镜像：`docker build -t rm-hello:latest .`
  - 运行容器：`docker run --rm rm-hello:latest`


