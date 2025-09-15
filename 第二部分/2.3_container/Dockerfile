# 多阶段构建：builder 安装依赖，runtime 精简镜像

# builder 阶段
FROM python:3.11-slim AS builder
WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app/ ./app/

# runtime 阶段
FROM python:3.11-slim AS runtime
WORKDIR /app

# 仅复制虚拟环境与应用代码
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/app /app/app

ENV PATH="/opt/venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "app.main"]


