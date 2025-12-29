FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-venv \
    ca-certificates curl \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# venv for clean copy into runtime
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt


FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Hugging Face cache
ENV HF_HOME="/runpod-volume/.cache/huggingface/"
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# NLLB variables
ENV NLLB_MODEL_SIZE="600M" \
    NLLB_MODEL_ID="" \
    TP_SIZE="auto" \
    DTYPE="fp16" \
    MAX_BATCH_SIZE="32" \
    MAX_INPUT_LENGTH="10000" \
    PORT="8000" \
    HOST="0.0.0.0"

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    ca-certificates \
    curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /
COPY server.py /server.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

CMD ["/start.sh"]
