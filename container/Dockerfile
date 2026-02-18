FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI for Docker-in-Docker if needed.
RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

# Install Docker Compose V2 plugin (required for --format json flag)
RUN mkdir -p /usr/local/lib/docker/cli-plugins && \
    curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
    -o /usr/local/lib/docker/cli-plugins/docker-compose && \
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# Install Helm (required for RE-Bench K8s sandbox)
RUN curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install uv (fast Python package installer)
RUN pip install --no-cache-dir uv

# Clone and install METR Task Bridge with uv (handles git dependencies)
WORKDIR /app
RUN git clone https://github.com/METR/inspect-metr-task-bridge.git /app/mtb
WORKDIR /app/mtb
RUN uv sync --no-dev

# Install additional LLM SDKs into the venv
RUN uv pip install "inspect-ai>=0.3.170" inspect-k8s-sandbox "openai>=2.8.0" "anthropic>=0.75.0" google-genai gcsfs

# Install metr-task-protected-scoring (git dependency)
RUN uv pip install "metr-task-protected-scoring @ git+https://github.com/METR/task-protected-scoring.git@v0.2.5"

# Clone RE-Bench task source code
RUN git clone https://github.com/METR/inspect-tasks-public.git /app/inspect-tasks-public

# Copy and install anthropic_perf_takehome task
# Clone the Anthropic performance takehome repo
RUN git clone https://github.com/anthropics/original_performance_takehome.git /app/anthropic_repo

# Create the anthropic_perf_takehome Inspect task package
RUN mkdir -p /app/anthropic_perf_takehome/sandbox
COPY . /app/anthropic_perf_takehome/
RUN uv pip install -e /app/anthropic_perf_takehome

RUN uv pip install -e /app/inspect-tasks-public/re_bench/common \
    && uv pip install -e /app/inspect-tasks-public/re_bench/rust_codecontests_inference \
    && uv pip install -e /app/inspect-tasks-public/re_bench/fix_embedding \
    && uv pip install -e /app/inspect-tasks-public/re_bench/triton_cumsum \
    && uv pip install -e /app/inspect-tasks-public/re_bench/nanogpt_chat_rl \
    && uv pip install -e /app/inspect-tasks-public/re_bench/restricted_mlm \
    && uv pip install -e /app/inspect-tasks-public/re_bench/optimize_llm_foundry \
    && uv pip install -e /app/inspect-tasks-public/re_bench/small_scaling_law


# Set working directory
WORKDIR /app

# Create logs directory
RUN mkdir -p /logs

# Set PATH to include uv's venv
ENV PATH="/app/mtb/.venv/bin:$PATH"

# Copy embedded sandbox images (built during Cloud Build)
COPY sandbox_standard.tar sandbox_easy.tar /app/sandbox_images/

# Copy and set up the sandbox image loader script
COPY container/load-sandbox-images.sh /app/load-sandbox-images.sh
RUN chmod +x /app/load-sandbox-images.sh

# Use entrypoint to load sandbox images before running commands
ENTRYPOINT ["/app/load-sandbox-images.sh"]

# Default command
CMD ["bash"]

