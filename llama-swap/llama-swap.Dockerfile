# llama-swap.Dockerfile

# pinned 2026-06-14; verify on next rebuild
FROM ubuntu:22.04@sha256:4f838adc7181d9039ac795a7d0aba05a9bd9ecd480d294483169c5def983b64d

# Install dependencies AND docker.io so llama-swap can manage containers
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    wget \
    curl \
    docker.io \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Pinned, checksummed llama-swap release (was releases/latest, unverified).
ARG LLAMA_SWAP_VERSION=224
ARG LLAMA_SWAP_SHA256=d62c1d140a6ba3482c50b19f254b085f116a1d3d282a9d0f1ff4113b8a56f4cd
RUN mkdir -p /tmp/llama-swap-build && cd /tmp/llama-swap-build && \
    wget -q "https://github.com/mostlygeek/llama-swap/releases/download/v${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION}_linux_arm64.tar.gz" && \
    echo "${LLAMA_SWAP_SHA256}  llama-swap_${LLAMA_SWAP_VERSION}_linux_arm64.tar.gz" | sha256sum -c - && \
    tar -xzf "llama-swap_${LLAMA_SWAP_VERSION}_linux_arm64.tar.gz" && \
    mv llama-swap /usr/bin/llama-swap && chmod +x /usr/bin/llama-swap && \
    cd / && rm -rf /tmp/llama-swap-build

WORKDIR /app

EXPOSE 8080

ENTRYPOINT ["/usr/bin/llama-swap"]
