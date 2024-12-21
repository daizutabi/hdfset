FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-24.04

RUN apt-get update && apt-get install -y --no-install-recommends python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
