#!/bin/bash

echo 'eval "$(starship init bash)"' >> ~/.bashrc
mkdir -p ~/.config
cp .devcontainer/starship.toml ~/.config

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv tool install ruff@latest
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc