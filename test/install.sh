#!/bin/bash
set -euo pipefail

echo "🧹  Removing any legacy containerd / docker packages …"
sudo apt-get remove -y docker docker.io containerd runc || true

echo "🔑  Adding Docker’s official GPG key and repository …"
sudo apt-get update -qq
sudo apt-get install -y ca-certificates curl gnupg lsb-release

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg |
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

release="$(lsb_release -cs)"
arch="$(dpkg --print-architecture)"
echo \
  "deb [arch=${arch} signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu ${release} stable" |
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

echo "📦  Installing Docker Engine 24.x and containerd 1.7+ …"
sudo apt-get update -qq
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

echo "✅  Installed versions:"
docker --version
containerd --version
echo "✨  Done – shim-v3 capable runtime is now available."