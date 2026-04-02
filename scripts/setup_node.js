#!/usr/bin/env bash

set -euo pipefail

echo "Installing Node.js 20..."

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get remove -y nodejs npm || true
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
else
  echo "apt-get not found. Install Node.js 20+ manually for your platform."
  exit 1
fi

echo "Node version:"
node -v

echo "npm version:"
npm -v
