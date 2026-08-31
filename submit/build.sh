#!/bin/bash
set -e
cd "$(dirname "$0")"
docker build -t mbt2026_v002 .
