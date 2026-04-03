#!/usr/bin/env bash
# Run from repo root: sample images with the generator (see inference_generator.py for flags).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m accelerate.commands.launch \
  --main_process_port 29503 \
  inference_generator.py \
  --config configs/gpstoken_generator.yaml \
  --model_path weights/generator.safetensors \
  --gpstoken_path weights/gpstoken_m128.safetensors \
  --initg_path weights/initg.pickle \
  --data_count 50 \
  --class_count 1000 \
  --cfg_scale 1.50 \
  --output results/sde_cfg1.50_gl200_gh1000 \
  --max_count 50000 \
  --guidance_low 200
