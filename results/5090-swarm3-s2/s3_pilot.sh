#!/bin/bash
# SWARM-3 S2: memory and behaviour pilot on the 8 reserved functions, Qwen3-4B bf16, batch 16, no prompt search.
cd ~/ax/swarm-collusion-probe/runs/swarm3-s2
sha256sum collusion_probe.py swarm3.py > swarm3-s2.log
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ../../waitgpu.sh 13000 \
  ~/quant-omp-foba/.venv/bin/python -u -c "import swarm3; swarm3.main(out='results', pilot=True, batch=16)" >> swarm3-s2.log 2>&1
