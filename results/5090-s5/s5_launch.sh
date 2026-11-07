#!/bin/bash
# SWARM-1 S5: seeds 1 to 7 in sequence; each seed waits until the GPU has 10500 MiB free (waitgpu.sh), as S4 did.
cd ~/ax/swarm-collusion-probe/runs/s5 || exit 1
mkdir -p results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for s in 1 2 3 4 5 6 7; do
  SEED=$s ~/ax/swarm-collusion-probe/waitgpu.sh 10500 ~/quant-omp-foba/.venv/bin/python -u run.py > results/swarm1-s5-seed$s.log 2>&1
done
echo "S5 loop done $(date -u +%FT%TZ)"
