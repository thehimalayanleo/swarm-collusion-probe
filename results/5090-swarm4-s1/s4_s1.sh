#!/bin/bash
# SWARM-4 S1: offline smoke on the 5090 CPU. test_swarm3.py (unchanged code still passes) and test_swarm4.py in parallel.
cd ~/ax/swarm-collusion-probe/runs/swarm4-s1
sha256sum collusion_probe.py swarm3.py swarm4.py test_swarm3.py test_swarm4.py > swarm4-s1.log
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
PY=~/quant-omp-foba/.venv/bin/python
(timeout 3600 $PY -u test_swarm3.py results3 > test3.log 2>&1; echo "test3 exit $?" >> swarm4-s1.log) &
(timeout 3600 $PY -u test_swarm4.py results4 > test4.log 2>&1; echo "test4 exit $?" >> swarm4-s1.log) &
wait
echo "done $(date -u +%FT%TZ)" >> swarm4-s1.log
