#!/usr/bin/env bash
# Uso: ./lancia_parallelo.sh M K
# Avvia M istanze di SCRIPT con progressivi K, K+1, ..., K+M-1
SCRIPT="long_sim_batch.py"

M=$1
K=$2

for ((i = K; i < K + M; i++)); do
    python3 "$SCRIPT" "$i" &
done
wait