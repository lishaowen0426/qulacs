#!/usr/bin/env bash
set -euo pipefail
set +x

num_of_rank=4
qubits=(12)

./script/build_mpicc.sh >/dev/null 2>&1

for q in "${qubits[@]}"; do
   srun -n "${num_of_rank}" ./bin/insitu_benchmark "${q}"
done
