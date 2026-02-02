#!/usr/bin/env bash
set -euo pipefail
set +x

cd "${HOME}/qulacs"

num_of_rank=4
qubits=(12 16 20 24 28 32)

./script/build_mpicc.sh >/dev/null 2>&1

for q in "${qubits[@]}"; do
   srun --mpi=pmix -n "${num_of_rank}" ./bin/insitu_benchmark "${q}"
done
