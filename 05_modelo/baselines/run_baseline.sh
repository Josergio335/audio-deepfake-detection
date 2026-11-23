#!/usr/bin/env bash
# Entrena un baseline con el corpus del Exp-4 y lo evalua en los escenarios de test.
# Uso:  ./run_baseline.sh <lfcc_lcnn|rawnet2|aasist> [device=cuda:1] [horas=0] [epocas=50] [batch=32]
#   horas=0 (por defecto): sin limite de tiempo, se completan todas las epocas.
set -euo pipefail

MODELO=${1:?uso: ./run_baseline.sh <lfcc_lcnn|rawnet2|aasist> [device] [horas] [epocas] [batch]}
DEVICE=${2:-cuda:1}
HORAS=${3:-0}
EPOCAS=${4:-50}
BATCH=${5:-32}

LIMITE=()
if [ "$HORAS" != "0" ]; then
    LIMITE=(--max-horas "$HORAS")
fi

cd "$(dirname "$0")"
python train_baseline.py --modelo "$MODELO" --exp 4 --device "$DEVICE" \
    --epochs "$EPOCAS" --bs "$BATCH" ${LIMITE[@]+"${LIMITE[@]}"}
CK="$(cat "$HOME/entrenamientos_baseline/ultimo_${MODELO}.txt")/checkpoint_mejor.pt"
python eval_baseline.py --checkpoint "$CK" --device "$DEVICE"
