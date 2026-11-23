#!/usr/bin/env bash
# Prueba de humo (~minutos): entrena 1 epoca con 128 muestras y evalua 64 por escenario,
# para cada baseline. Si algo falla, falla aqui y no a mitad de la noche.
# Uso:  ./prueba_rapida.sh [device=cuda:1]
set -euo pipefail

DEVICE=${1:-cuda:1}
cd "$(dirname "$0")"

for MODELO in lfcc_lcnn rawnet2 aasist; do
    echo "=============== $MODELO ==============="
    python train_baseline.py --modelo "$MODELO" --exp 4 --epochs 1 --limite 128 \
        --bs 16 --workers 2 --device "$DEVICE" 2>&1 | tail -n 4
    CK="$(cat "$HOME/entrenamientos_baseline/ultimo_${MODELO}.txt")/checkpoint_mejor.pt"
    python eval_baseline.py --checkpoint "$CK" --limite 64 --workers 2 --device "$DEVICE" 2>&1 | tail -n 9
done
echo "PRUEBA OK"
