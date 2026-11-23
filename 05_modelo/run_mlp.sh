#!/bin/bash
# Linea base MLP sobre embeddings concatenados (sin RASA/Gating/SERM), Exp-4, 3 semillas.
# Uso (en tmux):  bash run_mlp.sh [device=cuda:1] [exp=4] [semillas="1 2 3"] [hidden=256]
DEVICE=${1:-cuda:1}
EXP=${2:-4}
SEMILLAS=${3:-"1 2 3"}
HIDDEN=${4:-256}
cd "$(dirname "$0")"
for s in $SEMILLAS; do
  echo "===== Exp-$EXP MLP hidden=$HIDDEN, semilla $s ====="
  python train_mlp_concat.py --exp "$EXP" --seed "$s" --hidden "$HIDDEN" --device "$DEVICE" 2>&1 | grep -E "RESULTADO|Salida|Epoch (1|50|100)/|Error|Traceback"
done
echo "===== Terminado ====="
ls -td ~/entrenamientos/exp${EXP}mlp_* ~/entrenamientos/exp${EXP}logreg_* 2>/dev/null | head -n 3
