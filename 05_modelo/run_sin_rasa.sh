#!/bin/bash
# Ablacion de RASA (R6): entrena el Exp-N (por defecto 4) SIN RASA (3 corridas) y las evalua en todos los escenarios.
# Misma configuracion que la variante A del Exp-4 (lambda_serm 0.05, 100 epocas, bs 256, lr 1e-3);
# la unica diferencia es --sin-rasa (RASA reemplazado por la identidad).
# Uso (en tmux):  bash run_sin_rasa.sh [device=cuda:1] [exp=4] [semillas="1 2 3"]
DEVICE=${1:-cuda:1}
EXP=${2:-4}
SEMILLAS=${3:-"1 2 3"}
cd "$(dirname "$0")"
ESCENARIOS="indistribucion fueradominio compresion_mp3_32kbps zeroshot_exp4 zs_limpio zs_mp3_32kbps zs_plr10"
for s in $SEMILLAS; do
  echo "===== Exp-$EXP sin RASA, semilla $s ====="
  python train.py --exp "$EXP" --sin-rasa --seed "$s" --device "$DEVICE" 2>&1 | tee -a ~/entrenamientos/log_sinrasa_exp${EXP}_seed$s.txt
  RUN=$(ls -td ~/entrenamientos/exp${EXP}sinrasa_* | head -n 1)
  echo "$RUN" >> ~/entrenamientos/lista_sinrasa_exp${EXP}.txt
  for esc in $ESCENARIOS; do
    python evaluate.py --checkpoint "$RUN/checkpoint_mejor.pt" --escenario "$esc" --device "$DEVICE" 2>&1 | grep -E "Checkpoint|Escenario|Accuracy|AUROC|EER"
  done
done
echo "===== Terminado. Corridas: ====="; cat ~/entrenamientos/lista_sinrasa_exp${EXP}.txt
