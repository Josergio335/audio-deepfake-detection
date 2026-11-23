#!/bin/bash
# R6: Exp-N con RASA y lambda alto (variante C: orth 1.0, reco 1.0, serm 0.05), 3 semillas, y evaluacion en todos los escenarios.
# Sirve para comparar con la variante A (0.01/0.01/0.05) y con la ablacion sin RASA.
# Uso (en tmux):  bash run_lambda1.sh [device=cuda:1] [exp=4] [semillas="1 2 3"]
DEVICE=${1:-cuda:1}
EXP=${2:-4}
SEMILLAS=${3:-"1 2 3"}
cd "$(dirname "$0")"
ESCENARIOS="indistribucion fueradominio compresion_mp3_32kbps zeroshot_exp4 zs_limpio zs_mp3_32kbps zs_plr10"
for s in $SEMILLAS; do
  echo "===== Exp-$EXP RASA lambda 1/1/0.05, semilla $s ====="
  python train.py --exp "$EXP" --lambda-orth 1.0 --lambda-reco 1.0 --lambda-serm 0.05 --seed "$s" --device "$DEVICE" 2>&1 | tee -a ~/entrenamientos/log_lambda1_exp${EXP}_seed$s.txt
  RUN=$(ls -td ~/entrenamientos/exp${EXP}_* | head -n 1)
  echo "$RUN" >> ~/entrenamientos/lista_lambda1_exp${EXP}.txt
  for esc in $ESCENARIOS; do
    python evaluate.py --checkpoint "$RUN/checkpoint_mejor.pt" --escenario "$esc" --device "$DEVICE" 2>&1 | grep -E "Checkpoint|Escenario|Accuracy|AUROC|EER"
  done
done
echo "===== Terminado. Corridas: ====="; cat ~/entrenamientos/lista_lambda1_exp${EXP}.txt
