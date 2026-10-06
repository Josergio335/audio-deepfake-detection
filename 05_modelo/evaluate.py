"""
evaluate.py
Evaluacion del modelo entrenado sobre el split de test.

Calcula las metricas principales del E2:
  - EER  (Equal Error Rate):        metrica primaria del paper y del IOV R7
  - t-DCF (tandem Detection Cost):   metrica secundaria, estandar ASVspoof
  - AUROC:                           area bajo la curva ROC

Uso:
    python evaluate.py --checkpoint D:\Entrenamientos\exp3_...\checkpoint_mejor.pt --exp 3
    python evaluate.py --checkpoint checkpoint_mejor.pt --exp 1 --split val
    python evaluate.py --checkpoint checkpoint_mejor.pt --exp 3 --guardar-scores

Salida:
    metricas_test.json   <- EER, t-DCF, AUROC y desglose por umbral
    scores_test.csv      <- score de cada muestra (si --guardar-scores)
    roc_curve.csv        <- curva ROC completa (para graficar en la tesis)

R7 IOV: "1 reporte documentando EER y t-DCF en escenarios de compresion
         (MP3/AAC) en multiples idiomas" — este script cubre la parte base;
         los escenarios de compresion se agregan en evaluate_robustez.py.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importar modulos del modelo
sys.path.insert(0, str(Path(__file__).parent))
from models.rasa   import RASA
from models.serm   import SERM
from models.gating import GatingNetwork
from train import ModeloDeepfake, EmbeddingDataset, PARTICIONES, EMBEDDINGS_DIR

# ---------------------------------------------------------------------------
# Calculo de EER
# ---------------------------------------------------------------------------

def calcular_eer(scores, etiquetas):
    """
    Calcula el Equal Error Rate (EER).

    El EER es el punto donde FAR (False Acceptance Rate) = FRR (False Rejection Rate).
    En deteccion de deepfake:
      - bonafide = clase negativa (etiqueta 0)
      - spoof    = clase positiva (etiqueta 1)
      - FAR: bonafides clasificados como spoof
      - FRR: spoofs clasificados como bonafide

    Args:
        scores:    array de scores (logits o probabilidades), mayor = mas spoof
        etiquetas: array de etiquetas binarias {0=bonafide, 1=spoof}
    Returns:
        eer:       float en [0, 1]
        umbral:    umbral optimo donde FAR ≈ FRR
    """
    from sklearn.metrics import roc_curve

    # roc_curve devuelve FPR, TPR, thresholds
    # FPR = FAR, FNR = 1 - TPR = FRR
    fpr, tpr, umbrales = roc_curve(etiquetas, scores, pos_label=1)
    fnr = 1.0 - tpr

    # EER: punto donde |FAR - FRR| es minimo
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    umbral_eer = umbrales[idx]

    return float(eer), float(umbral_eer)


# ---------------------------------------------------------------------------
# Calculo de t-DCF (formula estandar ASVspoof)
# ---------------------------------------------------------------------------

def calcular_tdcf(scores, etiquetas,
                  p_spoof=0.05,
                  c_miss=1.0,
                  c_fa=10.0):
    """
    Calcula el normalized tandem Detection Cost Function (t-DCF).

    Formula segun Kinnunen et al. (2020) "t-DCF: a detection cost function
    for the tandem assessment of spoofing countermeasures and ASV":

        t-DCF = C_miss * P_miss * P_spoof + C_fa * P_fa * (1 - P_spoof)

    normalizado por el minimo alcanzable de un sistema perfecto.

    Args:
        scores:   array de scores del CM (countermeasure), mayor = mas spoof
        etiquetas: {0=bonafide, 1=spoof}
        p_spoof:  prior de ataque (0.05 por defecto, estandar ASVspoof)
        c_miss:   costo de no detectar un spoof (1.0)
        c_fa:     costo de rechazar un bonafide (10.0)
    Returns:
        min_tdcf: float — minimo t-DCF sobre todos los umbrales
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(etiquetas, scores, pos_label=1)
    fnr = 1.0 - tpr  # P_miss

    # t-DCF para cada umbral
    tdcf = (c_miss * fnr * p_spoof
            + c_fa  * fpr * (1.0 - p_spoof))

    # Normalizar: dividir por el costo de un clasificador trivial
    tdcf_norm = tdcf / min(c_miss * p_spoof, c_fa * (1.0 - p_spoof))

    return float(np.min(tdcf_norm))


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

@torch.no_grad()
def inferir(modelo, loader, device):
    """
    Corre el modelo sobre el loader y retorna scores y etiquetas reales.

    Returns:
        scores:    np.array (N,) — logits sin sigmoid
        etiquetas: np.array (N,) — {0, 1}
    """
    modelo.eval()
    todos_scores    = []
    todas_etiquetas = []

    for emb_w2v, emb_wavlm, emb_whisper, etiquetas in tqdm(loader, desc="inferencia"):
        emb_w2v     = emb_w2v.to(device)
        emb_wavlm   = emb_wavlm.to(device)
        emb_whisper = emb_whisper.to(device)

        score, _ = modelo(emb_w2v, emb_wavlm, emb_whisper)

        todos_scores.append(score.cpu().numpy())
        todas_etiquetas.append(etiquetas.numpy())

    scores    = np.concatenate(todos_scores)
    etiquetas = np.concatenate(todas_etiquetas)
    return scores, etiquetas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evalua el modelo entrenado y calcula EER, t-DCF y AUROC"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Ruta al archivo .pt del checkpoint (checkpoint_mejor.pt)")
    parser.add_argument("--exp", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Split a evaluar (test por defecto)")
    parser.add_argument("--bs", type=int, default=512, dest="batch_size")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--guardar-scores", action="store_true",
                        help="Guarda el score de cada muestra en scores_test.csv")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Directorio de salida: mismo directorio que el checkpoint
    ck_path = Path(args.checkpoint)
    out_dir = ck_path.parent
    logging.info(f"Checkpoint : {ck_path}")
    logging.info(f"Experimento: Exp-{args.exp} | Split: {args.split}")
    logging.info(f"Device     : {args.device}")

    # Dataset y DataLoader
    particion_dir = PARTICIONES[args.exp]
    ds = EmbeddingDataset(particion_dir, args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers,
                    pin_memory=(args.device == "cuda"))

    logging.info(f"Muestras en {args.split}: {len(ds):,}")

    # Cargar modelo
    modelo = ModeloDeepfake().to(args.device)
    ck = torch.load(ck_path, map_location=args.device, weights_only=False)
    modelo.load_state_dict(ck["model_state"])
    logging.info(f"Modelo cargado (epoch {ck['epoch']})")

    # Inferencia
    scores, etiquetas = inferir(modelo, dl, args.device)

    # Probabilidades (para AUROC)
    probs = 1.0 / (1.0 + np.exp(-scores))  # sigmoid manual

    # Metricas
    from sklearn.metrics import roc_auc_score, roc_curve

    eer, umbral_eer = calcular_eer(scores, etiquetas)
    min_tdcf        = calcular_tdcf(scores, etiquetas)
    auroc           = float(roc_auc_score(etiquetas, probs))

    # Accuracy en el umbral del EER
    predicciones = (scores >= umbral_eer).astype(float)
    accuracy     = float((predicciones == etiquetas).mean())

    # Mostrar resultados
    logging.info("=" * 50)
    logging.info(f"  EER      : {eer*100:.2f}%")
    logging.info(f"  min t-DCF: {min_tdcf:.4f}")
    logging.info(f"  AUROC    : {auroc:.4f}")
    logging.info(f"  Accuracy : {accuracy*100:.2f}%  (umbral={umbral_eer:.4f})")
    logging.info("=" * 50)

    # Guardar metricas en JSON
    resultado = {
        "experimento": args.exp,
        "split":       args.split,
        "checkpoint":  str(ck_path),
        "epoch":       ck["epoch"],
        "n_muestras":  len(ds),
        "eer_pct":     round(eer * 100, 4),
        "min_tdcf":    round(min_tdcf, 6),
        "auroc":       round(auroc, 6),
        "accuracy_pct": round(accuracy * 100, 4),
        "umbral_eer":  round(umbral_eer, 6),
    }

    metricas_path = out_dir / f"metricas_{args.split}.json"
    with open(metricas_path, "w") as f:
        json.dump(resultado, f, indent=2)
    logging.info(f"Metricas guardadas en: {metricas_path}")

    # Guardar curva ROC (para graficar en la tesis)
    fpr, tpr, umbrales_roc = roc_curve(etiquetas, scores, pos_label=1)
    roc_path = out_dir / f"roc_{args.split}.csv"
    with open(roc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "umbral"])
        for fp, tp, u in zip(fpr, tpr, umbrales_roc):
            writer.writerow([f"{fp:.6f}", f"{tp:.6f}", f"{u:.6f}"])
    logging.info(f"Curva ROC guardada en: {roc_path}")

    # Guardar scores individuales (opcional, para analisis por idioma/dataset)
    if args.guardar_scores:
        scores_path = out_dir / f"scores_{args.split}.csv"
        with open(scores_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "score", "prob", "etiqueta", "prediccion"])
            for i, (s, p, e) in enumerate(zip(scores, probs, etiquetas)):
                pred = int(s >= umbral_eer)
                writer.writerow([i, f"{s:.6f}", f"{p:.6f}", int(e), pred])
        logging.info(f"Scores individuales en: {scores_path}")


if __name__ == "__main__":
    main()
