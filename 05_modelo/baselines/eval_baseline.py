# -*- coding: utf-8 -*-
"""
eval_baseline.py
Evaluacion de los baselines entrenados con train_baseline.py sobre los escenarios de test.

Lee los mismos indices de test que evaluate.py (~/embeddings_test/indice_test_{escenario}.csv,
con las etiquetas ya corregidas) y usa solo los archivos cuyo audio existe en disco,
por lo que el conjunto evaluado coincide con el de los modelos principales.

Metricas (calculadas igual que en desglose_subconjuntos.py): EER, AUROC y, con umbral 0.5,
accuracy, deteccion de spoof y falsa alarma sobre bonafide. En indistribucion se
anade ademas el EER solo sobre ASVspoof 2019 LA `eval`, comparable con la literatura.

Uso:
    python eval_baseline.py --checkpoint $(cat ~/entrenamientos_baseline/ultimo_rawnet2.txt)/checkpoint_mejor.pt
    python eval_baseline.py --checkpoint ... --escenario indistribucion
    python eval_baseline.py --checkpoint ... --limite 64      # prueba rapida

Salida (junto al checkpoint):
    eval_{escenario}.json            <- metricas
    eval_{escenario}_scores.csv      <- score de spoof por archivo
    eval_resumen.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train_baseline import MODELOS, resolver_ruta, cargar_audio, MAX_SEGUNDOS, SR_TARGET

EMBEDDINGS_TEST_DIR = Path.home() / "embeddings_test"

ESCENARIOS = ["indistribucion", "fueradominio", "compresion_mp3_32kbps",
              "zeroshot_exp4", "zeroshot_v2"]
# zeroshot_v2 incluye los 9 idiomas de Common Voice que el Exp-4 usa en entrenamiento:
# no se evalua por defecto para no mezclar idiomas vistos y no vistos.
POR_DEFECTO = ESCENARIOS[:4]

_trapz = getattr(np, "trapezoid", None) or np.trapz


class TestDataset(Dataset):
    def __init__(self, escenario: str, limite: int = 0, max_segundos: float = MAX_SEGUNDOS):
        self.max_muestras = int(max_segundos * SR_TARGET)
        self.muestras = []  # (ruta_audio, etiqueta, ruta_original)
        total = faltantes = 0

        indice = EMBEDDINGS_TEST_DIR / f"indice_test_{escenario}.csv"
        with open(indice, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                total += 1
                ruta = resolver_ruta(row["ruta_original"])
                if not ruta.exists():
                    faltantes += 1
                    continue
                self.muestras.append((ruta, 1 if row["label"] == "spoof" else 0,
                                      row["ruta_original"]))
                if limite and len(self.muestras) >= limite:
                    break
        print(f"  {escenario}: {len(self.muestras):,} muestras con audio "
              f"({faltantes:,} omitidas de {total:,})")
        if not self.muestras:
            raise RuntimeError(f"Sin audio disponible para {escenario}")

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        ruta, etiqueta, ruta_orig = self.muestras[i]
        return cargar_audio(ruta, self.max_muestras, estricto=True), etiqueta, ruta_orig


def roc(pos: np.ndarray, neg: np.ndarray):
    """AUROC y EER con pos=spoof, neg=bonafide."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(-s, kind="stable")
    s, y = s[o], y[o]
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    ult = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tpr = np.r_[0.0, tps[ult] / len(pos)]
    fpr = np.r_[0.0, fps[ult] / len(neg)]
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float(_trapz(tpr, fpr)), float((fpr[i] + fnr[i]) / 2)


def metricas(labels: np.ndarray, scores: np.ndarray) -> dict:
    auroc, eer = roc(scores[labels == 1], scores[labels == 0])
    pred = scores >= 0.5
    return {
        "n": int(len(labels)),
        "accuracy": float(np.mean(pred == (labels == 1))),
        "det_spoof": float(np.mean(pred[labels == 1])) if (labels == 1).any() else float("nan"),
        "falsa_alarma": float(np.mean(pred[labels == 0])) if (labels == 0).any() else float("nan"),
        "auroc": auroc,
        "eer": eer,
    }


@torch.no_grad()
def inferir(modelo, ds, device, batch_size, workers):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True)
    modelo.eval()
    scores, labels, rutas = [], [], []
    for wav, etiquetas, ruta_orig in tqdm(dl, desc="inferencia", leave=False):
        scores.append(torch.sigmoid(modelo(wav.to(device))).cpu().numpy())
        labels.append(etiquetas.numpy())
        rutas.extend(ruta_orig)
    return np.concatenate(labels), np.concatenate(scores), rutas


def evaluar_escenario(modelo, escenario, device, out_dir, batch_size, workers, limite):
    print(f"\n  Escenario: {escenario}")
    ds = TestDataset(escenario, limite)
    labels, scores, rutas = inferir(modelo, ds, device, batch_size, workers)

    res = {"escenario": escenario, **metricas(labels, scores)}
    if escenario == "indistribucion":
        es_eval = np.array(["/LA/eval/" in r.replace("\\", "/") for r in rutas])
        if es_eval.any():
            res["eer_solo_eval"] = metricas(labels[es_eval], scores[es_eval])["eer"]

    print(f"  EER={res['eer'] * 100:.2f}%  AUROC={res['auroc']:.4f}  "
          f"acc={res['accuracy'] * 100:.2f}%  n={res['n']:,}"
          + (f"  EER(eval)={res['eer_solo_eval'] * 100:.2f}%" if "eer_solo_eval" in res else ""))

    (out_dir / f"eval_{escenario}.json").write_text(json.dumps(res, indent=2))
    with open(out_dir / f"eval_{escenario}_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ruta", "label", "score_spoof"])
        for r, l, s in zip(rutas, labels.tolist(), scores.tolist()):
            w.writerow([r, int(l), f"{s:.6f}"])
    return res


def main():
    ap = argparse.ArgumentParser(description="Evaluacion de baseline (R8)")
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="checkpoint_mejor.pt generado por train_baseline.py")
    ap.add_argument("--escenario", default="todos",
                    help="'todos' (los 4 principales), o uno o varios escenarios separados por coma "
                         "(cualquiera con indice en ~/embeddings_test/, p. ej. zs_limpio,zs_mp3_32kbps)")
    ap.add_argument("--bs", type=int, default=64, dest="batch_size")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limite", type=int, default=0, help="Solo para pruebas: N muestras por escenario")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"ERROR: checkpoint no encontrado: {args.checkpoint}")
    out_dir = args.checkpoint.parent
    config_path = out_dir / "config.json"
    if not config_path.exists():
        sys.exit("ERROR: config.json no encontrado junto al checkpoint")
    nombre_modelo = json.loads(config_path.read_text())["modelo"]

    print("=" * 60)
    print(f"  EVALUACION BASELINE: {nombre_modelo.upper()}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)

    device = torch.device(args.device)
    modelo = MODELOS[nombre_modelo]().to(device)
    modelo.load_state_dict(torch.load(args.checkpoint, map_location=device)["modelo"])

    escenarios = POR_DEFECTO if args.escenario == "todos" else args.escenario.split(",")
    resultados = []
    for esc in escenarios:
        try:
            resultados.append(evaluar_escenario(modelo, esc, device, out_dir,
                                                args.batch_size, args.workers, args.limite))
        except Exception as e:
            print(f"  [!] {esc}: {e}")

    print("\n" + "=" * 60)
    print(f"  {'Escenario':<26}{'EER%':>8}{'AUROC':>8}{'Acc%':>8}{'EER eval%':>11}")
    for r in resultados:
        eval_eer = f"{r['eer_solo_eval'] * 100:>11.2f}" if "eer_solo_eval" in r else f"{'-':>11}"
        print(f"  {r['escenario']:<26}{r['eer'] * 100:>8.2f}{r['auroc']:>8.4f}"
              f"{r['accuracy'] * 100:>8.2f}{eval_eer}")
    (out_dir / "eval_resumen.json").write_text(json.dumps(resultados, indent=2))
    print(f"\n  Resumen guardado en: {out_dir / 'eval_resumen.json'}")


if __name__ == "__main__":
    main()
