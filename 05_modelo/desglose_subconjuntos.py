"""
desglose_subconjuntos.py
Desglosa un escenario de test por subconjunto (tomado de la ruta original) para
ver donde falla el modelo. Reutiliza ModeloDeepfake de evaluate.py.

Uso (desde 05_modelo):
    python desglose_subconjuntos.py --escenario fueradominio \
        --checkpoint ~/entrenamientos/A/checkpoint_mejor.pt ~/entrenamientos/B/checkpoint_mejor.pt

Columnas:
    n_bona / n_spoof : muestras de cada clase en el subconjunto
    acc              : accuracy con umbral 0.5
    det_spoof        : % de spoof detectado (umbral 0.5)       -> NaN si no hay spoof
    falsa_alarma     : % de bonafide marcado como spoof (0.5)  -> NaN si no hay bonafide
    auroc_pool/eer_pool : si el grupo tiene spoof: sus spoof vs TODO el bonafide del
                       escenario; si solo tiene bonafide: TODO el spoof vs su bonafide.
                       Sirve aunque el grupo tenga una sola clase.
    auroc_int/eer_int: AUROC y EER usando solo las muestras del propio grupo
                       (NaN si el grupo tiene una sola clase). Para ASVspoof,
                       --nivel 2 da el EER solo de `eval`, comparable con la literatura.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import ModeloDeepfake, EMBEDDINGS_TEST_DIR, EVALUACIONES_DIR

_trapz = getattr(np, "trapezoid", None) or np.trapz


def subgrupo(row, nivel=1):
    """Subconjunto = `nivel` componentes de la ruta justo despues de la carpeta de la DB."""
    partes = [p for p in row["ruta_original"].replace("\\", "/").split("/") if p]
    db = row["db"].lower()
    for i, p in enumerate(partes[:-1]):
        if p.lower() == db and i + nivel < len(partes) - 1:
            return f'{row["db"]}/' + "/".join(partes[i + 1:i + 1 + nivel])
    return row["db"]


class DS(Dataset):
    def __init__(self, escenario, nivel=1):
        self.items = []
        ruta = EMBEDDINGS_TEST_DIR / f"indice_test_{escenario}.csv"
        with open(ruta, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pts = [Path(r[k]) for k in ("wav2vec2bert_pt", "wavlm_pt", "whisper_pt")]
                if all(p.exists() for p in pts):
                    self.items.append((pts, 1 if r["label"] == "spoof" else 0,
                                       subgrupo(r, nivel), r["ruta_original"]))
        print(f"  {len(self.items):,} muestras con embeddings")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pts, label, _, _ = self.items[i]
        e = [torch.load(p, map_location="cpu", weights_only=True).float() for p in pts]
        return e[0], e[1], e[2], label


@torch.no_grad()
def puntuar(modelo, ds, device):
    dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4)
    out = []
    for a, b, c, _ in tqdm(dl, desc="eval"):
        out.append(torch.sigmoid(modelo(a.to(device), b.to(device), c.to(device))).cpu().numpy())
    return np.concatenate(out)


def roc(pos, neg):
    """AUROC y EER con pos=spoof, neg=bonafide (rapido, por ordenamiento)."""
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
    auroc = float(_trapz(tpr, fpr))
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return auroc, float((fpr[i] + fnr[i]) / 2)


def pct(x):
    return "   -  " if np.isnan(x) else f"{x * 100:5.1f}%"


def evaluar(ckpt, escenario, device, nivel=1):
    print(f"\n{'=' * 78}\nCheckpoint: {ckpt.parent.name} | Escenario: {escenario}\n{'=' * 78}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    modelo = ModeloDeepfake(M=ck.get("config", {}).get("M", 160)).to(device)
    modelo.load_state_dict(ck["model_state"])
    modelo.eval()

    ds = DS(escenario, nivel)
    scores = puntuar(modelo, ds, device)
    labels = np.array([it[1] for it in ds.items])
    grupos = np.array([it[2] for it in ds.items])
    pool_bona, pool_spoof = scores[labels == 0], scores[labels == 1]

    orden = sorted(set(grupos), key=lambda g: -(grupos == g).sum())
    filas = []
    print(f"\n{'subconjunto':<34}{'n_bona':>7}{'n_spoof':>8}{'acc':>8}{'det_spoof':>10}"
          f"{'falsa_al.':>10}{'auroc_pool':>11}{'eer_pool':>9}{'auroc_int':>10}{'eer_int':>8}")
    for g in orden:
        m = grupos == g
        sg, yg = scores[m], labels[m]
        nb, ns = int((yg == 0).sum()), int((yg == 1).sum())
        acc = float(np.mean((sg >= 0.5) == yg))
        det = float(np.mean(sg[yg == 1] >= 0.5)) if ns else float("nan")
        fa = float(np.mean(sg[yg == 0] >= 0.5)) if nb else float("nan")
        if ns:
            au, eer = roc(sg[yg == 1], pool_bona)
        else:
            au, eer = roc(pool_spoof, sg[yg == 0])
        au_i, eer_i = roc(sg[yg == 1], sg[yg == 0])
        print(f"{g:<34}{nb:>7}{ns:>8}{pct(acc):>8}{pct(det):>10}{pct(fa):>10}"
              f"{au:>11.3f}{pct(eer):>9}{au_i:>10.3f}{pct(eer_i):>8}")
        filas.append([g, nb, ns, acc, det, fa, au, eer, au_i, eer_i])

    print("\nEjemplos de ruta por subconjunto (verifica que el agrupamiento tiene sentido):")
    for g in orden:
        ej = next(it[3] for it in ds.items if it[2] == g)
        print(f"  {g}: {ej}")

    EVALUACIONES_DIR.mkdir(parents=True, exist_ok=True)
    salida = EVALUACIONES_DIR / f"desglose_{escenario}_{ckpt.parent.name}.csv"
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subconjunto", "n_bona", "n_spoof", "acc", "det_spoof",
                    "falsa_alarma", "auroc_pool", "eer_pool", "auroc_int", "eer_int"])
        w.writerows(filas)
    print(f"\nGuardado: {salida}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escenario", default="fueradominio")
    ap.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--nivel", type=int, default=1,
                    help="Componentes de ruta tras la DB que definen el subconjunto "
                         "(ASVspoof: 1 = LA, 2 = LA/dev y LA/eval)")
    a = ap.parse_args()
    for ck in a.checkpoint:
        evaluar(ck, a.escenario, a.device, a.nivel)


if __name__ == "__main__":
    main()
