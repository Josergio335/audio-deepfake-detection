"""
exportar_scores.py
Guarda el score (logit de spoof) por archivo de uno o varios checkpoints en varios escenarios,
para analisis posteriores (t-DCF, intervalos de confianza, EER por idioma) con analisis_r8.py.
Carga cada lote de embeddings una sola vez y lo pasa por todos los checkpoints.

Uso (en el servidor):
    python exportar_scores.py --device cuda:1 \
        --escenario indistribucion fueradominio compresion_mp3_32kbps zeroshot_exp4 \
        --checkpoint ~/entrenamientos/exp3_20261119_143040/checkpoint_mejor.pt \
                     ~/entrenamientos/exp4_20261119_125609/checkpoint_mejor.pt

Salida:
    ~/evaluaciones/scores/<escenario>/<nombre_de_la_corrida>.csv
        columnas: ruta_original, label, db, idioma, logit   (mayor logit = mas spoof)
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import ModeloDeepfake, EMBEDDINGS_TEST_DIR, EVALUACIONES_DIR

SALIDA = EVALUACIONES_DIR / "scores"


class DS(Dataset):
    def __init__(self, escenario: str):
        self.items = []
        faltantes = 0
        with open(EMBEDDINGS_TEST_DIR / f"indice_test_{escenario}.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pts = [Path(r[k]) for k in ("wav2vec2bert_pt", "wavlm_pt", "whisper_pt")]
                if all(p.exists() for p in pts):
                    self.items.append((pts, r["ruta_original"], r["label"], r["db"], r["idioma"]))
                else:
                    faltantes += 1
        print(f"  {escenario}: {len(self.items):,} muestras ({faltantes:,} sin embeddings)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e = [torch.load(p, map_location="cpu", weights_only=True).float() for p in self.items[i][0]]
        return e[0], e[1], e[2], i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escenario", nargs="+", required=True)
    ap.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    modelos = []
    for ck in a.checkpoint:
        d = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = d.get("config", {})
        m = ModeloDeepfake(M=cfg.get("M", 160), sin_rasa=cfg.get("sin_rasa", False)).to(a.device)
        m.load_state_dict(d["model_state"])
        m.eval()
        modelos.append((ck.parent.name, m))

    for esc in a.escenario:
        print(f"\nEscenario {esc}")
        ds = DS(esc)
        dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=a.workers)
        logits = {nombre: [] for nombre, _ in modelos}
        with torch.no_grad():
            for e1, e2, e3, _ in tqdm(dl, desc=esc):
                e1, e2, e3 = e1.to(a.device), e2.to(a.device), e3.to(a.device)
                for nombre, m in modelos:
                    logits[nombre].extend(m(e1, e2, e3).cpu().tolist())
        (SALIDA / esc).mkdir(parents=True, exist_ok=True)
        for nombre, valores in logits.items():
            destino = SALIDA / esc / f"{nombre}.csv"
            with open(destino, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ruta_original", "label", "db", "idioma", "logit"])
                for (_, ruta, label, db, idioma), v in zip(ds.items, valores):
                    w.writerow([ruta, label, db, idioma, f"{v:.6f}"])
            print(f"  -> {destino}")


if __name__ == "__main__":
    main()
