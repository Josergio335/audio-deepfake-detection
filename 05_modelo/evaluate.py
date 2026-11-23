"""
evaluate.py
Evaluacion del modelo de deteccion de audio deepfake sobre los conjuntos de test.

Metricas reportadas:
    - Accuracy
    - AUROC (Area bajo la curva ROC)
    - EER   (Equal Error Rate)
    - minDCF (Detection Cost Function minima, parametros ASVspoof LA 2019)

Uso:
    python evaluate.py --checkpoint ~/entrenamientos/ablacion_M160_20261111_101359/checkpoint_mejor.pt \
                       --escenario indistribucion

    python evaluate.py --checkpoint ~/entrenamientos/ablacion_M160_20261111_101359/checkpoint_mejor.pt \
                       --escenario zeroshot_v2

    python evaluate.py --checkpoint ~/entrenamientos/ablacion_M160_20261111_101359/checkpoint_mejor.pt \
                       --escenario fueradominio

    python evaluate.py --checkpoint ~/entrenamientos/ablacion_M160_20261111_101359/checkpoint_mejor.pt \
                       --escenario compresion_mp3_32kbps

    python evaluate.py --checkpoint ~/entrenamientos/exp4_.../checkpoint_mejor.pt \
                       --escenario zeroshot_exp4

    # Evaluar todos los checkpoints sobre un escenario:
    python evaluate.py --todos --escenario indistribucion

Salida:
    ~/evaluaciones/{escenario}/{nombre_checkpoint}.json   <- metricas individuales
    ~/evaluaciones/resumen_{escenario}.csv                <- tabla comparativa
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.rasa   import RASA
from models.serm   import SERM
from models.gating import GatingNetwork

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

EMBEDDINGS_TEST_DIR = Path.home() / "embeddings_test"
ENTRENAMIENTOS_DIR  = Path.home() / "entrenamientos"
EVALUACIONES_DIR    = Path.home() / "evaluaciones"

DIMS = {
    "wav2vec2bert": 1024,
    "wavlm":        1024,
    "whisper":      1280,
}

# Parametros minDCF — ASVspoof 2019 LA
P_TARGET  = 0.05
C_MISS    = 1.0
C_FA      = 10.0

# ---------------------------------------------------------------------------
# Dataset de test
# ---------------------------------------------------------------------------

class TestDataset(Dataset):
    """
    Carga embeddings de test desde el indice generado por extraer_embeddings_test.py.
    El indice ya contiene label, db, idioma, escenario por entrada.
    """

    def __init__(self, escenario: str):
        self.muestras = []  # (pt_w2v, pt_wavlm, pt_whisper, label, db, idioma)

        indice_path = EMBEDDINGS_TEST_DIR / f"indice_test_{escenario}.csv"
        if not indice_path.exists():
            raise FileNotFoundError(
                f"No se encontro el indice de test: {indice_path}\n"
                "Ejecuta primero extraer_embeddings_test.py para este escenario."
            )

        encontrados = ausentes = 0
        with open(indice_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pt_w2v    = Path(row["wav2vec2bert_pt"])
                pt_wavlm  = Path(row["wavlm_pt"])
                pt_whisper = Path(row["whisper_pt"])

                if pt_w2v.exists() and pt_wavlm.exists() and pt_whisper.exists():
                    label = 1 if row["label"] == "spoof" else 0
                    self.muestras.append((
                        pt_w2v, pt_wavlm, pt_whisper,
                        label, row["db"], row["idioma"],
                    ))
                    encontrados += 1
                else:
                    ausentes += 1

        print(f"  Escenario '{escenario}': {encontrados:,} muestras | {ausentes:,} sin embeddings")
        if encontrados == 0:
            raise RuntimeError("No se encontraron muestras. Verifica que extraer_embeddings_test.py haya terminado.")

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx):
        pt_w2v, pt_wavlm, pt_whisper, label, db, idioma = self.muestras[idx]
        emb_w2v    = torch.load(pt_w2v,    map_location="cpu", weights_only=True).float()
        emb_wavlm  = torch.load(pt_wavlm,  map_location="cpu", weights_only=True).float()
        emb_whisper = torch.load(pt_whisper, map_location="cpu", weights_only=True).float()
        return emb_w2v, emb_wavlm, emb_whisper, torch.tensor(label, dtype=torch.float32), db, idioma


# ---------------------------------------------------------------------------
# Modelo (identico a train.py)
# ---------------------------------------------------------------------------

class ModeloDeepfake(nn.Module):
    def __init__(self, M: int = 160, dims=DIMS, sin_rasa: bool = False):
        super().__init__()
        d_w2v   = dims["wav2vec2bert"]
        d_wavlm = dims["wavlm"]
        d_whisp = dims["whisper"]

        if sin_rasa:   # ablacion: RASA reemplazado por la identidad (ver train.py)
            self.rasa_w2v = self.rasa_wavlm = self.rasa_whisp = nn.Identity()
        else:
            self.rasa_w2v   = RASA(d_w2v,   M=M)
            self.rasa_wavlm = RASA(d_wavlm, M=M)
            self.rasa_whisp = RASA(d_whisp, M=M)

        self.head_w2v   = nn.Linear(d_w2v,   1)
        self.head_wavlm = nn.Linear(d_wavlm, 1)
        self.head_whisp = nn.Linear(d_whisp, 1)

        self.gating = GatingNetwork(dims_entrada=[d_w2v, d_wavlm, d_whisp], hidden_dim=256)

        d_concat = d_w2v + d_wavlm + d_whisp
        self.serm = SERM(embed_dim=d_concat, dim_poincare=128)

    def forward(self, emb_w2v, emb_wavlm, emb_whisper):
        e1 = self.rasa_w2v(emb_w2v)
        e2 = self.rasa_wavlm(emb_wavlm)
        e3 = self.rasa_whisp(emb_whisper)
        s1 = self.head_w2v(e1).squeeze(-1)
        s2 = self.head_wavlm(e2).squeeze(-1)
        s3 = self.head_whisp(e3).squeeze(-1)
        score_final = self.gating.fusionar(scores=[s1, s2, s3], embeddings=[e1, e2, e3])
        return score_final


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def calcular_eer(etiquetas: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate: umbral donde FAR = FRR."""
    umbrales = np.sort(np.unique(scores))
    mejor_diferencia = float("inf")
    mejor_eer = 1.0
    for u in umbrales:
        pred_spoof = scores >= u
        far = np.sum((pred_spoof == 1) & (etiquetas == 0)) / max(np.sum(etiquetas == 0), 1)
        frr = np.sum((pred_spoof == 0) & (etiquetas == 1)) / max(np.sum(etiquetas == 1), 1)
        diferencia = abs(far - frr)
        if diferencia < mejor_diferencia:
            mejor_diferencia = diferencia
            mejor_eer = (far + frr) / 2
    return float(mejor_eer)


def calcular_min_dcf(etiquetas: np.ndarray, scores: np.ndarray,
                     p_target=P_TARGET, c_miss=C_MISS, c_fa=C_FA) -> float:
    """
    minDCF normalizado. Parametros por defecto: ASVspoof 2019 LA.
    DCF = C_miss * P_target * FRR + C_fa * (1 - P_target) * FAR
    Normalizado por min(C_miss * P_target, C_fa * (1 - P_target)).
    """
    umbrales = np.sort(np.unique(scores))
    n_bonafide = np.sum(etiquetas == 0)
    n_spoof    = np.sum(etiquetas == 1)
    if n_bonafide == 0 or n_spoof == 0:
        return float("nan")

    dcf_min = float("inf")
    for u in umbrales:
        pred_spoof = scores >= u
        far = np.sum((pred_spoof == 1) & (etiquetas == 0)) / n_bonafide
        frr = np.sum((pred_spoof == 0) & (etiquetas == 1)) / n_spoof
        dcf = c_miss * p_target * frr + c_fa * (1 - p_target) * far
        if dcf < dcf_min:
            dcf_min = dcf

    norm = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(dcf_min / norm)


def calcular_auroc(etiquetas: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via regla del trapecio."""
    umbrales = np.sort(np.unique(scores))[::-1]
    n_pos = np.sum(etiquetas == 1)
    n_neg = np.sum(etiquetas == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tpr_prev = fpr_prev = 0.0
    auroc = 0.0
    for u in umbrales:
        pred = scores >= u
        tpr = np.sum((pred == 1) & (etiquetas == 1)) / n_pos
        fpr = np.sum((pred == 1) & (etiquetas == 0)) / n_neg
        auroc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr

    auroc += (1.0 - fpr_prev) * (1.0 + tpr_prev) / 2
    return float(auroc)


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

@torch.no_grad()
def inferir(modelo, loader, device):
    """Corre el modelo sobre el loader y devuelve scores, etiquetas, dbs, idiomas."""
    modelo.eval()
    all_scores = []
    all_labels = []
    all_dbs    = []
    all_idiomas = []

    for emb_w2v, emb_wavlm, emb_whisper, etiquetas, dbs, idiomas in tqdm(loader, desc="eval"):
        emb_w2v     = emb_w2v.to(device)
        emb_wavlm   = emb_wavlm.to(device)
        emb_whisper = emb_whisper.to(device)

        scores = torch.sigmoid(modelo(emb_w2v, emb_wavlm, emb_whisper)).cpu().numpy()
        all_scores.extend(scores.tolist())
        all_labels.extend(etiquetas.numpy().tolist())
        all_dbs.extend(dbs)
        all_idiomas.extend(idiomas)

    return (np.array(all_scores), np.array(all_labels),
            np.array(all_dbs), np.array(all_idiomas))


# ---------------------------------------------------------------------------
# Evaluacion de un checkpoint
# ---------------------------------------------------------------------------

def evaluar_checkpoint(ckpt_path: Path, escenario: str, device: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Checkpoint : {ckpt_path.parent.name}/{ckpt_path.name}")
    print(f"Escenario  : {escenario}")
    print(f"{'='*60}")

    # Cargar modelo
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config_ck = ck.get("config", {})
    M = config_ck.get("M", 160)
    modelo = ModeloDeepfake(M=M, sin_rasa=config_ck.get("sin_rasa", False)).to(device)
    modelo.load_state_dict(ck["model_state"])
    modelo.eval()

    # Dataset y loader
    ds = TestDataset(escenario)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4,
                        collate_fn=lambda batch: (
                            torch.stack([b[0] for b in batch]),
                            torch.stack([b[1] for b in batch]),
                            torch.stack([b[2] for b in batch]),
                            torch.stack([b[3] for b in batch]),
                            [b[4] for b in batch],
                            [b[5] for b in batch],
                        ))

    scores, etiquetas, dbs, idiomas = inferir(modelo, loader, device)

    # Metricas globales
    preds = (scores >= 0.5).astype(float)
    accuracy = float(np.mean(preds == etiquetas))
    eer      = calcular_eer(etiquetas, scores)
    min_dcf  = calcular_min_dcf(etiquetas, scores)
    auroc    = calcular_auroc(etiquetas, scores)

    tp = float(np.sum((preds == 1) & (etiquetas == 1)))
    fp = float(np.sum((preds == 1) & (etiquetas == 0)))
    fn = float(np.sum((preds == 0) & (etiquetas == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

    print(f"\nResultados globales:")
    print(f"  Accuracy  : {accuracy*100:.2f}%")
    print(f"  Precision : {precision*100:.2f}%  (de los que predijo spoof, cuantos lo eran)")
    print(f"  Recall    : {recall*100:.2f}%  (de los spoof reales, cuantos detecto)")
    print(f"  F1        : {f1*100:.2f}%")
    print(f"  AUROC     : {auroc:.4f}")
    print(f"  EER       : {eer*100:.2f}%")
    print(f"  minDCF    : {min_dcf:.4f}")

    # Metricas por DB
    metricas_db = {}
    for db in np.unique(dbs):
        mask = dbs == db
        if mask.sum() < 10:
            continue
        metricas_db[db] = {
            "n":        int(mask.sum()),
            "accuracy": float(np.mean((scores[mask] >= 0.5) == etiquetas[mask])),
            "auroc":    calcular_auroc(etiquetas[mask], scores[mask]),
            "eer":      calcular_eer(etiquetas[mask], scores[mask]),
        }

    # Metricas por idioma (util para zero-shot)
    metricas_idioma = {}
    for idioma in np.unique(idiomas):
        mask = idiomas == idioma
        if mask.sum() < 10:
            continue
        metricas_idioma[idioma] = {
            "n":        int(mask.sum()),
            "accuracy": float(np.mean((scores[mask] >= 0.5) == etiquetas[mask])),
            "auroc":    calcular_auroc(etiquetas[mask], scores[mask]),
            "eer":      calcular_eer(etiquetas[mask], scores[mask]),
        }

    resultado = {
        "checkpoint":    str(ckpt_path),
        "exp":           config_ck.get("exp"),
        "lambda_orth":   config_ck.get("lambda_orth"),
        "lambda_reco":   config_ck.get("lambda_reco"),
        "escenario":     escenario,
        "n_total":       len(etiquetas),
        "accuracy":      accuracy,
        "precision":     precision,
        "recall":        recall,
        "f1":            f1,
        "auroc":         auroc,
        "eer":           eer,
        "min_dcf":       min_dcf,
        "por_db":        metricas_db,
        "por_idioma":    metricas_idioma,
    }

    return resultado


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evalua modelos de deteccion de audio deepfake sobre conjuntos de test"
    )
    parser.add_argument("--checkpoint", type=Path,
                        help="Ruta al checkpoint_mejor.pt")
    parser.add_argument("--todos", action="store_true",
                        help="Evaluar todos los checkpoints en ~/entrenamientos/")
    parser.add_argument("--escenario", required=True,
                        help="Escenario de test: cualquiera con indice ~/embeddings_test/indice_test_<escenario>.csv "
                             "(indistribucion, fueradominio, compresion_mp3_32kbps, zeroshot_exp4, "
                             "zeroshot_v2, zs_limpio, zs_mp3_32kbps, zs_plr10, ...)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint and not args.todos:
        parser.error("Especifica --checkpoint o --todos")

    # Recopilar checkpoints a evaluar
    if args.todos:
        ckpts = sorted(ENTRENAMIENTOS_DIR.glob("*/checkpoint_mejor.pt"))
        print(f"Evaluando {len(ckpts)} checkpoints...")
    else:
        ckpts = [args.checkpoint]

    # Directorio de salida
    out_dir = EVALUACIONES_DIR / args.escenario
    out_dir.mkdir(parents=True, exist_ok=True)

    resultados = []
    for ckpt in ckpts:
        try:
            res = evaluar_checkpoint(ckpt, args.escenario, args.device)
            resultados.append(res)

            # Guardar JSON individual
            nombre = ckpt.parent.name
            json_path = out_dir / f"{nombre}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            print(f"  -> Guardado: {json_path}")

        except Exception as e:
            print(f"  ERROR en {ckpt}: {e}")

    # Guardar resumen CSV
    if resultados:
        resumen_path = EVALUACIONES_DIR / f"resumen_{args.escenario}.csv"
        campos = ["checkpoint", "exp", "lambda_orth", "lambda_reco",
                  "n_total", "accuracy", "precision", "recall", "f1", "auroc", "eer", "min_dcf"]
        with open(resumen_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
            writer.writeheader()
            for r in resultados:
                r["checkpoint"] = Path(r["checkpoint"]).parent.name
                writer.writerow(r)

        print(f"\nResumen guardado en: {resumen_path}")

        # Imprimir tabla final
        print(f"\n{'Checkpoint':<35} {'Exp':>4} {'lambda':>6} {'Acc':>7} {'AUROC':>7} {'EER':>7} {'minDCF':>8}")
        print("-" * 80)
        for r in sorted(resultados, key=lambda x: x["eer"]):
            nombre = Path(r["checkpoint"]).name if Path(r["checkpoint"]).name != Path(r["checkpoint"]).parent.name else r["checkpoint"]
            print(
                f"{nombre:<35} {str(r['exp']):>4} {str(r['lambda_orth']):>6} "
                f"{r['accuracy']*100:>6.2f}% {r['auroc']:>7.4f} "
                f"{r['eer']*100:>6.2f}% {r['min_dcf']:>8.4f}"
            )


if __name__ == "__main__":
    main()
