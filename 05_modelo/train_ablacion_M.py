"""
train_ablacion_M.py
Ablación del tamaño del style bank (M) en RASA.

Entrena 6 modelos idénticos salvo por M, secuencialmente, usando la misma
infraestructura que train.py. Cada run produce su propio directorio en
~/entrenamientos/ablacion_M{m}_{timestamp}/.

Uso:
    python train_ablacion_M.py --exp 3
    python train_ablacion_M.py --exp 3 --epochs 50 --device cuda
    python train_ablacion_M.py --exp 3 --m-values 32 64 128 160  # subconjunto

Los valores de M por defecto cubren por debajo y por encima del óptimo del paper:
    M ∈ {16, 32, 64, 128, 160, 256}
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.rasa   import RASA
from models.serm   import SERM
from models.gating import GatingNetwork

# Reutilizar helpers de train.py directamente
from train import (
    PARTICIONES,
    EMBEDDINGS_DIR,
    SALIDA_BASE,
    DIMS,
    VAL_PCT,
    EmbeddingDataset,
    entrenar_epoch,
    evaluar,
    guardar_checkpoint,
    cargar_checkpoint,
)

M_DEFAULT = [16, 32, 64, 128, 160, 256]


# ---------------------------------------------------------------------------
# Modelo parametrizable por M
# ---------------------------------------------------------------------------

class ModeloDeepfakeM(nn.Module):
    """
    Igual que ModeloDeepfake de train.py pero con M configurable.
    """

    def __init__(self, M: int, dims=DIMS):
        super().__init__()
        self.M = M
        d_w2v   = dims["wav2vec2bert"]
        d_wavlm = dims["wavlm"]
        d_whisp = dims["whisper"]

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

        score_final = self.gating.fusionar(
            scores=[s1, s2, s3],
            embeddings=[e1, e2, e3],
        )
        return score_final, [e1, e2, e3]

    def perdidas_auxiliares(self, emb_list, etiquetas):
        e1, e2, e3 = emb_list

        l_orth = (self.rasa_w2v.orthogonal_loss()
                  + self.rasa_wavlm.orthogonal_loss()
                  + self.rasa_whisp.orthogonal_loss()) / 3

        l_reco = (self.rasa_w2v.reconstruction_loss(e1)
                  + self.rasa_wavlm.reconstruction_loss(e2)
                  + self.rasa_whisp.reconstruction_loss(e3)) / 3

        e_concat = torch.cat([e1, e2, e3], dim=-1)
        l_serm = self.serm.perdida_total(e_concat, etiquetas.long())

        return {"l_orth": l_orth, "l_reco": l_reco, "l_serm": l_serm}


# ---------------------------------------------------------------------------
# Entrenamiento de un valor de M
# ---------------------------------------------------------------------------

def entrenar_un_M(M: int, args, ts_global: str):
    """
    Entrena el modelo completo para un valor de M dado.
    Retorna el mejor val_loss alcanzado.
    """
    out_dir = SALIDA_BASE / f"ablacion_M{M}_{ts_global}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging independiente por run (archivo + consola)
    log = logging.getLogger(f"M{M}")
    log.setLevel(logging.INFO)
    log.propagate = False
    if not log.handlers:
        fh = logging.FileHandler(out_dir / "entrenamiento.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(f"[M={M}] %(asctime)s %(message)s"))
        log.addHandler(fh)
        log.addHandler(sh)

    log.info("=" * 60)
    log.info(f"ABLACION M={M}  |  Exp-{args.exp}")
    log.info(f"Salida: {out_dir}")
    log.info(f"Device: {args.device} | Epochs: {args.epochs} | BS: {args.batch_size} | LR: {args.lr}")
    log.info("=" * 60)

    config = {**vars(args), "M": M, "out_dir": str(out_dir)}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    lambdas = {
        "orth": args.lambda_orth,
        "reco": args.lambda_reco,
        "serm": args.lambda_serm,
    }

    particion_dir = PARTICIONES[args.exp]
    ds_train = EmbeddingDataset(particion_dir, "train", args.val_pct)
    ds_val   = EmbeddingDataset(particion_dir, "val",   args.val_pct)

    pin = args.device == "cuda"
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=pin)

    log.info(f"Train: {len(ds_train):,} muestras | Val: {len(ds_val):,} muestras")

    modelo = ModeloDeepfakeM(M=M).to(args.device)
    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    log.info(f"Parametros entrenables: {n_params:,}  (M={M})")

    optimizer = optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, verbose=True
    )
    criterio = nn.BCEWithLogitsLoss()

    metricas_path = out_dir / "metricas.csv"
    with open(metricas_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_bce", "train_orth", "train_reco", "train_serm", "train_total",
            "val_bce", "val_total", "val_accuracy", "lr"
        ])

    mejor_val_loss = float("inf")
    ck_ultimo = out_dir / "checkpoint_ultimo.pt"
    ck_mejor  = out_dir / "checkpoint_mejor.pt"

    for epoch in range(1, args.epochs + 1):
        log.info(f"--- Epoch {epoch}/{args.epochs} ---")

        train_m = entrenar_epoch(modelo, dl_train, optimizer, criterio, args.device, lambdas)
        val_m   = evaluar(modelo, dl_val, criterio, args.device, lambdas)

        lr_actual = optimizer.param_groups[0]["lr"]
        scheduler.step(val_m["total"])

        log.info(
            f"Train: bce={train_m['bce']:.4f} orth={train_m['orth']:.4f} "
            f"reco={train_m['reco']:.4f} serm={train_m['serm']:.4f} total={train_m['total']:.4f}"
        )
        log.info(
            f"Val:   bce={val_m['bce']:.4f} total={val_m['total']:.4f} "
            f"accuracy={val_m['accuracy']:.4f} | LR={lr_actual:.2e}"
        )

        with open(metricas_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_m['bce']:.6f}", f"{train_m['orth']:.6f}",
                f"{train_m['reco']:.6f}", f"{train_m['serm']:.6f}", f"{train_m['total']:.6f}",
                f"{val_m['bce']:.6f}", f"{val_m['total']:.6f}",
                f"{val_m['accuracy']:.6f}", f"{lr_actual:.2e}",
            ])

        guardar_checkpoint(ck_ultimo, modelo, optimizer, scheduler, epoch, mejor_val_loss, config)

        if val_m["total"] < mejor_val_loss:
            mejor_val_loss = val_m["total"]
            guardar_checkpoint(ck_mejor, modelo, optimizer, scheduler, epoch, mejor_val_loss, config)
            log.info(f"  -> Nuevo mejor modelo (val_loss={mejor_val_loss:.4f})")

    log.info(f"M={M} completado. Mejor val_loss: {mejor_val_loss:.4f}")
    log.info(f"Checkpoints en: {out_dir}")
    return mejor_val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ablacion del tamaño del style bank M en RASA"
    )
    parser.add_argument("--exp",      type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--bs",       type=int, default=256, dest="batch_size")
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-pct",  type=int, default=VAL_PCT, dest="val_pct")
    parser.add_argument("--lambda-orth", type=float, default=1.0)
    parser.add_argument("--lambda-reco", type=float, default=1.0)
    parser.add_argument("--lambda-serm", type=float, default=0.05)
    parser.add_argument(
        "--m-values", type=int, nargs="+", default=M_DEFAULT,
        metavar="M",
        help=f"Valores de M a probar (default: {M_DEFAULT})"
    )
    args = parser.parse_args()

    # Un timestamp compartido para agrupar todos los runs de esta ablación
    ts_global = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Logging raíz (resumen global)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info(f"Ablacion M  |  Exp-{args.exp}  |  M={args.m_values}")
    logging.info(f"Epochs={args.epochs} | BS={args.batch_size} | LR={args.lr}")
    logging.info(f"λ_orth={args.lambda_orth} λ_reco={args.lambda_reco} λ_serm={args.lambda_serm}")
    logging.info(f"Timestamp global: {ts_global}")
    logging.info("")

    resultados = {}

    for M in args.m_values:
        logging.info(f"{'='*60}")
        logging.info(f"INICIANDO M={M}  ({args.m_values.index(M)+1}/{len(args.m_values)})")
        logging.info(f"{'='*60}")

        mejor = entrenar_un_M(M, args, ts_global)
        resultados[M] = mejor

        logging.info(f"M={M} terminado. Mejor val_loss={mejor:.4f}")
        logging.info("")

        # Liberar memoria GPU entre runs
        torch.cuda.empty_cache()

    # Resumen final
    logging.info("=" * 60)
    logging.info("RESUMEN ABLACION M")
    logging.info("=" * 60)
    logging.info(f"{'M':>6}  {'mejor_val_loss':>16}")
    logging.info("-" * 26)
    for M, loss in sorted(resultados.items()):
        marker = " <-- paper" if M == 160 else ""
        logging.info(f"{M:>6}  {loss:>16.4f}{marker}")

    mejor_M = min(resultados, key=resultados.get)
    logging.info("")
    logging.info(f"Mejor M encontrado: {mejor_M}  (val_loss={resultados[mejor_M]:.4f})")
    logging.info("")
    logging.info("Siguiente paso: evaluate.py --checkpoint ablacion_M{M}_{ts}/checkpoint_mejor.pt")


if __name__ == "__main__":
    main()
