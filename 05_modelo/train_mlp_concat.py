"""
train_mlp_concat.py (R6 / pregunta del jurado: ¿aporta la arquitectura RASA + Gating?)

Linea base sin RASA, sin Gating y sin SERM: un MLP (o regresion logistica con --hidden 0)
sobre la concatenacion de los 3 embeddings de los modelos de base. Misma particion
train/val, mismo optimizador, scheduler, epocas y criterio de seleccion que train.py
(menor val_bce), para que la comparacion con el modelo completo sea justa.

Despues de entrenar evalua en los escenarios pedidos y guarda resultados.json.

Uso:
  python train_mlp_concat.py --exp 4 --seed 1 --device cuda:1
  python train_mlp_concat.py --exp 4 --seed 1 --hidden 0        # regresion logistica
"""
import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train import (EmbeddingDataset, PARTICIONES, SALIDA_BASE, DIMS, VAL_PCT, fijar_semilla)
from evaluate import TestDataset, calcular_eer, calcular_auroc


class MLPConcat(nn.Module):
    """Concatena los 3 embeddings y los clasifica con un MLP (hidden>0) o una capa lineal (hidden=0)."""

    def __init__(self, dims=DIMS, hidden: int = 256):
        super().__init__()
        d = dims["wav2vec2bert"] + dims["wavlm"] + dims["whisper"]
        if hidden > 0:
            self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.net = nn.Linear(d, 1)

    def forward(self, e1, e2, e3):
        return self.net(torch.cat([e1, e2, e3], dim=-1)).squeeze(-1)


def collate_test(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]), torch.stack([b[3] for b in batch]),
            [b[4] for b in batch], [b[5] for b in batch])


def epoch_train(modelo, loader, opt, crit, device):
    modelo.train()
    tot = n = 0
    for e1, e2, e3, y in tqdm(loader, desc="train", leave=False):
        e1, e2, e3, y = e1.to(device), e2.to(device), e3.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(modelo(e1, e2, e3), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        opt.step()
        tot += loss.item() * y.size(0)
        n += y.size(0)
    return tot / n


@torch.no_grad()
def epoch_val(modelo, loader, crit, device):
    modelo.eval()
    tot = ok = n = 0
    for e1, e2, e3, y in tqdm(loader, desc="val", leave=False):
        e1, e2, e3, y = e1.to(device), e2.to(device), e3.to(device), y.to(device)
        s = modelo(e1, e2, e3)
        tot += crit(s, y).item() * y.size(0)
        ok += ((torch.sigmoid(s) > 0.5).float() == y).sum().item()
        n += y.size(0)
    return tot / n, ok / n


@torch.no_grad()
def evaluar_escenario(modelo, escenario, device):
    ds = TestDataset(escenario)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, collate_fn=collate_test)
    modelo.eval()
    S, Y, ID = [], [], []
    for e1, e2, e3, y, _dbs, idiomas in tqdm(loader, desc=f"test {escenario}", leave=False):
        S.append(modelo(e1.to(device), e2.to(device), e3.to(device)).cpu().numpy())
        Y.append(y.numpy())
        ID.extend(idiomas)
    s, y, idi = np.concatenate(S), np.concatenate(Y), np.array(ID)
    r = {"n": int(len(y)), "eer": calcular_eer(y, s) * 100, "auroc": calcular_auroc(y, s),
         "accuracy": float(np.mean((s >= 0) == y))}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, default=4, choices=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=256, help="0 = regresion logistica")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=256, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--escenarios", nargs="+",
                    default=["indistribucion", "fueradominio", "zeroshot_exp4", "zs_limpio",
                             "zs_mp3_32kbps", "zs_plr10"])
    args = ap.parse_args()

    if args.seed is not None:
        fijar_semilla(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tipo = "mlp" if args.hidden > 0 else "logreg"
    out_dir = SALIDA_BASE / f"exp{args.exp}{tipo}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "entrenamiento.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    logging.info(f"Salida: {out_dir}")
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    part = PARTICIONES[args.exp]
    ds_tr = EmbeddingDataset(part, "train", VAL_PCT)
    ds_va = EmbeddingDataset(part, "val", VAL_PCT)
    pin = args.device.startswith("cuda")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)
    logging.info(f"Train: {len(ds_tr):,} | Val: {len(ds_va):,}")

    modelo = MLPConcat(hidden=args.hidden).to(args.device)
    logging.info(f"Parametros: {sum(p.numel() for p in modelo.parameters()):,}")
    opt = optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=10, factor=0.5)
    crit = nn.BCEWithLogitsLoss()

    csv_path = out_dir / "metricas.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_bce", "val_bce", "val_accuracy", "lr"])

    mejor = float("inf")
    ck = out_dir / "checkpoint_mejor.pt"
    for ep in range(1, args.epochs + 1):
        tr = epoch_train(modelo, dl_tr, opt, crit, args.device)
        vb, va = epoch_val(modelo, dl_va, crit, args.device)
        lr = opt.param_groups[0]["lr"]
        sched.step(vb)
        logging.info(f"Epoch {ep}/{args.epochs} train_bce={tr:.4f} val_bce={vb:.4f} val_acc={va:.4f} lr={lr:.2e}")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tr:.6f}", f"{vb:.6f}", f"{va:.6f}", f"{lr:.2e}"])
        if vb < mejor:
            mejor = vb
            torch.save({"epoch": ep, "model_state": modelo.state_dict(), "config": vars(args)}, ck)

    # Evaluacion con el mejor checkpoint (criterio igual que train.py: menor val_bce)
    modelo.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model_state"])
    resultados = {"checkpoint": str(ck), "val_bce_mejor": mejor, "escenarios": {}}
    for esc in args.escenarios:
        try:
            resultados["escenarios"][esc] = evaluar_escenario(modelo, esc, args.device)
            r = resultados["escenarios"][esc]
            print(f"RESULTADO {out_dir.name} {esc}: EER {r['eer']:.2f}% AUROC {r['auroc']:.4f} (n={r['n']:,})")
        except Exception as e:  # un escenario sin embeddings no debe tumbar el resto
            print(f"RESULTADO {out_dir.name} {esc}: ERROR {e}")
    with open(out_dir / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    logging.info("Terminado.")


if __name__ == "__main__":
    main()
