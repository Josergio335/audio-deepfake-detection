# -*- coding: utf-8 -*-
"""
train_baseline.py
Entrenamiento de modelos baseline para comparacion (R8).

Baselines implementados (versiones SIMPLIFICADAS inspiradas en los modelos publicados,
no las implementaciones oficiales; en la tesis deben nombrarse asi, p. ej.
"RawNet2 simplificado", y no atribuirles los EER de los articulos originales):
  - lfcc_lcnn : LFCC (60 coef) + LCNN (Light CNN)
  - rawnet2   : filtros sinc + bloques residuales 1D + GRU, sobre waveform
  - aasist    : ramas espectral y temporal + una capa de atencion sobre grafo

Los tres leen audio WAV directamente desde disco usando las rutas del manifest
(solo se usan archivos que existen en disco) y se entrenan con la misma particion
train/val que train.py (hash determinista, semilla 42).

Uso:
    python train_baseline.py --modelo lfcc_lcnn --exp 4
    python train_baseline.py --modelo rawnet2   --exp 4 --max-horas 8 --device cuda:1
    python train_baseline.py --modelo aasist    --exp 4 --epochs 30
    python train_baseline.py --modelo rawnet2   --exp 4 --epochs 1 --limite 128   # prueba rapida

Salida:
    ~/entrenamientos_baseline/{modelo}_exp{N}_{timestamp}/
        checkpoint_mejor.pt
        checkpoint_ultimo.pt
        metricas.csv
        config.json
    ~/entrenamientos_baseline/ultimo_{modelo}.txt   <- ruta de la ultima corrida
"""

import argparse
import csv
import hashlib
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuracion de rutas (misma logica que train.py)
# ---------------------------------------------------------------------------

PARTICIONES = {
    1: Path.home() / "particiones/Exp1",
    2: Path.home() / "particiones/Exp2",
    3: Path.home() / "particiones/Exp3",
    4: Path.home() / "particiones/Exp4",
}

WIN_PREFIXES = [
    ("D:\\Bases de Datos procesadas\\",           Path.home() / "datos/Bases de Datos procesadas", False),
    ("D:\\Bases de Datos extraidas\\CommonVoice\\", Path.home() / "datos/Bases de Datos procesadas/CommonVoice", True),
    ("D:\\Bases de Datos extraidas\\",             Path.home() / "datos/Bases de Datos procesadas", True),
]

SALIDA_BASE   = Path.home() / "entrenamientos_baseline"
SR_TARGET     = 16_000   # Hz — igual que el preprocesamiento
MAX_SEGUNDOS  = 6.0      # truncar/padear a 6s (96,000 muestras)
SEMILLA_SPLIT = 42
VAL_PCT       = 20
ETIQUETAS     = {"bonafide": 0, "spoof": 1}

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def resolver_ruta(ruta_str: str) -> Path:
    """Convierte ruta Windows del manifest a ruta Linux real."""
    if ruta_str.startswith("/"):
        return Path(ruta_str)
    for win_pref, linux_base, a_wav in WIN_PREFIXES:
        if ruta_str.startswith(win_pref):
            rel = ruta_str[len(win_pref):].replace("\\", "/")
            p = linux_base / rel
            return p.with_suffix(".wav") if a_wav else p
    return Path(ruta_str)


def asignar_split(ruta: str, val_pct: int = VAL_PCT, semilla: int = SEMILLA_SPLIT) -> str:
    h = int(hashlib.md5(f"{semilla}:{ruta}".encode()).hexdigest(), 16) % 100
    return "val" if h < val_pct else "train"


def cargar_audio(ruta: Path, max_muestras: int, estricto: bool = False) -> torch.Tensor:
    """
    Carga WAV, remuestrea a SR_TARGET si es necesario, convierte a mono.
    Devuelve tensor (max_muestras,) — truncado o cero-padeado.
    """
    try:
        wav, sr = torchaudio.load(str(ruta))
    except Exception as e:
        if estricto:
            raise RuntimeError(f"No se pudo leer el audio {ruta}: {e}") from e
        return torch.zeros(max_muestras)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR_TARGET:
        wav = torchaudio.functional.resample(wav, sr, SR_TARGET)
    wav = wav.squeeze(0)

    if wav.shape[0] >= max_muestras:
        wav = wav[:max_muestras]
    else:
        wav = torch.nn.functional.pad(wav, (0, max_muestras - wav.shape[0]))
    return wav


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """Lee audio WAV directamente desde disco usando rutas del manifest."""

    def __init__(self, particion_dir: Path, split: str, max_segundos: float = MAX_SEGUNDOS,
                 limite: int = 0):
        self.max_muestras = int(max_segundos * SR_TARGET)
        self.muestras = []  # (ruta_linux, etiqueta)
        faltantes = 0

        manifest_path = particion_dir / "manifest.csv"
        with open(manifest_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row_split = row.get("split") or asignar_split(row["ruta"])
                if row_split != split:
                    continue
                label_str = row.get("label", "")
                if label_str not in ETIQUETAS:
                    continue
                ruta_linux = resolver_ruta(row["ruta"])
                if not ruta_linux.exists():
                    faltantes += 1
                    continue
                self.muestras.append((ruta_linux, ETIQUETAS[label_str]))
                if limite and len(self.muestras) >= limite:
                    break

        logging.info(f"[AudioDataset {split}] {len(self.muestras):,} muestras "
                     f"({faltantes:,} omitidas por no existir el audio)")
        if not self.muestras:
            raise RuntimeError(f"Dataset vacio para split={split} en {particion_dir}")

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx):
        ruta, etiqueta = self.muestras[idx]
        wav = cargar_audio(ruta, self.max_muestras)
        return wav, torch.tensor(etiqueta, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Modelos baseline
# ---------------------------------------------------------------------------

# --- LFCC-LCNN ---

class LFCCFrontend(nn.Module):
    """Extrae LFCC (60 coeficientes) como en el baseline oficial ASVspoof 2019."""

    def __init__(self, sr: int = SR_TARGET, n_lfcc: int = 60, n_fft: int = 512,
                 win_length: int = 400, hop_length: int = 160):
        super().__init__()
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=sr,
            n_lfcc=n_lfcc,
            speckwargs={"n_fft": n_fft, "win_length": win_length, "hop_length": hop_length},
        )
        self.delta  = torchaudio.transforms.ComputeDeltas()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) -> (B, 3*n_lfcc, frames)"""
        feat   = self.lfcc(wav)        # (B, n_lfcc, frames)
        d1     = self.delta(feat)
        d2     = self.delta(d1)
        return torch.cat([feat, d1, d2], dim=1)  # (B, 3*n_lfcc, frames)


class LCNN(nn.Module):
    """
    Light CNN para clasificacion de anti-spoofing.
    Arquitectura de Wu et al. (2020), adaptada al paper de ASVspoof 2019.
    Entrada: (B, C, T) — salida: (B,) logit.
    """

    def __init__(self, in_channels: int = 180):
        super().__init__()

        def mfm_block(in_ch, out_ch, k=3, s=1, p=1):
            # Max-Feature-Map (MFM): convolucion con 2x canales, luego max entre pares
            return nn.Conv2d(in_ch, out_ch * 2, k, s, p)

        self.conv1 = mfm_block(1, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = mfm_block(32, 48)
        self.conv3 = mfm_block(48, 48)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = mfm_block(48, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv5 = mfm_block(64, 64)
        self.pool4 = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(64 * 4 * 4, 1)
        self.drop  = nn.Dropout(0.5)

    @staticmethod
    def mfm(x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1] // 2
        return torch.max(x[:, :c], x[:, c:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> unsqueeze a (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.mfm(self.conv1(x)); x = self.pool1(x)
        x = self.mfm(self.conv2(x))
        x = self.mfm(self.conv3(x)); x = self.pool2(x)
        x = self.mfm(self.conv4(x)); x = self.pool3(x)
        x = self.mfm(self.conv5(x)); x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        return self.fc(x).squeeze(-1)


class LFCC_LCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = LFCCFrontend(n_lfcc=60)
        self.lcnn     = LCNN(in_channels=180)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feat = self.frontend(wav)   # (B, 180, frames)
        return self.lcnn(feat)


# --- RawNet2 ---
# Basado en: Tak et al. "End-to-End Anti-Spoofing with RawNet2", ICASSP 2021

class SincConv(nn.Module):
    """Capa Sinc-based filterbank (version simplificada compatible con RawNet2)."""

    def __init__(self, out_channels: int = 128, kernel_size: int = 1024, sr: int = SR_TARGET):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.sr           = sr

        # Frecuencias iniciales distribuidas en mel-scale
        low_hz  = 30.0
        mel_low  = 2595 * np.log10(1 + low_hz / 700)
        mel_high = 2595 * np.log10(1 + sr / 2 / 700)
        mel_pts  = np.linspace(mel_low, mel_high, out_channels + 2)
        hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)

        self.f1 = nn.Parameter(torch.FloatTensor(hz_pts[:-2]).unsqueeze(1))
        self.f2 = nn.Parameter(torch.FloatTensor(hz_pts[1:-1]).unsqueeze(1))

        n = torch.arange(-(kernel_size // 2), kernel_size // 2, dtype=torch.float32)
        self.register_buffer("n", n)
        self.register_buffer("window", torch.hamming_window(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = torch.abs(self.f1) / self.sr
        f2 = (torch.abs(self.f1) + torch.abs(self.f2 - self.f1)) / self.sr
        n  = self.n.unsqueeze(0)
        band_pass = (2 * f2 * torch.sinc(2 * f2 * n)
                   - 2 * f1 * torch.sinc(2 * f1 * n))
        band_pass = band_pass * self.window
        band_pass = band_pass / (band_pass.norm(dim=1, keepdim=True) + 1e-8)
        filters = band_pass.unsqueeze(1)   # (C, 1, K)
        return torch.nn.functional.conv1d(x, filters, padding=self.kernel_size // 2)


class ResBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bn1  = nn.BatchNorm1d(channels)
        self.lrelu = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.lrelu(self.bn1(x))
        x = self.conv1(x)
        x = self.lrelu(self.bn2(x))
        x = self.conv2(x)
        return x + residual


class RawNet2(nn.Module):
    """
    RawNet2 simplificado (NO es la implementacion oficial de Tak et al., ICASSP 2021).
    Filtros sinc, 4 bloques residuales 1D con reduccion temporal y una GRU de 1024 unidades.
    La senal se reduce ~243 veces antes de la GRU (6 s -> ~395 pasos) para que el
    entrenamiento sea viable.
    """

    def __init__(self):
        super().__init__()
        self.sinc    = SincConv(out_channels=128, kernel_size=1024)
        self.pool0   = nn.MaxPool1d(3)
        self.bn_sinc = nn.BatchNorm1d(128)
        self.lrelu   = nn.LeakyReLU(0.3)

        self.bloques = nn.ModuleList([ResBlock1D(128) for _ in range(4)])
        self.pool    = nn.MaxPool1d(3)

        self.gru = nn.GRU(input_size=128, hidden_size=1024, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(1024, 1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.unsqueeze(1)                        # (B, 1, T)
        x = torch.abs(self.sinc(x))                 # (B, 128, T)
        x = self.pool0(x)
        x = self.lrelu(self.bn_sinc(x))
        for bloque in self.bloques:
            x = self.pool(bloque(x))                # cada bloque reduce 3 veces
        x = x.transpose(1, 2)                       # (B, T', 128)
        _, h = self.gru(x)                          # h: (1, B, 1024)
        return self.fc(h[-1]).squeeze(-1)           # (B,)


# --- AASIST simplificado ---
# Jung et al. "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
# Graph Attention Networks", ICASSP 2022

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W  = nn.Linear(in_dim, out_dim, bias=False)
        self.a  = nn.Linear(2 * out_dim, 1, bias=False)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        h  = self.W(x)                                   # (B, N, out)
        N  = h.shape[1]
        h1 = h.unsqueeze(2).expand(-1, -1, N, -1)        # (B, N, N, out)
        h2 = h.unsqueeze(1).expand(-1, N, -1, -1)        # (B, N, N, out)
        e  = self.act(self.a(torch.cat([h1, h2], dim=-1)).squeeze(-1))  # (B,N,N)
        alpha = torch.softmax(e, dim=-1)
        return torch.bmm(alpha, h)                        # (B, N, out)


class AASIST(nn.Module):
    """
    AASIST simplificado (NO es la implementacion oficial de Jung et al., ICASSP 2022):
    ramas espectral y temporal, una capa de atencion sobre grafo y un clasificador lineal.
    """

    def __init__(self):
        super().__init__()
        # Espectro-rama (RawNet2-style encoder)
        self.sinc    = SincConv(out_channels=70, kernel_size=128)
        self.bn_sinc = nn.BatchNorm1d(70)
        self.lrelu   = nn.LeakyReLU(0.3)

        self.spec_encoder = nn.Sequential(
            nn.Conv1d(70, 128, 3, padding=1), nn.BatchNorm1d(128), nn.LeakyReLU(0.3),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.LeakyReLU(0.3),
            nn.AdaptiveAvgPool1d(32),
        )
        # Temporal-rama
        self.temp_encoder = nn.Sequential(
            nn.Conv1d(1, 64, 128, stride=64), nn.BatchNorm1d(64), nn.LeakyReLU(0.3),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.LeakyReLU(0.3),
            nn.AdaptiveAvgPool1d(32),
        )
        # Graph attention
        self.gat = GraphAttentionLayer(in_dim=128, out_dim=128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # Rama espectral
        x_s = wav.unsqueeze(1)                           # (B, 1, T)
        x_s = self.lrelu(self.bn_sinc(self.sinc(x_s)))  # (B, 70, T)
        x_s = self.spec_encoder(x_s)                     # (B, 128, 32)

        # Rama temporal
        x_t = wav.unsqueeze(1)
        x_t = self.temp_encoder(x_t)                     # (B, 128, 32)

        # Fusion + GAT
        x = (x_s + x_t).transpose(1, 2)                 # (B, 32, 128)
        x = self.gat(x)                                  # (B, 32, 128)
        x = x.transpose(1, 2)                            # (B, 128, 32)
        x = self.pool(x).squeeze(-1)                     # (B, 128)
        x = self.drop(x)
        return self.fc(x).squeeze(-1)                    # (B,)


MODELOS = {
    "lfcc_lcnn": LFCC_LCNN,
    "rawnet2":   RawNet2,
    "aasist":    AASIST,
}

# ---------------------------------------------------------------------------
# Bucles de entrenamiento y validacion
# ---------------------------------------------------------------------------

def entrenar_epoca(modelo, loader, optimizer, criterio, device):
    modelo.train()
    total_loss = correctos = n = 0
    for wav, etiquetas in tqdm(loader, desc="train", leave=False):
        wav      = wav.to(device)
        etiquetas = etiquetas.to(device)
        optimizer.zero_grad()
        logits = modelo(wav)
        loss   = criterio(logits, etiquetas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(wav)
        correctos  += ((logits > 0) == etiquetas.bool()).sum().item()
        n          += len(wav)
    return total_loss / n, correctos / n


def validar_epoca(modelo, loader, criterio, device):
    modelo.eval()
    total_loss = correctos = n = 0
    with torch.no_grad():
        for wav, etiquetas in tqdm(loader, desc="val", leave=False):
            wav       = wav.to(device)
            etiquetas = etiquetas.to(device)
            logits    = modelo(wav)
            loss      = criterio(logits, etiquetas)
            total_loss += loss.item() * len(wav)
            correctos  += ((logits > 0) == etiquetas.bool()).sum().item()
            n          += len(wav)
    return total_loss / n, correctos / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de baselines (R8)")
    parser.add_argument("--modelo", choices=list(MODELOS), required=True,
                        help="Baseline a entrenar: lfcc_lcnn | rawnet2 | aasist")
    parser.add_argument("--exp",    type=int, choices=[1, 2, 3, 4], required=True,
                        help="Experimento de particion (1=252h, 2=434h, 3=602h, 4=corpus del Exp-4)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bs",     type=int, default=32, dest="batch_size")
    parser.add_argument("--lr",     type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed",   type=int, default=42,
                        help="Semilla de inicializacion y shuffle")
    parser.add_argument("--max-horas", type=float, default=None, dest="max_horas",
                        help="Presupuesto de tiempo: no inicia una epoca que no quepa "
                             "y conserva el mejor checkpoint")
    parser.add_argument("--limite", type=int, default=0,
                        help="Solo para pruebas: usa N muestras de train y de val")
    parser.add_argument("--reanudar", action="store_true",
                        help="Continuar desde checkpoint_ultimo.pt si existe")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    sufijo  = "_prueba" if args.limite else ""
    out_dir = SALIDA_BASE / f"{args.modelo}_exp{args.exp}{sufijo}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (SALIDA_BASE / f"ultimo_{args.modelo}.txt").write_text(str(out_dir))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )

    logging.info("=" * 60)
    logging.info(f"BASELINE: {args.modelo.upper()}  |  Exp-{args.exp}")
    logging.info(f"Salida: {out_dir}")
    logging.info("=" * 60)

    config = vars(args)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    particion_dir = PARTICIONES[args.exp]
    ds_train = AudioDataset(particion_dir, "train", limite=args.limite)
    ds_val   = AudioDataset(particion_dir, "val",   limite=args.limite)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device)
    modelo = MODELOS[args.modelo]().to(device)
    logging.info(f"Parametros: {sum(p.numel() for p in modelo.parameters()):,}")

    optimizer = optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterio  = nn.BCEWithLogitsLoss()

    inicio_epoca = 1
    mejor_val_loss = float("inf")

    if args.reanudar:
        ckpt_path = out_dir.parent / f"{args.modelo}_exp{args.exp}_ultimo.pt"
        if not ckpt_path.exists():
            # Buscar el mas reciente
            candidatos = sorted(SALIDA_BASE.glob(f"{args.modelo}_exp{args.exp}_*/checkpoint_ultimo.pt"))
            if candidatos:
                ckpt_path = candidatos[-1]
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            modelo.load_state_dict(ckpt["modelo"])
            optimizer.load_state_dict(ckpt["optimizer"])
            inicio_epoca = ckpt["epoca"] + 1
            mejor_val_loss = ckpt.get("mejor_val_loss", float("inf"))
            logging.info(f"Reanudando desde epoca {inicio_epoca}")

    metricas_path = out_dir / "metricas.csv"
    with open(metricas_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoca", "train_loss", "train_acc", "val_loss", "val_acc"])

    t_inicio = time.time()
    for epoca in range(inicio_epoca, args.epochs + 1):
        t_epoca = time.time()
        tr_loss, tr_acc = entrenar_epoca(modelo, dl_train, optimizer, criterio, device)
        vl_loss, vl_acc = validar_epoca(modelo, dl_val, criterio, device)
        scheduler.step()
        dur_epoca = time.time() - t_epoca

        logging.info(
            f"Epoca {epoca:3d}/{args.epochs} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val   loss={vl_loss:.4f} acc={vl_acc:.4f} | {dur_epoca / 60:.1f} min"
        )

        with open(metricas_path, "a", newline="") as f:
            csv.writer(f).writerow([epoca, f"{tr_loss:.5f}", f"{tr_acc:.5f}",
                                    f"{vl_loss:.5f}", f"{vl_acc:.5f}"])

        estado = {
            "epoca": epoca,
            "modelo": modelo.state_dict(),
            "optimizer": optimizer.state_dict(),
            "mejor_val_loss": mejor_val_loss,
        }
        torch.save(estado, out_dir / "checkpoint_ultimo.pt")

        if vl_loss < mejor_val_loss:
            mejor_val_loss = vl_loss
            estado["mejor_val_loss"] = mejor_val_loss
            torch.save(estado, out_dir / "checkpoint_mejor.pt")
            logging.info(f"  -> Mejor modelo guardado (val_loss={vl_loss:.4f})")

        if args.max_horas is not None:
            transcurrido = time.time() - t_inicio
            if transcurrido + dur_epoca > args.max_horas * 3600:
                logging.info(f"Presupuesto de {args.max_horas} h agotado tras la epoca {epoca}; "
                             "se detiene el entrenamiento.")
                break

    logging.info("Entrenamiento finalizado.")
    logging.info(f"Mejor val_loss: {mejor_val_loss:.4f}")


if __name__ == "__main__":
    main()
