"""
train.py
Entrenamiento del modelo de deteccion de audio deepfake.

Los Foundation Models (Wav2Vec2BERT, WavLM, Whisper) estan CONGELADOS — sus embeddings
ya fueron extraidos a disco por extraer_embeddings.py. Este script solo entrena los
modulos livianos encima: RASA (alineacion de estilo), SERM (prototipos jerarquicos)
y el Gating Network (fusion adaptativa de los 3 streams).

Flujo de datos:
    indice.csv + manifest.csv  ->  EmbeddingDataset  ->  DataLoader
                                          |
                              [emb1, emb2, emb3] + etiqueta
                                          |
                         RASA_1  RASA_2  RASA_3    <- uno por stream
                             |       |      |
                         head_1  head_2  head_3    <- clasificadores lineales
                             |       |      |
                         score_1 score_2 score_3
                              \      |     /
                             GatingNetwork
                                    |
                               score_final
                                    |
                               L_BCE + L_RASA + L_SERM

Salida:
    D:\Entrenamientos\{exp}_{timestamp}/
        checkpoint_mejor.pt      <- mejor modelo por val_loss
        checkpoint_ultimo.pt     <- ultimo estado (para reanudacion)
        metricas.csv             <- perdida/accuracy por epoca
        config.json              <- hiperparametros del experimento

Uso:
    python train.py --exp 3                         # entrenar Exp-3
    python train.py --exp 1 --epochs 50 --lr 3e-4   # con hiperparametros
    python train.py --exp 3 --device cpu            # forzar CPU
    python train.py --exp 3 --reanudar              # continuar desde checkpoint_ultimo.pt

R1 IOV: "1 red neuronal clasificadora entrenada que logre convergencia estable en la
         funcion de perdida" — la convergencia se evidencia en metricas.csv y en que
         val_loss decrece monotonamente durante el entrenamiento.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Importar modulos del modelo
sys.path.insert(0, str(Path(__file__).parent))
from models.rasa   import RASA
from models.serm   import SERM
from models.gating import GatingNetwork

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

# Directorio raiz de los manifests por experimento
# AJUSTAR estas rutas según el entorno de ejecución
PARTICIONES = {
    1: Path.home() / "particiones/Exp1",   # Cambiar por la ruta donde está el manifest de Exp1
    2: Path.home() / "particiones/Exp2",   # Cambiar por la ruta donde está el manifest de Exp2
    3: Path.home() / "particiones/Exp3",   # Cambiar por la ruta donde está el manifest de Exp3
}

EMBEDDINGS_DIR = Path.home() / "embeddings"       # Cambiar por la ruta donde están los embeddings
SALIDA_BASE    = Path.home() / "entrenamientos"   # Cambiar por la ruta donde se guardarán los checkpoints

# Dimensiones de cada FM (fijas — son los modelos usados en extraer_embeddings.py)
DIMS = {
    "wav2vec2bert": 1024,
    "wavlm":        1024,
    "whisper":      1280,
}

# Particion train/val: los manifests no traen columna `split`, asi que se asigna
# de forma determinista a partir de un hash de la ruta del audio.
SEMILLA_SPLIT = 42
VAL_PCT       = 20   # porcentaje de archivos que va a validacion

ETIQUETAS = {"bonafide": 0, "spoof": 1}


def asignar_split(ruta: str, val_pct: int = VAL_PCT, semilla: int = SEMILLA_SPLIT) -> str:
    """
    Devuelve 'train' o 'val' para un archivo, segun el hash de (semilla, ruta).

    Al depender solo de la ruta, un archivo tiene el mismo split en Exp-1, Exp-2 y
    Exp-3 (los experimentos son anidados), y el resultado es reproducible sin guardar
    nada en disco.
    """
    h = int(hashlib.md5(f"{semilla}:{ruta}".encode("utf-8")).hexdigest(), 16) % 100
    return "val" if h < val_pct else "train"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """
    Dataset que carga embeddings precomputados (.pt) desde disco.

    Cada muestra es un triplete (emb_w2v, emb_wavlm, emb_whisper) + etiqueta (0/1).
    Los archivos .pt se cargan bajo demanda (lazy loading) para no saturar la RAM.

    El dataset se construye a partir de dos fuentes:
    - indice.csv:   mapea ruta_original -> rutas de los 3 .pt (generado por extraer_embeddings.py)
    - manifest.csv: contiene la etiqueta (`label` bonafide/spoof, o `etiqueta` 0/1),
      idioma y db. Si trae columna `split` se respeta; si no, train/val se asignan con
      asignar_split() (hash de la ruta). El split 'test' no se genera aqui: requiere
      una columna `split` explicita.

    Solo se incluyen muestras para las que los 3 .pt existen en disco.
    """

    def __init__(self, particion_dir: Path, split: str, val_pct: int = VAL_PCT):
        """
        Args:
            particion_dir: directorio con manifest.csv (ej. D:\Particiones\Exp3)
            split:         'train', 'val' o 'test'
            val_pct:       % de archivos para validacion cuando el manifest no trae `split`
        """
        self.muestras = []  # lista de (ruta_w2v, ruta_wavlm, ruta_whisper, etiqueta)

        # Leer manifest para obtener etiquetas y splits
        manifest_path = particion_dir / "manifest.csv"
        manifest = {}
        with open(manifest_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row_split = row.get("split") or asignar_split(row["ruta"], val_pct)
                if row_split != split:
                    continue
                if row.get("etiqueta"):
                    manifest[row["ruta"]] = int(row["etiqueta"])  # 0=bonafide, 1=spoof
                else:
                    manifest[row["ruta"]] = ETIQUETAS[row["label"]]

        # Leer indice para obtener rutas de los embeddings
        indice_path = EMBEDDINGS_DIR / "indice.csv"
        encontrados = ausentes = 0

        with open(indice_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ruta_orig = row["ruta_original"]
                if ruta_orig not in manifest:
                    continue  # no pertenece a este split/experimento

                pt_w2v    = Path(row["wav2vec2bert_pt"])
                pt_wavlm  = Path(row["wavlm_pt"])
                pt_whisper = Path(row["whisper_pt"])

                # Solo incluir si los 3 embeddings existen en disco
                if pt_w2v.exists() and pt_wavlm.exists() and pt_whisper.exists():
                    etiqueta = manifest[ruta_orig]
                    self.muestras.append((pt_w2v, pt_wavlm, pt_whisper, etiqueta))
                    encontrados += 1
                else:
                    ausentes += 1

        logging.info(
            f"[Dataset {split}] {encontrados:,} muestras cargadas | {ausentes:,} sin embeddings"
        )

        if encontrados == 0:
            raise RuntimeError(
                f"No se encontraron muestras para split={split} en {particion_dir}. "
                "Verifica que extraer_embeddings.py haya terminado y que indice.csv este actualizado."
            )

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx):
        pt_w2v, pt_wavlm, pt_whisper, etiqueta = self.muestras[idx]

        # Cargar tensores desde disco (cada uno tiene shape (D,))
        emb_w2v    = torch.load(pt_w2v,    map_location="cpu", weights_only=True)
        emb_wavlm  = torch.load(pt_wavlm,  map_location="cpu", weights_only=True)
        emb_whisper = torch.load(pt_whisper, map_location="cpu", weights_only=True)

        # Asegurar float32 (algunos modelos pueden guardar en float16)
        emb_w2v     = emb_w2v.float()
        emb_wavlm   = emb_wavlm.float()
        emb_whisper = emb_whisper.float()

        etiqueta_tensor = torch.tensor(etiqueta, dtype=torch.float32)

        return emb_w2v, emb_wavlm, emb_whisper, etiqueta_tensor


# ---------------------------------------------------------------------------
# Modelo completo
# ---------------------------------------------------------------------------

class ModeloDeepfake(nn.Module):
    """
    Modelo completo de deteccion de audio deepfake.

    Toma los 3 embeddings precomputados, los procesa con RASA y un clasificador
    lineal por stream, y los fusiona con el Gating Network.
    SERM se usa solo durante entrenamiento (como perdida adicional).
    """

    def __init__(self, dims=DIMS):
        super().__init__()
        d_w2v   = dims["wav2vec2bert"]
        d_wavlm = dims["wavlm"]
        d_whisp = dims["whisper"]

        # RASA: un modulo por stream (cada uno aprende su propio style bank)
        self.rasa_w2v   = RASA(d_w2v,   M=160)
        self.rasa_wavlm = RASA(d_wavlm, M=160)
        self.rasa_whisp = RASA(d_whisp, M=160)

        # Clasificadores lineales: embedding alineado -> score escalar por stream
        # Un solo numero por stream: logit de bonafide/spoof
        self.head_w2v   = nn.Linear(d_w2v,   1)
        self.head_wavlm = nn.Linear(d_wavlm, 1)
        self.head_whisp = nn.Linear(d_whisp, 1)

        # Gating Network: fusiona los 3 streams de forma adaptativa
        self.gating = GatingNetwork(dims_entrada=[d_w2v, d_wavlm, d_whisp], hidden_dim=256)

        # SERM: recibe el embedding concatenado de los 3 streams
        d_concat = d_w2v + d_wavlm + d_whisp
        self.serm = SERM(embed_dim=d_concat, dim_poincare=128)

    def forward(self, emb_w2v, emb_wavlm, emb_whisper):
        """
        Args:
            emb_w2v:    (B, 1024)
            emb_wavlm:  (B, 1024)
            emb_whisper: (B, 1280)
        Returns:
            score_final: (B,) logit final (sin sigmoid — usar BCEWithLogitsLoss)
            emb_list:    lista de embeddings alineados por RASA (para SERM)
        """
        # Alinear estilo de cada stream con RASA
        e1 = self.rasa_w2v(emb_w2v)       # (B, 1024)
        e2 = self.rasa_wavlm(emb_wavlm)   # (B, 1024)
        e3 = self.rasa_whisp(emb_whisper) # (B, 1280)

        # Score escalar de cada stream (logit)
        s1 = self.head_w2v(e1).squeeze(-1)    # (B,)
        s2 = self.head_wavlm(e2).squeeze(-1)  # (B,)
        s3 = self.head_whisp(e3).squeeze(-1)  # (B,)

        # Fusion adaptativa con Gating Network
        score_final = self.gating.fusionar(
            scores=[s1, s2, s3],
            embeddings=[e1, e2, e3],
        )  # (B,)

        return score_final, [e1, e2, e3]

    def perdidas_auxiliares(self, emb_list, etiquetas):
        """
        Calcula las perdidas auxiliares de RASA y SERM.

        Args:
            emb_list:  lista de embeddings RASA [(B, D1), (B, D2), (B, D3)]
            etiquetas: (B,) etiquetas binarias {0, 1}
        Returns:
            dict con cada perdida individual
        """
        e1, e2, e3 = emb_list

        # Perdidas RASA: orth + reco para cada stream
        l_orth = (self.rasa_w2v.orthogonal_loss()
                  + self.rasa_wavlm.orthogonal_loss()
                  + self.rasa_whisp.orthogonal_loss()) / 3

        l_reco = (self.rasa_w2v.reconstruction_loss(e1)
                  + self.rasa_wavlm.reconstruction_loss(e2)
                  + self.rasa_whisp.reconstruction_loss(e3)) / 3

        # Perdida SERM: usa la concatenacion de los 3 embeddings alineados
        e_concat = torch.cat([e1, e2, e3], dim=-1)  # (B, D1+D2+D3)
        etiquetas_int = etiquetas.long()
        l_serm = self.serm.perdida_total(e_concat, etiquetas_int)

        return {"l_orth": l_orth, "l_reco": l_reco, "l_serm": l_serm}


# ---------------------------------------------------------------------------
# Loop de entrenamiento
# ---------------------------------------------------------------------------

def entrenar_epoch(modelo, loader, optimizer, criterio, device, lambdas):
    """
    Un epoch de entrenamiento. Retorna el promedio de cada perdida.
    """
    modelo.train()
    tot_bce = tot_orth = tot_reco = tot_serm = tot_total = 0.0
    n = 0

    for emb_w2v, emb_wavlm, emb_whisper, etiquetas in tqdm(loader, desc="train", leave=False):
        emb_w2v    = emb_w2v.to(device)
        emb_wavlm  = emb_wavlm.to(device)
        emb_whisper = emb_whisper.to(device)
        etiquetas  = etiquetas.to(device)

        optimizer.zero_grad()

        score, emb_list = modelo(emb_w2v, emb_wavlm, emb_whisper)
        l_bce  = criterio(score, etiquetas)
        aux    = modelo.perdidas_auxiliares(emb_list, etiquetas)

        # Perdida total: BCE + RASA_orth + RASA_reco + SERM
        total = (l_bce
                 + lambdas["orth"] * aux["l_orth"]
                 + lambdas["reco"] * aux["l_reco"]
                 + lambdas["serm"] * aux["l_serm"])

        total.backward()
        # Gradient clipping: evita explosiones de gradiente, comun en geometria hiperbolica
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        optimizer.step()

        b = etiquetas.size(0)
        tot_bce   += l_bce.item()   * b
        tot_orth  += aux["l_orth"].item() * b
        tot_reco  += aux["l_reco"].item() * b
        tot_serm  += aux["l_serm"].item() * b
        tot_total += total.item()   * b
        n += b

    return {
        "bce":   tot_bce   / n,
        "orth":  tot_orth  / n,
        "reco":  tot_reco  / n,
        "serm":  tot_serm  / n,
        "total": tot_total / n,
    }


@torch.no_grad()
def evaluar(modelo, loader, criterio, device, lambdas):
    """
    Evaluacion sin gradientes. Retorna perdidas y accuracy.
    """
    modelo.eval()
    tot_bce = tot_total = correctos = n = 0

    for emb_w2v, emb_wavlm, emb_whisper, etiquetas in tqdm(loader, desc="val", leave=False):
        emb_w2v    = emb_w2v.to(device)
        emb_wavlm  = emb_wavlm.to(device)
        emb_whisper = emb_whisper.to(device)
        etiquetas  = etiquetas.to(device)

        score, emb_list = modelo(emb_w2v, emb_wavlm, emb_whisper)
        l_bce = criterio(score, etiquetas)
        aux   = modelo.perdidas_auxiliares(emb_list, etiquetas)

        total = (l_bce
                 + lambdas["orth"] * aux["l_orth"]
                 + lambdas["reco"] * aux["l_reco"]
                 + lambdas["serm"] * aux["l_serm"])

        predicciones = (torch.sigmoid(score) > 0.5).float()
        correctos += (predicciones == etiquetas).sum().item()

        b = etiquetas.size(0)
        tot_bce   += l_bce.item() * b
        tot_total += total.item() * b
        n += b

    return {
        "bce":      tot_bce   / n,
        "total":    tot_total / n,
        "accuracy": correctos / n,
    }


# ---------------------------------------------------------------------------
# Guardar / cargar checkpoints
# ---------------------------------------------------------------------------

def guardar_checkpoint(ruta, modelo, optimizer, scheduler, epoch, mejor_val_loss, config):
    torch.save({
        "epoch":          epoch,
        "model_state":    modelo.state_dict(),
        "optim_state":    optimizer.state_dict(),
        "sched_state":    scheduler.state_dict() if scheduler else None,
        "mejor_val_loss": mejor_val_loss,
        "config":         config,
    }, ruta)


def cargar_checkpoint(ruta, modelo, optimizer, scheduler):
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    modelo.load_state_dict(ck["model_state"])
    optimizer.load_state_dict(ck["optim_state"])
    if scheduler and ck["sched_state"]:
        scheduler.load_state_dict(ck["sched_state"])
    return ck["epoch"], ck["mejor_val_loss"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de deteccion de audio deepfake"
    )
    parser.add_argument("--exp",      type=int, default=3, choices=[1, 2, 3],
                        help="Experimento a entrenar (1, 2 o 3)")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--bs",       type=int, default=256, dest="batch_size",
                        help="Batch size (256 es viable con embeddings en RAM)")
    parser.add_argument("--lr",       type=float, default=1e-3,
                        help="Learning rate inicial")
    parser.add_argument("--workers",  type=int, default=4,
                        help="Numero de workers para DataLoader")
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reanudar", action="store_true",
                        help="Continuar desde el ultimo checkpoint guardado")
    parser.add_argument("--val-pct",  type=int, default=VAL_PCT, dest="val_pct",
                        help="Porcentaje de archivos para validacion (el resto es train)")

    # Pesos de las perdidas auxiliares
    parser.add_argument("--lambda-orth", type=float, default=0.01)
    parser.add_argument("--lambda-reco", type=float, default=0.01)
    parser.add_argument("--lambda-serm", type=float, default=0.05)

    args = parser.parse_args()

    # Directorio de salida con timestamp para no sobrescribir experimentos previos
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SALIDA_BASE / f"exp{args.exp}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging a archivo y consola
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "entrenamiento.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.info("=" * 60)
    logging.info(f"ENTRENAMIENTO Exp-{args.exp}")
    logging.info(f"Salida: {out_dir}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Epochs: {args.epochs} | BS: {args.batch_size} | LR: {args.lr}")
    logging.info("=" * 60)

    config = vars(args)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    lambdas = {
        "orth": args.lambda_orth,
        "reco": args.lambda_reco,
        "serm": args.lambda_serm,
    }

    # Datasets y DataLoaders
    particion_dir = PARTICIONES[args.exp]
    logging.info(f"Cargando datasets desde {particion_dir} ...")

    ds_train = EmbeddingDataset(particion_dir, "train", args.val_pct)
    ds_val   = EmbeddingDataset(particion_dir, "val",   args.val_pct)

    # pin_memory=True acelera la transferencia CPU->GPU
    pin = args.device == "cuda"
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=pin)

    logging.info(f"Train: {len(ds_train):,} muestras | Val: {len(ds_val):,} muestras")

    # Modelo
    modelo = ModeloDeepfake().to(args.device)
    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Parametros entrenables: {n_params:,}")

    # Optimizador y scheduler
    optimizer = optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    # ReduceLROnPlateau: reduce LR si val_loss no mejora en 10 epocas
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, verbose=True
    )

    # Funcion de perdida principal: BCE con logits (numericamente mas estable que BCE + sigmoid)
    criterio = nn.BCEWithLogitsLoss()

    # CSV para registrar metricas por epoca (evidencia de convergencia para R1 IOV)
    metricas_path = out_dir / "metricas.csv"
    with open(metricas_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_bce", "train_orth", "train_reco", "train_serm", "train_total",
            "val_bce", "val_total", "val_accuracy", "lr"
        ])

    epoch_inicio = 1
    mejor_val_loss = float("inf")
    ck_ultimo = out_dir / "checkpoint_ultimo.pt"
    ck_mejor  = out_dir / "checkpoint_mejor.pt"

    # Reanudacion desde checkpoint anterior
    if args.reanudar and ck_ultimo.exists():
        logging.info(f"Reanudando desde {ck_ultimo} ...")
        epoch_inicio, mejor_val_loss = cargar_checkpoint(ck_ultimo, modelo, optimizer, scheduler)
        epoch_inicio += 1
        logging.info(f"Continuando desde epoch {epoch_inicio} | mejor val_loss: {mejor_val_loss:.4f}")

    # Loop principal
    for epoch in range(epoch_inicio, args.epochs + 1):
        logging.info(f"--- Epoch {epoch}/{args.epochs} ---")

        train_m = entrenar_epoch(modelo, dl_train, optimizer, criterio, args.device, lambdas)
        val_m   = evaluar(modelo, dl_val, criterio, args.device, lambdas)

        lr_actual = optimizer.param_groups[0]["lr"]
        scheduler.step(val_m["total"])

        logging.info(
            f"Train: bce={train_m['bce']:.4f} orth={train_m['orth']:.4f} "
            f"reco={train_m['reco']:.4f} serm={train_m['serm']:.4f} total={train_m['total']:.4f}"
        )
        logging.info(
            f"Val:   bce={val_m['bce']:.4f} total={val_m['total']:.4f} "
            f"accuracy={val_m['accuracy']:.4f} | LR={lr_actual:.2e}"
        )

        # Registrar en CSV
        with open(metricas_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_m['bce']:.6f}", f"{train_m['orth']:.6f}",
                f"{train_m['reco']:.6f}", f"{train_m['serm']:.6f}", f"{train_m['total']:.6f}",
                f"{val_m['bce']:.6f}", f"{val_m['total']:.6f}",
                f"{val_m['accuracy']:.6f}", f"{lr_actual:.2e}",
            ])

        # Guardar checkpoint ultimo (siempre) y mejor (si mejoro val_loss)
        guardar_checkpoint(ck_ultimo, modelo, optimizer, scheduler, epoch, mejor_val_loss, config)

        if val_m["total"] < mejor_val_loss:
            mejor_val_loss = val_m["total"]
            guardar_checkpoint(ck_mejor, modelo, optimizer, scheduler, epoch, mejor_val_loss, config)
            logging.info(f"  -> Nuevo mejor modelo guardado (val_loss={mejor_val_loss:.4f})")

    logging.info("Entrenamiento completado.")
    logging.info(f"Mejor val_loss: {mejor_val_loss:.4f}")
    logging.info(f"Checkpoints en: {out_dir}")
    logging.info(f"Metricas en:    {metricas_path}")
    logging.info("")
    logging.info("Siguiente paso: evaluate.py --checkpoint checkpoint_mejor.pt --split test")


if __name__ == "__main__":
    main()
