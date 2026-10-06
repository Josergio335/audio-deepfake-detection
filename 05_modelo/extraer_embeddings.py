"""
extraer_embeddings.py
Extrae embeddings de los 3 Foundation Models (Wav2Vec2BERT, WavLM, Whisper)
para todos los audios del manifest de Exp-3.

Los FMs se procesan uno a la vez para minimizar uso de VRAM.
Los embeddings se guardan como tensores .pt individuales.

Uso:
    python extraer_embeddings.py                          # extrae los 3 modelos
    python extraer_embeddings.py --modelo wav2vec2bert    # extrae solo uno
    python extraer_embeddings.py --dry-run                # muestra resumen sin procesar
    python extraer_embeddings.py --workers 4              # num. workers del dataloader

Salida:
    EMBEDDINGS_DIR/wav2vec2bert/{hash}.pt
    EMBEDDINGS_DIR/wavlm/{hash}.pt
    EMBEDDINGS_DIR/whisper/{hash}.pt
    EMBEDDINGS_DIR/indice.csv   ← mapea ruta_procesada -> archivos de embedding
"""

import os
import csv
import hashlib
import argparse
import logging
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2BertModel,
    WavLMModel,
    WhisperModel,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# AJUSTAR estas rutas según el entorno de ejecución
MANIFEST      = Path.home() / "particiones/Exp3/manifest.csv"   # Cambiar por la ruta donde está el manifest
INPUT_BASE    = Path.home() / "datos/Bases de Datos procesadas"  # Cambiar por la ruta donde están los audios procesados
EMBEDDINGS_DIR = Path.home() / "embeddings"                      # Cambiar por la ruta donde se guardarán los embeddings
LOG_FILE      = Path.home() / "embeddings/extraccion_log.txt"    # Cambiar por la ruta donde se guardará el log

SAMPLE_RATE = 16000

MODELOS = {
    "wav2vec2bert": "facebook/w2v-bert-2.0",
    "wavlm":        "microsoft/wavlm-large",
    "whisper":      "openai/whisper-large-v3",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def ruta_a_hash(ruta: str) -> str:
    """Hash MD5 de la ruta — sirve como nombre de archivo de embedding."""
    return hashlib.md5(ruta.encode()).hexdigest()


def leer_manifest():
    """Lee el manifest de Exp-3 y devuelve lista de rutas procesadas."""
    # Prefijo Windows que precede a la parte relativa en las rutas del manifest
    WIN_PREFIX = "D:\\Bases de Datos extraidas\\"
    rutas = []
    with open(MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ruta_original = row["ruta"]
            # Convertir ruta Windows a ruta Linux relativa y construir ruta procesada
            rel = ruta_original.replace(WIN_PREFIX, "").replace("\\", "/")
            # El preprocesamiento convierte todos los audios a .wav
            ruta_procesada = (INPUT_BASE / rel).with_suffix(".wav")
            rutas.append((ruta_original, str(ruta_procesada)))
    return rutas


def cargar_audio(ruta_procesada: str):
    """Carga audio procesado y devuelve tensor (1, T) a 16kHz."""
    audio, sr = sf.read(ruta_procesada, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono por si acaso
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            torch.tensor(audio), sr, SAMPLE_RATE
        ).numpy()
    return torch.tensor(audio).unsqueeze(0)  # (1, T)

# ---------------------------------------------------------------------------
# Extractor por modelo
# ---------------------------------------------------------------------------

class ExtractorWav2Vec2BERT:
    def __init__(self, device):
        self.device = device
        nombre = MODELOS["wav2vec2bert"]
        logging.info(f"Cargando {nombre} ...")
        self.processor = AutoFeatureExtractor.from_pretrained(nombre)
        self.model = Wav2Vec2BertModel.from_pretrained(nombre).to(device)
        self.model.eval()

    @torch.no_grad()
    def extraer(self, audio_tensor):
        """audio_tensor: (1, T) float32. Devuelve embedding (D,)."""
        audio_np = audio_tensor.squeeze(0).numpy()
        inputs = self.processor(
            audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # Promedio temporal sobre la secuencia de hidden states
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
        return embedding


class ExtractorWavLM:
    def __init__(self, device):
        self.device = device
        nombre = MODELOS["wavlm"]
        logging.info(f"Cargando {nombre} ...")
        self.processor = AutoFeatureExtractor.from_pretrained(nombre)
        self.model = WavLMModel.from_pretrained(nombre).to(device)
        self.model.eval()

    @torch.no_grad()
    def extraer(self, audio_tensor):
        audio_np = audio_tensor.squeeze(0).numpy()
        inputs = self.processor(
            audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
        return embedding


class ExtractorWhisper:
    def __init__(self, device):
        self.device = device
        nombre = MODELOS["whisper"]
        logging.info(f"Cargando {nombre} ...")
        self.processor = AutoFeatureExtractor.from_pretrained(nombre)
        self.model = WhisperModel.from_pretrained(nombre).to(device)
        self.model.eval()

    @torch.no_grad()
    def extraer(self, audio_tensor):
        audio_np = audio_tensor.squeeze(0).numpy()
        inputs = self.processor(
            audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Whisper encoder — no usa decoder para extracción de características
        encoder_outputs = self.model.encoder(
            inputs["input_features"],
            attention_mask=inputs.get("attention_mask"),
        )
        embedding = encoder_outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
        return embedding


EXTRACTORES = {
    "wav2vec2bert": ExtractorWav2Vec2BERT,
    "wavlm":        ExtractorWavLM,
    "whisper":      ExtractorWhisper,
}

# ---------------------------------------------------------------------------
# Proceso principal por modelo
# ---------------------------------------------------------------------------

def extraer_modelo(nombre_modelo, rutas, device, dry_run=False):
    out_dir = EMBEDDINGS_DIR / nombre_modelo
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filtrar rutas ya procesadas (reanudable)
    pendientes = []
    for ruta_orig, ruta_proc in rutas:
        h = ruta_a_hash(ruta_orig)
        destino = out_dir / f"{h}.pt"
        if not destino.exists():
            pendientes.append((ruta_orig, ruta_proc, h, destino))

    logging.info(
        f"[{nombre_modelo}] Total: {len(rutas):,} | "
        f"Ya extraídos: {len(rutas)-len(pendientes):,} | "
        f"Pendientes: {len(pendientes):,}"
    )

    if dry_run or not pendientes:
        return

    extractor = EXTRACTORES[nombre_modelo](device)

    ok = errores = vacios = 0
    for ruta_orig, ruta_proc, h, destino in tqdm(pendientes, desc=nombre_modelo):
        try:
            if not Path(ruta_proc).exists():
                logging.warning(f"Archivo no encontrado: {ruta_proc}")
                errores += 1
                continue

            audio = cargar_audio(ruta_proc)

            if audio.shape[1] == 0:
                vacios += 1
                continue

            embedding = extractor.extraer(audio)
            torch.save(embedding, destino)
            ok += 1

        except Exception as e:
            logging.error(f"Error en {ruta_proc}: {e}")
            errores += 1

    logging.info(
        f"[{nombre_modelo}] Completado — "
        f"ok: {ok:,} | vacíos: {vacios:,} | errores: {errores:,}"
    )

    # Liberar memoria antes del siguiente modelo
    del extractor
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Generar índice CSV
# ---------------------------------------------------------------------------

def generar_indice(rutas):
    """Genera indice.csv mapeando ruta_original → hashes de embeddings."""
    indice_path = EMBEDDINGS_DIR / "indice.csv"
    logging.info(f"Generando índice en {indice_path} ...")

    with open(indice_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ruta_original", "hash", "wav2vec2bert_pt", "wavlm_pt", "whisper_pt"])
        for ruta_orig, _ in rutas:
            h = ruta_a_hash(ruta_orig)
            writer.writerow([
                ruta_orig,
                h,
                str(EMBEDDINGS_DIR / "wav2vec2bert" / f"{h}.pt"),
                str(EMBEDDINGS_DIR / "wavlm"        / f"{h}.pt"),
                str(EMBEDDINGS_DIR / "whisper"      / f"{h}.pt"),
            ])

    logging.info(f"Índice generado: {len(rutas):,} entradas")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extrae embeddings de Foundation Models para el corpus de Exp-3"
    )
    parser.add_argument(
        "--modelo", choices=list(MODELOS.keys()),
        help="Extraer solo este modelo (por defecto extrae los 3)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Muestra resumen sin procesar"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo (cuda / cpu)"
    )
    args = parser.parse_args()

    setup_logging()
    logging.info("=" * 60)
    logging.info("EXTRACCIÓN DE EMBEDDINGS")
    logging.info(f"Manifest : {MANIFEST}")
    logging.info(f"Salida   : {EMBEDDINGS_DIR}")
    logging.info(f"Dispositivo: {args.device}")
    if args.dry_run:
        logging.info("*** DRY-RUN: no se escribirán archivos ***")
    logging.info("=" * 60)

    rutas = leer_manifest()
    logging.info(f"Archivos en manifest: {len(rutas):,}")

    modelos_a_extraer = [args.modelo] if args.modelo else list(MODELOS.keys())

    for nombre in modelos_a_extraer:
        extraer_modelo(nombre, rutas, args.device, args.dry_run)

    if not args.dry_run:
        generar_indice(rutas)

    logging.info("Extracción completada.")


if __name__ == "__main__":
    main()
