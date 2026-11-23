"""
extraer_embeddings_test.py
Extrae embeddings de los 3 Foundation Models para los audios de test.

Uso:
    python extraer_embeddings_test.py --manifest ~/particiones/manifest_test_indistribucion.csv
    python extraer_embeddings_test.py --manifest ~/particiones/manifest_test_zeroshot.csv
    python extraer_embeddings_test.py --manifest ~/particiones/manifest_test_fueradominio.csv
    python extraer_embeddings_test.py --manifest ... --modelo wav2vec2bert
    python extraer_embeddings_test.py --manifest ... --dry-run

Salida:
    EMBEDDINGS_TEST_DIR/wav2vec2bert/{hash}.pt
    EMBEDDINGS_TEST_DIR/wavlm/{hash}.pt
    EMBEDDINGS_TEST_DIR/whisper/{hash}.pt
    EMBEDDINGS_TEST_DIR/indice_test_{escenario}.csv
"""

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

EMBEDDINGS_TEST_DIR = Path.home() / "embeddings_test"

# Mapeo de prefijos Windows → rutas Linux procesadas
WIN_PREFIXES = [
    # (prefijo_windows, base_linux, cambiar_sufijo_a_wav)
    # Archivos ya procesados (MP3 comprimidos) — mantener extensión original
    ("D:\\Bases de Datos procesadas\\",  Path.home() / "datos/Bases de Datos procesadas", False),
    # Archivos extraídos → convertidos a WAV en el servidor
    ("D:\\Bases de Datos extraidas\\",   Path.home() / "datos/Bases de Datos procesadas", True),
]

SAMPLE_RATE = 16000

MODELOS = {
    "wav2vec2bert": "facebook/w2v-bert-2.0",
    "wavlm":        "microsoft/wavlm-large",
    "whisper":      "openai/whisper-large-v3",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path):
    EMBEDDINGS_TEST_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def ruta_a_hash(ruta: str) -> str:
    return hashlib.md5(ruta.encode()).hexdigest()


def leer_manifest(manifest: Path):
    """
    Lee el manifest de test. Devuelve lista de tuplas:
    (ruta_original, ruta_procesada_linux, label, db, idioma, escenario)
    """
    filas = []
    with open(manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ruta_original = row["ruta"]
            escenario = row.get("escenario") or Path(str(manifest)).stem.replace("manifest_test_", "")
            # Ruta Linux absoluta (manifest generado en servidor)
            if ruta_original.startswith("/"):
                filas.append((
                    ruta_original, ruta_original,
                    row["label"], row["db"], row["idioma"], escenario,
                ))
                continue
            # Convertir ruta Windows → ruta Linux según prefijo
            ruta_procesada = None
            for win_pref, linux_base, a_wav in WIN_PREFIXES:
                if ruta_original.startswith(win_pref):
                    rel = ruta_original[len(win_pref):].replace("\\", "/")
                    linux_path = linux_base / rel
                    if a_wav:
                        linux_path = linux_path.with_suffix(".wav")
                    ruta_procesada = linux_path
                    break
            if ruta_procesada is None:
                rel = Path(ruta_original).name
                ruta_procesada = (Path.home() / "datos/Bases de Datos procesadas" / rel).with_suffix(".wav")
            filas.append((
                ruta_original,
                str(ruta_procesada),
                row["label"],
                row["db"],
                row["idioma"],
                escenario,
            ))
    return filas


def cargar_audio(ruta_procesada: str):
    audio, sr = sf.read(ruta_procesada, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            torch.tensor(audio), sr, SAMPLE_RATE
        ).numpy()
    return torch.tensor(audio).unsqueeze(0)  # (1, T)

# ---------------------------------------------------------------------------
# Extractores
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
        audio_np = audio_tensor.squeeze(0).numpy()
        inputs = self.processor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()


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
        inputs = self.processor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()


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
        encoder_outputs = self.model.encoder(
            inputs["input_features"],
            attention_mask=inputs.get("attention_mask"),
        )
        return encoder_outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()


EXTRACTORES = {
    "wav2vec2bert": ExtractorWav2Vec2BERT,
    "wavlm":        ExtractorWavLM,
    "whisper":      ExtractorWhisper,
}

# ---------------------------------------------------------------------------
# Proceso principal por modelo
# ---------------------------------------------------------------------------

def extraer_modelo(nombre_modelo, filas, device, dry_run=False):
    out_dir = EMBEDDINGS_TEST_DIR / nombre_modelo
    out_dir.mkdir(parents=True, exist_ok=True)

    pendientes = []
    for fila in filas:
        ruta_orig = fila[0]
        h = ruta_a_hash(ruta_orig)
        destino = out_dir / f"{h}.pt"
        if not destino.exists():
            pendientes.append((*fila, h, destino))

    logging.info(
        f"[{nombre_modelo}] Total: {len(filas):,} | "
        f"Ya extraídos: {len(filas)-len(pendientes):,} | "
        f"Pendientes: {len(pendientes):,}"
    )

    if dry_run or not pendientes:
        return

    extractor = EXTRACTORES[nombre_modelo](device)

    ok = errores = vacios = 0
    for ruta_orig, ruta_proc, label, db, idioma, escenario, h, destino in tqdm(pendientes, desc=nombre_modelo):
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

    del extractor
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Generar índice CSV
# ---------------------------------------------------------------------------

def generar_indice(filas, escenario: str):
    indice_path = EMBEDDINGS_TEST_DIR / f"indice_test_{escenario}.csv"
    logging.info(f"Generando índice en {indice_path} ...")

    with open(indice_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ruta_original", "hash",
            "wav2vec2bert_pt", "wavlm_pt", "whisper_pt",
            "label", "db", "idioma", "escenario",
        ])
        for ruta_orig, _, label, db, idioma, escenario_fila in filas:
            h = ruta_a_hash(ruta_orig)
            writer.writerow([
                ruta_orig, h,
                str(EMBEDDINGS_TEST_DIR / "wav2vec2bert" / f"{h}.pt"),
                str(EMBEDDINGS_TEST_DIR / "wavlm"        / f"{h}.pt"),
                str(EMBEDDINGS_TEST_DIR / "whisper"      / f"{h}.pt"),
                label, db, idioma, escenario_fila,
            ])

    logging.info(f"Índice generado: {len(filas):,} entradas")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extrae embeddings de Foundation Models para los conjuntos de test"
    )
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Ruta al manifest de test (manifest_test_*.csv)")
    parser.add_argument("--modelo", choices=list(MODELOS.keys()),
                        help="Extraer solo este modelo (por defecto los 3)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    escenario = args.manifest.stem.replace("manifest_test_", "")
    log_file = EMBEDDINGS_TEST_DIR / f"extraccion_test_{escenario}.log"

    setup_logging(log_file)
    logging.info("=" * 60)
    logging.info("EXTRACCIÓN DE EMBEDDINGS — TEST")
    logging.info(f"Manifest   : {args.manifest}")
    logging.info(f"Escenario  : {escenario}")
    logging.info(f"Input bases: {[str(b) for _, b, _ in WIN_PREFIXES]}")
    logging.info(f"Salida     : {EMBEDDINGS_TEST_DIR}")
    logging.info(f"Dispositivo: {args.device}")
    if args.dry_run:
        logging.info("*** DRY-RUN ***")
    logging.info("=" * 60)

    filas = leer_manifest(args.manifest)
    logging.info(f"Archivos en manifest: {len(filas):,}")

    modelos_a_extraer = [args.modelo] if args.modelo else list(MODELOS.keys())

    for nombre in modelos_a_extraer:
        extraer_modelo(nombre, filas, args.device, args.dry_run)

    if not args.dry_run:
        generar_indice(filas, escenario)

    logging.info("Extracción completada.")


if __name__ == "__main__":
    main()
