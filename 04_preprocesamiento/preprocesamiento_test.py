"""
preprocesamiento_test.py
Preprocesa los audios de los conjuntos de prueba (test).

Igual que preprocesamiento_dataset.py pero apunta a:
    - INPUT_BASE  : D:/Bases de Datos extraidas   (audios originales sin procesar)
    - OUTPUT_BASE : D:/Archivos Test Procesados    (carpeta separada del train)
    - MANIFEST    : uno de los 3 manifests de test (pasado por argumento)

Operaciones por archivo:
    1. Carga el audio (WAV o FLAC)
    2. Resamplea a 16 kHz mono
    3. Aplica WebRTC VAD para eliminar frames de silencio
    4. Guarda como WAV 16-bit replicando la estructura del original

Caracteristicas:
    - Reanudable: salta archivos que ya existen en destino
    - Multiproceso: N workers en paralelo (default: 4)
    - Log de exito y errores junto al manifest de test

Uso:
    python preprocesamiento_test.py --manifest D:/Particiones/manifest_test_indistribucion.csv
    python preprocesamiento_test.py --manifest D:/Particiones/manifest_test_zeroshot.csv
    python preprocesamiento_test.py --manifest D:/Particiones/manifest_test_fueradominio.csv
    python preprocesamiento_test.py --manifest ... --dry-run
    python preprocesamiento_test.py --manifest ... --workers 8

En el servidor:
    python preprocesamiento_test.py --manifest ~/particiones/manifest_test_indistribucion.csv
"""

import csv
import argparse
import logging
import multiprocessing
import struct
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import webrtcvad

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# AJUSTAR según el entorno de ejecución
INPUT_BASE  = Path(r"D:\Bases de Datos extraidas")    # Cambiar por la ruta donde están los audios originales
OUTPUT_BASE = Path(r"D:\Archivos Test Procesados")    # Cambiar por la ruta donde se guardarán los audios de test

SAMPLE_RATE    = 16000
VAD_MODE       = 2
FRAME_MS       = 30
MIN_SPEECH_SEC = 0.5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path):
    logging.basicConfig(
        filename=str(log_file),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

def aplicar_vad(audio_int16: np.ndarray, sr: int, mode: int) -> np.ndarray:
    vad = webrtcvad.Vad(mode)
    frame_samples = int(sr * FRAME_MS / 1000)
    frames_con_voz = []

    for inicio in range(0, len(audio_int16) - frame_samples + 1, frame_samples):
        frame = audio_int16[inicio: inicio + frame_samples]
        raw = struct.pack(f"{len(frame)}h", *frame)
        if vad.is_speech(raw, sr):
            frames_con_voz.append(frame)

    if not frames_con_voz:
        return np.array([], dtype=np.int16)

    return np.concatenate(frames_con_voz)

# ---------------------------------------------------------------------------
# Procesamiento de un archivo
# ---------------------------------------------------------------------------

def procesar_archivo(args):
    ruta_str, vad_mode = args
    ruta = Path(ruta_str)

    try:
        rel = ruta.relative_to(INPUT_BASE)
    except ValueError:
        rel = Path(ruta.name)

    destino = OUTPUT_BASE / rel.with_suffix(".wav")

    if destino.exists():
        return (ruta_str, "skip")

    try:
        audio_float, _ = librosa.load(ruta, sr=SAMPLE_RATE, mono=True)
        audio_int16 = (audio_float * 32767).clip(-32768, 32767).astype(np.int16)
        audio_vad = aplicar_vad(audio_int16, SAMPLE_RATE, vad_mode)

        if len(audio_vad) < SAMPLE_RATE * MIN_SPEECH_SEC:
            return (ruta_str, f"vacio (menos de {MIN_SPEECH_SEC}s de voz tras VAD)")

        destino.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destino), audio_vad, SAMPLE_RATE, subtype="PCM_16")
        return (ruta_str, "ok")

    except Exception as e:
        return (ruta_str, f"error: {e}")

# ---------------------------------------------------------------------------
# Cargar manifest
# ---------------------------------------------------------------------------

def cargar_pendientes(manifest: Path, vad_mode: int):
    pendientes = []
    ya_hechos = 0

    with open(manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ruta = Path(row["ruta"])
            try:
                rel = ruta.relative_to(INPUT_BASE)
            except ValueError:
                rel = Path(ruta.name)
            destino = OUTPUT_BASE / rel.with_suffix(".wav")

            if destino.exists():
                ya_hechos += 1
            else:
                pendientes.append((str(ruta), vad_mode))

    return pendientes, ya_hechos

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocesa audios de los conjuntos de prueba: resample 16kHz + WebRTC VAD"
    )
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Ruta al manifest de test (manifest_test_*.csv)")
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument("--vad-mode", type=int, default=VAD_MODE, choices=[0,1,2,3])
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    log_file = args.manifest.parent / (args.manifest.stem + "_preprocesamiento_log.txt")
    setup_logging(log_file)

    print("=" * 62)
    print(f"  PREPROCESAMIENTO TEST — {args.manifest.name}")
    print("=" * 62)
    print(f"  Manifest  : {args.manifest}")
    print(f"  Input     : {INPUT_BASE}")
    print(f"  Destino   : {OUTPUT_BASE}")
    print(f"  Workers   : {args.workers}")
    print(f"  VAD mode  : {args.vad_mode}")
    print()

    pendientes, ya_hechos = cargar_pendientes(args.manifest, args.vad_mode)
    total = len(pendientes) + ya_hechos

    print(f"  Total en manifest : {total:,}")
    print(f"  Ya procesados     : {ya_hechos:,}")
    print(f"  Pendientes        : {len(pendientes):,}")

    if args.dry_run or not pendientes:
        if not pendientes:
            print("\nTodos los archivos ya están procesados.")
        return

    print(f"\nIniciando procesamiento con {args.workers} workers ...")
    print("  (Ctrl+C para interrumpir — es reanudable)\n")

    t_inicio = time.time()
    contadores = {"ok": 0, "skip": 0, "vacio": 0, "error": 0}

    with multiprocessing.Pool(processes=args.workers) as pool:
        for i, (ruta, estado) in enumerate(
            pool.imap_unordered(procesar_archivo, pendientes, chunksize=8), start=1
        ):
            if estado == "ok":
                contadores["ok"] += 1
            elif estado == "skip":
                contadores["skip"] += 1
            elif estado.startswith("vacio"):
                contadores["vacio"] += 1
                logging.warning(f"VACIO {ruta} — {estado}")
            else:
                contadores["error"] += 1
                logging.error(f"ERROR {ruta} — {estado}")

            if i % 500 == 0 or i == len(pendientes):
                elapsed = time.time() - t_inicio
                velocidad = i / elapsed if elapsed > 0 else 0
                eta_h = (len(pendientes) - i) / velocidad / 3600 if velocidad > 0 else 0
                print(
                    f"  [{i/len(pendientes)*100:5.1f}%] {i:,}/{len(pendientes):,} "
                    f"| ok={contadores['ok']:,} err={contadores['error']:,} "
                    f"| {velocidad:.1f} arch/s | ETA: {eta_h:.1f}h"
                )

    elapsed_total = time.time() - t_inicio
    resumen = (
        f"\nCompletado en {elapsed_total/3600:.2f}h\n"
        f"  ok      : {contadores['ok']:,}\n"
        f"  vacios  : {contadores['vacio']:,}\n"
        f"  errores : {contadores['error']:,}\n"
    )
    print(resumen)
    logging.info(resumen)


if __name__ == "__main__":
    main()
