"""
Preprocesa los clips de Common Voice listados en manifest_test_zeroshot.csv.

Extrae solo las filas con db=CommonVoice, aplica el mismo pipeline que el
preprocesamiento principal (librosa 16kHz mono + WebRTC VAD + guardar WAV 16-bit)
y guarda en D:/Bases de Datos procesadas/CommonVoice/{lang}/clips/*.wav

Reanudable: salta archivos ya existentes en destino.

Uso:
    python preprocesamiento_commonvoice.py               # 4 workers
    python preprocesamiento_commonvoice.py --workers 8
    python preprocesamiento_commonvoice.py --dry-run
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
MANIFEST    = Path("D:/Particiones/manifest_test_zeroshot.csv")
INPUT_BASE  = Path("D:/Bases de Datos extraidas")
OUTPUT_BASE = Path("D:/Bases de Datos procesadas")
LOG_FILE    = Path("D:/Particiones/preprocesamiento_commonvoice_log.txt")

SAMPLE_RATE    = 16000
VAD_MODE       = 2
FRAME_MS       = 30
MIN_SPEECH_SEC = 0.5

# ---------------------------------------------------------------------------

def aplicar_vad(audio_int16, sr, mode):
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
        audio_int16 = (audio_float * 32767).astype(np.int16)
        audio_vad = aplicar_vad(audio_int16, SAMPLE_RATE, vad_mode)

        if len(audio_vad) / SAMPLE_RATE < MIN_SPEECH_SEC:
            return (ruta_str, "vacio")

        destino.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destino), audio_vad, SAMPLE_RATE, subtype="PCM_16")
        return (ruta_str, "ok")

    except Exception as e:
        return (ruta_str, f"error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vad-mode", type=int, default=VAD_MODE)
    args = parser.parse_args()

    logging.basicConfig(
        filename=str(LOG_FILE), filemode="a", level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Leer solo filas de CommonVoice del manifest
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        filas = [r for r in csv.DictReader(f) if r["db"] == "CommonVoice"]

    rutas = [r["ruta"] for r in filas]
    print(f"Clips Common Voice en manifest: {len(rutas):,}")

    # Contar cuántos ya están procesados
    ya_hechos = sum(
        1 for r in rutas
        if (OUTPUT_BASE / Path(r).relative_to(INPUT_BASE).with_suffix(".wav")).exists()
    )
    pendientes = len(rutas) - ya_hechos
    print(f"Ya procesados: {ya_hechos:,}  |  Pendientes: {pendientes:,}")

    if args.dry_run or pendientes == 0:
        if pendientes == 0:
            print("Todo ya procesado.")
        return

    tareas = [(r, args.vad_mode) for r in rutas]
    t0 = time.time()
    ok = vacios = errores = skips = 0

    with multiprocessing.Pool(args.workers) as pool:
        for i, (ruta, estado) in enumerate(pool.imap_unordered(procesar_archivo, tareas), 1):
            if estado == "ok":
                ok += 1
            elif estado == "skip":
                skips += 1
            elif estado == "vacio":
                vacios += 1
                logging.info(f"VACIO: {ruta}")
            else:
                errores += 1
                logging.warning(f"ERROR: {ruta} — {estado}")

            if i % 500 == 0:
                elapsed = time.time() - t0
                velocidad = i / elapsed
                restante = (len(tareas) - i) / velocidad
                print(f"  {i:,}/{len(tareas):,} — {velocidad:.1f} archivos/s — "
                      f"ETA {restante/60:.1f} min  (ok={ok} vacíos={vacios} err={errores})")

    elapsed = time.time() - t0
    print(f"\nFinalizado en {elapsed/60:.1f} min")
    print(f"ok={ok:,}  skip={skips:,}  vacíos={vacios:,}  errores={errores:,}")


if __name__ == "__main__":
    main()
