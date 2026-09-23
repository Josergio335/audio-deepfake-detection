"""
preprocesamiento_dataset.py
Preprocesa los audios listados en el manifest de Exp-3.

Operaciones por archivo:
    1. Carga el audio (WAV o FLAC)
    2. Resamplea a 16 kHz mono
    3. Aplica WebRTC VAD para eliminar frames de silencio
    4. Guarda como WAV 16-bit en D:/Bases de Datos procesadas/
       replicando la estructura de carpetas del original

Caracteristicas:
    - Reanudable: salta archivos que ya existen en destino
    - Multiproceso: N workers en paralelo (default: 4)
    - Log de exito y errores en D:/Particiones/preprocesamiento_log.txt

Uso:
    python preprocesamiento_dataset.py                # 4 workers
    python preprocesamiento_dataset.py --workers 8    # 8 workers
    python preprocesamiento_dataset.py --dry-run      # muestra cuantos faltan sin procesar
    python preprocesamiento_dataset.py --vad-mode 2   # agresividad VAD 0-3 (default 2)
"""

import csv
import argparse
import logging
import multiprocessing
import os
import struct
import time
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import webrtcvad

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

MANIFEST       = Path(r"D:\Particiones\Exp3\manifest.csv")
INPUT_BASE     = Path(r"D:\Bases de Datos extraidas")
OUTPUT_BASE    = Path(r"D:\Bases de Datos procesadas")
LOG_FILE       = Path(r"D:\Particiones\preprocesamiento_log.txt")

SAMPLE_RATE    = 16000   # Hz objetivo
VAD_MODE       = 2       # agresividad WebRTC VAD: 0 (suave) a 3 (agresivo)
FRAME_MS       = 30      # duracion de frame VAD en ms (10, 20 o 30)
MIN_SPEECH_SEC = 0.5     # descartar audio si queda menos de esto tras VAD

# ---------------------------------------------------------------------------
# Logging (solo al archivo — stdout lo maneja el proceso principal)
# ---------------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        filename=str(LOG_FILE),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ---------------------------------------------------------------------------
# VAD: eliminar frames de silencio
# ---------------------------------------------------------------------------

def aplicar_vad(audio_int16: np.ndarray, sr: int, mode: int) -> np.ndarray:
    """
    Divide el audio en frames de FRAME_MS ms y descarta los silenciosos.
    Devuelve el audio concatenado con solo los frames con voz.
    """
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
# Procesamiento de un archivo (ejecutado en worker)
# ---------------------------------------------------------------------------

def procesar_archivo(args):
    """
    Retorna (ruta_origen, estado) donde estado es 'ok', 'skip', 'vacio' o 'error: <msg>'.
    """
    ruta_str, vad_mode = args
    ruta = Path(ruta_str)

    # Calcular ruta de destino
    try:
        rel = ruta.relative_to(INPUT_BASE)
    except ValueError:
        # La ruta no esta bajo INPUT_BASE — usar solo el nombre de archivo
        rel = Path(ruta.name)

    destino = OUTPUT_BASE / rel.with_suffix(".wav")

    # Saltar si ya existe
    if destino.exists():
        return (ruta_str, "skip")

    try:
        # 1. Cargar y resamplear a 16kHz mono
        audio_float, _ = librosa.load(ruta, sr=SAMPLE_RATE, mono=True)

        # 2. Convertir a int16 para WebRTC VAD
        audio_int16 = (audio_float * 32767).clip(-32768, 32767).astype(np.int16)

        # 3. Aplicar VAD
        audio_vad = aplicar_vad(audio_int16, SAMPLE_RATE, vad_mode)

        if len(audio_vad) < SAMPLE_RATE * MIN_SPEECH_SEC:
            return (ruta_str, f"vacio (menos de {MIN_SPEECH_SEC}s de voz tras VAD)")

        # 4. Guardar
        destino.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destino), audio_vad, SAMPLE_RATE, subtype="PCM_16")

        return (ruta_str, "ok")

    except Exception as e:
        return (ruta_str, f"error: {e}")


# ---------------------------------------------------------------------------
# Cargar manifest y calcular pendientes
# ---------------------------------------------------------------------------

def cargar_pendientes(vad_mode):
    """Devuelve lista de (ruta, vad_mode) para archivos aun no procesados."""
    pendientes = []
    ya_hechos = 0

    with open(MANIFEST, encoding="utf-8") as f:
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
        description="Preprocesa audios de Exp-3: resample 16kHz + WebRTC VAD"
    )
    parser.add_argument("--workers",  type=int, default=4,
                        help="Numero de procesos paralelos (default: 4)")
    parser.add_argument("--vad-mode", type=int, default=VAD_MODE, choices=[0,1,2,3],
                        help="Agresividad del VAD: 0=suave, 3=agresivo (default: 2)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Muestra cuantos archivos faltan sin procesar ninguno")
    args = parser.parse_args()

    setup_logging()
    logging.info("="*60)
    logging.info(f"PREPROCESAMIENTO INICIADO — workers={args.workers}, vad_mode={args.vad_mode}")

    print("=" * 62)
    print("  PREPROCESAMIENTO DE DATASET  —  Exp-3")
    print("=" * 62)
    print(f"  Manifest  : {MANIFEST}")
    print(f"  Destino   : {OUTPUT_BASE}")
    print(f"  Workers   : {args.workers}")
    print(f"  VAD mode  : {args.vad_mode}")
    print()

    print("Calculando archivos pendientes ...")
    pendientes, ya_hechos = cargar_pendientes(args.vad_mode)
    total = len(pendientes) + ya_hechos

    print(f"  Total en manifest : {total:,}")
    print(f"  Ya procesados     : {ya_hechos:,}")
    print(f"  Pendientes        : {len(pendientes):,}")

    if args.dry_run or not pendientes:
        if not pendientes:
            print("\nTodos los archivos ya estan procesados.")
        return

    print(f"\nIniciando procesamiento con {args.workers} workers ...")
    print("  (Ctrl+C para interrumpir — es reanudable)\n")

    t_inicio = time.time()
    contadores = {"ok": 0, "skip": 0, "vacio": 0, "error": 0}
    intervalo_reporte = 500  # imprimir progreso cada N archivos

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

            if i % intervalo_reporte == 0 or i == len(pendientes):
                elapsed = time.time() - t_inicio
                velocidad = i / elapsed if elapsed > 0 else 0
                restantes = len(pendientes) - i
                eta_seg = restantes / velocidad if velocidad > 0 else 0
                eta_h = eta_seg / 3600
                pct = i / len(pendientes) * 100
                print(
                    f"  [{pct:5.1f}%] {i:,}/{len(pendientes):,} "
                    f"| ok={contadores['ok']:,} err={contadores['error']:,} "
                    f"| {velocidad:.1f} arch/s "
                    f"| ETA: {eta_h:.1f}h"
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
