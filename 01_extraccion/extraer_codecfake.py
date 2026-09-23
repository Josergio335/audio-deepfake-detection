"""
Extractor de CodecFake: convierte los 208 shards Arrow a archivos WAV.

Ruta de entrada:  D:\\Bases de Datos sin extraer\\CodecFake\\rogertseng___codec_fake\\
                  default\\0.0.0\\ae224c9cff0e7a19edf885a670c2f7821711aed9\\
Ruta de salida:   D:\\Bases de Datos extraidas\\CodecFake\\train\\{label}\\

Uso:
    python extraer_codecfake.py               # extrae todo
    python extraer_codecfake.py --dry-run     # muestra cuántos shards/archivos sin extraer
    python extraer_codecfake.py --shard 0     # extrae solo el shard 0 (para pruebas)
"""

import os
import sys
import time
import argparse
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

SHARDS_DIR = Path(
    r"D:\Bases de Datos sin extraer\CodecFake\rogertseng___codec_fake"
    r"\default\0.0.0\ae224c9cff0e7a19edf885a670c2f7821711aed9"
)
OUTPUT_DIR = Path(r"D:\Bases de Datos extraidas\CodecFake\train")
LOG_FILE   = Path(r"D:\extraccion_log.txt")  # mismo log que extraer_datasets.py

# Mapeo de etiquetas a nombres de carpeta
LABEL_MAP = {
    "spoofing": "spoof",
    "genuine":  "genuine",   # bonafide VCTK original
    "bonafide": "bonafide",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------

def extraer_shard(shard_path: Path, dry_run: bool) -> tuple[int, int]:
    """
    Lee un shard Arrow y escribe cada muestra como WAV.
    Devuelve (extraidos, saltados).
    """
    extraidos = 0
    saltados  = 0

    with open(shard_path, "rb") as f:
        reader = ipc.open_stream(f)
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break

            for i in range(batch.num_rows):
                audio_col  = batch.column("audio")[i].as_py()
                label_col  = batch.column("label")[i].as_py()

                wav_bytes  = audio_col["bytes"]
                orig_path  = audio_col["path"]      # e.g. "funcodec-...+p282_268.wav"
                filename   = Path(orig_path).name   # solo el nombre de archivo

                folder_name = LABEL_MAP.get(label_col, label_col)
                dest_dir    = OUTPUT_DIR / folder_name
                dest_file   = dest_dir / filename

                if dest_file.exists():
                    saltados += 1
                    continue

                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    with open(dest_file, "wb") as out:
                        out.write(wav_bytes)

                extraidos += 1

    return extraidos, saltados


def main():
    parser = argparse.ArgumentParser(
        description="Extractor de CodecFake (Arrow -> WAV)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra qué haría sin escribir archivos")
    parser.add_argument("--shard", type=int, default=None, metavar="N",
                        help="Extrae solo el shard N (0-207), para pruebas")
    args = parser.parse_args()

    if args.dry_run:
        log("*** MODO DRY-RUN: no se escribirán archivos ***")

    if not SHARDS_DIR.exists():
        log(f"ERROR: directorio de shards no encontrado: {SHARDS_DIR}", "ERROR")
        sys.exit(1)

    shards = sorted(SHARDS_DIR.glob("codec_fake-train-*.arrow"))
    if not shards:
        log("ERROR: no se encontraron archivos .arrow", "ERROR")
        sys.exit(1)

    if args.shard is not None:
        shards = [s for s in shards if f"-{args.shard:05d}-of-" in s.name]
        if not shards:
            log(f"ERROR: shard {args.shard} no encontrado", "ERROR")
            sys.exit(1)

    log(f"Shards a procesar: {len(shards)}")
    log(f"Salida: {OUTPUT_DIR}")

    total_extraidos = 0
    total_saltados  = 0
    t0 = time.time()

    for idx, shard in enumerate(shards, 1):
        t_shard = time.time()
        print(f"  [{idx:3d}/{len(shards)}] {shard.name} ...", end=" ", flush=True)

        try:
            ext, sal = extraer_shard(shard, args.dry_run)
        except Exception as e:
            print(f"ERROR: {e}")
            log(f"Error en {shard.name}: {e}", "ERROR")
            continue

        elapsed = time.time() - t_shard
        print(f"extraidos={ext:,}  saltados={sal:,}  ({elapsed:.1f}s)")

        total_extraidos += ext
        total_saltados  += sal

        # Progreso parcial cada 20 shards
        if idx % 20 == 0:
            elapsed_total = time.time() - t0
            avg = elapsed_total / idx
            restantes = len(shards) - idx
            eta = avg * restantes
            log(f"Progreso: {idx}/{len(shards)} shards | "
                f"extraidos={total_extraidos:,} | saltados={total_saltados:,} | "
                f"ETA ~{eta/60:.0f} min")

    elapsed_total = time.time() - t0
    log(f"=== Extracción completada ===")
    log(f"  Archivos extraidos: {total_extraidos:,}")
    log(f"  Archivos saltados (ya existían): {total_saltados:,}")
    log(f"  Tiempo total: {elapsed_total/60:.1f} min")

    if not args.dry_run and total_extraidos + total_saltados > 0:
        # Verificar conteo final en disco
        for label_dir in sorted(OUTPUT_DIR.iterdir()):
            if label_dir.is_dir():
                count = sum(1 for _ in label_dir.glob("*.wav"))
                log(f"  {label_dir.name}/: {count:,} archivos WAV")


if __name__ == "__main__":
    main()
