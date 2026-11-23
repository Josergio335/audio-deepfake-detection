"""
Compresión de ASVspoof2019 LA eval para el escenario R7 (robustez ante compresión).

Toma los .flac originales de eval (dev+eval), los recodifica a MP3 y AAC a varios
bitrates y genera un manifest por cada bitrate.

Destino: D:/Bases de Datos procesadas/ASVspoof_compresion/{codec}_{kbps}kbps/{bonafide|spoof}/
Manifests: D:/Particiones/manifest_test_compresion_{codec}_{kbps}kbps.csv

REANUDABLE: cada archivo de destino se salta si ya existe.
Requiere ffmpeg en el PATH.

Uso:
    python comprimir_asv_eval.py                    # todo
    python comprimir_asv_eval.py --dry-run          # muestra counts sin comprimir
    python comprimir_asv_eval.py --codec mp3        # solo MP3
    python comprimir_asv_eval.py --kbps 128         # solo un bitrate
    python comprimir_asv_eval.py --splits eval      # solo eval (no dev)
    python comprimir_asv_eval.py --splits dev eval  # dev y eval (default)
"""

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE        = Path("D:/")
FUENTE_LA   = BASE / "Bases de Datos extraidas" / "ASVspoof" / "ASVspoof2019" / "LA"
DESTINO     = BASE / "Bases de Datos procesadas" / "ASVspoof_compresion"
PARTICIONES = BASE / "Particiones"
LOG_FILE    = DESTINO / "compresion.log"

DESTINO.mkdir(parents=True, exist_ok=True)
PARTICIONES.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración de codecs y bitrates
# ---------------------------------------------------------------------------
CONFIGS = [
    # (codec, kbps, ffmpeg_codec, extension)
    ("mp3",  32,  "libmp3lame", "mp3"),
    ("mp3",  64,  "libmp3lame", "mp3"),
    ("mp3", 128,  "libmp3lame", "mp3"),
]

SPLITS = ["dev", "eval"]   # ambos son reservados (no usados en train)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verificar_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        log.error("ffmpeg no encontrado en el PATH. Instálalo o agrégalo al PATH.")
        sys.exit(1)


def comprimir_archivo(src: Path, dst: Path, ffmpeg_codec: str, kbps: int) -> bool:
    """Comprime src → dst. Devuelve True si OK, False si falla."""
    if dst.exists():
        return True   # ya hecho
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-c:a", ffmpeg_codec, "-b:a", f"{kbps}k",
         "-ar", "16000",   # mantener 16 kHz
         str(dst)],
        capture_output=True,
    )
    if result.returncode != 0:
        log.warning(f"Error comprimiendo {src.name}: {result.stderr.decode(errors='replace')[-200:]}")
        return False
    return True


def etiqueta(clase_dir: str) -> str:
    return "bonafide" if clase_dir == "bonafide" else "spoof"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def procesar(codec: str, kbps: int, ffmpeg_codec: str, ext: str,
             splits: list, dry_run: bool):
    nombre = f"{codec}_{kbps}kbps"
    directorio_dest = DESTINO / nombre
    manifest_path = PARTICIONES / f"manifest_test_compresion_{nombre}.csv"

    log.info(f"=== {nombre.upper()} ===")

    filas = []
    ok = salteados = errores = 0

    for split in splits:
        split_dir = FUENTE_LA / split / "en"
        if not split_dir.exists():
            log.warning(f"No existe {split_dir}, saltando.")
            continue

        for clase in ["bonafide", "spoof"]:
            clase_dir = split_dir / clase
            if not clase_dir.exists():
                continue
            archivos = sorted(clase_dir.glob("*.flac"))
            log.info(f"  {split}/{clase}: {len(archivos)} archivos")

            for src in archivos:
                dst = directorio_dest / split / clase / src.with_suffix(f".{ext}").name
                if dry_run:
                    filas.append({
                        "ruta": str(dst), "db": "ASVspoof2019_LA",
                        "idioma": "en", "label": etiqueta(clase),
                        "codec": codec, "kbps": kbps, "split_original": split,
                    })
                    continue

                exito = comprimir_archivo(src, dst, ffmpeg_codec, kbps)
                if exito:
                    if dst.stat().st_size > 0:
                        # Solo contar nuevo si no existía antes (aproximación)
                        ok += 1
                    filas.append({
                        "ruta": str(dst), "db": "ASVspoof2019_LA",
                        "idioma": "en", "label": etiqueta(clase),
                        "codec": codec, "kbps": kbps, "split_original": split,
                    })
                else:
                    errores += 1

                total = ok + salteados + errores
                if total % 1000 == 0 and total > 0:
                    log.info(f"  ... {total} procesados ({errores} errores)")

    # Escribir manifest
    if filas:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=filas[0].keys())
            writer.writeheader()
            writer.writerows(filas)
        log.info(f"  Manifest: {manifest_path} ({len(filas)} filas)")

    if not dry_run:
        log.info(f"  Resultado: {ok} comprimidos, {errores} errores")
    else:
        log.info(f"  [DRY-RUN] Se procesarían {len(filas)} archivos")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--codec",    choices=["mp3", "aac"], help="Solo este codec")
    parser.add_argument("--kbps",     type=int, help="Solo este bitrate")
    parser.add_argument("--splits",   nargs="+", default=SPLITS,
                        choices=["dev", "eval"], help="Splits a procesar")
    args = parser.parse_args()

    verificar_ffmpeg()

    configs = [
        c for c in CONFIGS
        if (args.codec is None or c[0] == args.codec)
        and (args.kbps  is None or c[1] == args.kbps)
    ]

    if not configs:
        log.error("Ninguna configuración coincide con los filtros dados.")
        sys.exit(1)

    for codec, kbps, ffmpeg_codec, ext in configs:
        procesar(codec, kbps, ffmpeg_codec, ext, args.splits, args.dry_run)

    log.info("Listo.")


if __name__ == "__main__":
    main()
