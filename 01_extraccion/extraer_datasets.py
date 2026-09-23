"""
Script de extracción de bases de datos para tesis de detección de audio deepfake.
Maneja: .zip, .tar, .tar.gz, .tgz, .tar.bz2 y archivos split (.tar.gzaa, .tar.gzab, ...)

Uso:
    python extraer_datasets.py               # extrae todo
    python extraer_datasets.py --dry-run     # muestra qué haría sin ejecutar
    python extraer_datasets.py --solo MLAAD  # extrae solo una base de datos
"""

import os
import sys
import time
import tarfile
import zipfile
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

BASE = Path("D:/")

# ---------------------------------------------------------------------------
# Registro de progreso
# ---------------------------------------------------------------------------

LOG_FILE = BASE / "extraccion_log.txt"

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------------------------------------------------------------------------
# Helpers de extracción
# ---------------------------------------------------------------------------

def extraer_zip(src, dest, dry_run=False):
    log(f"ZIP: {src.name}  ->  {dest}")
    if dry_run:
        return
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as z:
        members = z.infolist()
        total = len(members)
        for i, member in enumerate(members, 1):
            z.extract(member, dest)
            if i % 5000 == 0 or i == total:
                print(f"  {i}/{total} archivos extraidos...", end="\r")
    print()
    log(f"  Completado: {total} archivos")


def extraer_tar(src, dest, dry_run=False):
    log(f"TAR: {src.name}  ->  {dest}")
    if dry_run:
        return
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src) as t:
        members = t.getmembers()
        total = len(members)
        for i, member in enumerate(members, 1):
            t.extract(member, dest, filter="data")
            if i % 5000 == 0 or i == total:
                print(f"  {i}/{total} archivos extraidos...", end="\r")
    print()
    log(f"  Completado: {total} archivos")


def extraer_split_tar(partes, dest, dry_run=False):
    """Concatena archivos split (.tar.gzaa, .tar.gzab, ...) y extrae con tar externo."""
    nombre_base = partes[0].name[:-2]  # quita "aa"
    log(f"SPLIT TAR ({len(partes)} partes): {nombre_base}* -> {dest}")
    if dry_run:
        return

    dest.mkdir(parents=True, exist_ok=True)

    cat_cmd = ["cat"] + [str(p) for p in sorted(partes)]
    tar_cmd = ["tar", "-xzf", "-", "-C", str(dest)]

    log("  Concatenando y extrayendo (puede tardar varios minutos)...")
    t0 = time.time()
    cat_proc = subprocess.Popen(cat_cmd, stdout=subprocess.PIPE)
    tar_proc = subprocess.Popen(tar_cmd, stdin=cat_proc.stdout)
    cat_proc.stdout.close()
    tar_proc.wait()
    cat_proc.wait()

    if tar_proc.returncode != 0:
        log("  ERROR al extraer archivos split", "ERROR")
    else:
        elapsed = time.time() - t0
        log(f"  Completado en {elapsed:.0f}s")


def ya_extraido(dest, marcador=None):
    """Devuelve True si el destino ya existe y no esta vacio."""
    if marcador:
        return (dest / marcador).exists()
    return dest.exists() and any(dest.iterdir())


# ---------------------------------------------------------------------------
# Tareas por base de datos
# ---------------------------------------------------------------------------

def extraer_ASVspoof(dry_run):
    d = BASE / "ASVspoof"
    log("=== ASVspoof ===")

    # ASVspoof2019 LA y PA (ZIP)
    for nombre_zip, subcarpeta in [("LA.zip", "LA"), ("PA.zip", "PA")]:
        src = d / nombre_zip
        dest_check = d / subcarpeta
        if src.exists():
            if ya_extraido(dest_check):
                log(f"  Ya extraido: {nombre_zip}, saltando")
            else:
                extraer_zip(src, d, dry_run)
        else:
            log(f"  No encontrado: {nombre_zip}", "WARN")

    # ASVspoof5 flac_*.tar (train, dev, eval)
    for tar_file in sorted(d.glob("flac_*.tar")):
        parte = "_".join(tar_file.stem.split("_")[:2])  # "flac_T", "flac_D", "flac_E"
        dest_dir = d / parte
        if ya_extraido(dest_dir):
            log(f"  Ya existe {parte}/, saltando: {tar_file.name}")
        else:
            extraer_tar(tar_file, d, dry_run)

    # Protocolos ASVspoof5 y metadatos (.tar.gz)
    for gz in sorted(d.glob("ASVspoof5_protocols.tar.gz")):
        if (d / "ASVspoof5.train.tsv").exists():
            log(f"  Protocolos ASVspoof5 ya extraidos, saltando")
        else:
            extraer_tar(gz, d, dry_run)

    # Metadatos 2019/2021
    meta_gz = d / "ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz"
    if meta_gz.exists():
        if (d / "ASVspoof2019_2021_VCTK_VCC_MetaInfo").exists():
            log("  Metadatos 2019/2021 ya extraidos, saltando")
        else:
            extraer_tar(meta_gz, d, dry_run)

    # ASVspoof2021 LA eval
    la21 = d / "ASVspoof2021_LA_eval.tar.gz"
    la21_dest = d / "ASVspoof2021_LA_eval"
    if la21.exists():
        if ya_extraido(la21_dest):
            log("  Ya extraido: ASVspoof2021_LA_eval, saltando")
        else:
            extraer_tar(la21, d, dry_run)
    else:
        log("  No encontrado: ASVspoof2021_LA_eval.tar.gz", "WARN")

    # ASVspoof2021 PA eval (partes)
    pa_parts = sorted(d.glob("ASVspoof2021_PA_eval_part*.tar.gz"))
    if pa_parts:
        pa21_dest = d / "ASVspoof2021_PA_eval"
        if ya_extraido(pa21_dest):
            log("  Ya extraido: ASVspoof2021_PA_eval, saltando")
        else:
            log(f"  ASVspoof2021 PA eval: {len(pa_parts)} partes")
            for p in pa_parts:
                extraer_tar(p, d, dry_run)


def extraer_MLAAD(dry_run):
    log("=== MLAAD ===")
    d = BASE / "MLAAD"
    fake_dir = d / "fake"
    if fake_dir.exists():
        count = sum(1 for f in fake_dir.rglob("*") if f.is_file())
        log(f"  fake/ ya extraida ({count} archivos). Sin mas archivos comprimidos en MLAAD.")
    else:
        log("  Carpeta fake/ no encontrada", "WARN")
    log("  Los audios reales (bonafide) de MLAAD estan en M-AILABS")


def extraer_MAILABS(dry_run):
    log("=== M-AILABS (bonafide de MLAAD) ===")
    d = BASE / "M-AILABS"
    for tgz in sorted(d.glob("*.tgz")):
        lang = tgz.stem  # "en_UK", "es_ES", etc.
        dest_dir = d / lang
        if ya_extraido(dest_dir):
            log(f"  Ya extraido: {tgz.name}, saltando")
        else:
            extraer_tar(tgz, d, dry_run)


def extraer_CodecFake(dry_run):
    log("=== CodecFake ===")
    shards_dir = (BASE / "Bases de Datos sin extraer" / "CodecFake" /
                  "rogertseng___codec_fake" / "default" / "0.0.0" /
                  "ae224c9cff0e7a19edf885a670c2f7821711aed9")
    output_dir = BASE / "Bases de Datos extraidas" / "CodecFake" / "train"

    shards = sorted(shards_dir.glob("codec_fake-train-*.arrow")) if shards_dir.exists() else []
    if not shards:
        log("  Shards Arrow no encontrados", "WARN")
        return

    already = sum(1 for _ in output_dir.rglob("*.wav")) if output_dir.exists() else 0
    log(f"  {len(shards)} shards Arrow encontrados")
    log(f"  WAVs ya extraidos: {already:,}")

    if dry_run:
        log("  (dry-run) Para extraer: python extraer_codecfake.py")
        return

    import subprocess
    script = BASE / "extraer_codecfake.py"
    if script.exists():
        log("  Lanzando extraer_codecfake.py ...")
        subprocess.run([sys.executable, str(script)], check=True)
    else:
        log("  Script extraer_codecfake.py no encontrado. Ejecutalo manualmente.", "WARN")


def extraer_SONAR(dry_run):
    log("=== SONAR ===")
    d = BASE / "SONAR"

    # ZIPs directos
    zips = {
        "SONAR_dataset.zip": d / "SONAR_dataset",
        "generated_audio.zip": d / "generated_audio",
        "LibriSeVoc.zip": d / "LibriSeVoc",
    }
    for nombre, dest in zips.items():
        src = d / nombre
        if src.exists():
            if ya_extraido(dest):
                log(f"  Ya extraido: {nombre}, saltando")
            else:
                extraer_zip(src, dest, dry_run)
        else:
            log(f"  No encontrado: {nombre}", "WARN")

    # LJSpeech (tar.bz2)
    ljspeech_src = d / "LJSpeech-1.1.tar.bz2"
    ljspeech_dest = d / "LJSpeech-1.1"
    if ljspeech_src.exists():
        if ya_extraido(ljspeech_dest):
            log("  Ya extraido: LJSpeech-1.1.tar.bz2, saltando")
        else:
            extraer_tar(ljspeech_src, d, dry_run)
    else:
        log("  No encontrado: LJSpeech-1.1.tar.bz2", "WARN")

    # In-the-Wild (ZIP anidado en carpeta HF cache)
    itw_zip = (d / "In the Wild" / "mueller91___in-the-wild" /
               "datasets--mueller91--In-The-Wild" / "snapshots" /
               "eee168f92c367f8c82ff2cf42b6f61e362fd6211" /
               "release_in_the_wild.zip")
    itw_dest = d / "In the Wild" / "release_in_the_wild"
    if itw_zip.exists():
        if ya_extraido(itw_dest):
            log("  Ya extraido: release_in_the_wild.zip, saltando")
        else:
            extraer_zip(itw_zip, itw_dest, dry_run)
    else:
        log("  No encontrado: release_in_the_wild.zip", "WARN")


def extraer_SpoofCeleb(dry_run):
    log("=== SpoofCeleb ===")
    d = BASE / "SpoofCeleb"
    inner_flac = d / "spoofceleb" / "spoofceleb" / "flac"

    if ya_extraido(inner_flac):
        log("  SpoofCeleb ya extraido (carpeta flac/ existe), saltando")
        return

    partes = sorted(d.glob("spoofceleb.tar.gz*"))
    if not partes:
        log("  No se encontraron partes spoofceleb.tar.gz*", "WARN")
        return

    log(f"  {len(partes)} partes encontradas")
    extraer_split_tar(partes, d, dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BASES_DE_DATOS = {
    "ASVspoof":   extraer_ASVspoof,
    "MLAAD":      extraer_MLAAD,
    "M-AILABS":   extraer_MAILABS,
    "CodecFake":  extraer_CodecFake,
    "SONAR":      extraer_SONAR,
    "SpoofCeleb": extraer_SpoofCeleb,
}

def main():
    parser = argparse.ArgumentParser(
        description="Extractor de datasets para tesis deepfake audio"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra que se haria sin ejecutar nada")
    parser.add_argument("--solo", metavar="DB",
                        help="Extrae solo una BD: " + ", ".join(BASES_DE_DATOS))
    args = parser.parse_args()

    if args.dry_run:
        log("*** MODO DRY-RUN: no se extraera nada ***")

    log(f"Inicio — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Directorio base: {BASE}")

    if args.solo:
        if args.solo not in BASES_DE_DATOS:
            print(f"ERROR: '{args.solo}' no reconocido. Opciones: {', '.join(BASES_DE_DATOS)}")
            sys.exit(1)
        BASES_DE_DATOS[args.solo](args.dry_run)
    else:
        for nombre, func in BASES_DE_DATOS.items():
            try:
                func(args.dry_run)
            except Exception as e:
                log(f"ERROR en {nombre}: {e}", "ERROR")

    log("Extraccion finalizada")

if __name__ == "__main__":
    main()
