"""
Script de extracción de bases de datos para tesis de detección de audio deepfake.
Maneja: .zip, .tar, .tar.gz, .tgz, .tar.bz2 y archivos split (.tar.gzaa, .tar.gzab, ...)

Estructura esperada en disco:
    D:\Bases de Datos sin extraer\<BD>\   <- archivos comprimidos
    D:\Bases de Datos extraidas\<BD>\     <- destino de extracción

Excepciones:
    SONAR: los ZIPs conviven con las carpetas extraídas en D:\Bases de Datos extraidas\SONAR\
    CodecFake: shards Arrow en su propia ruta HuggingFace cache

Uso:
    python extraer_datasets.py               # extrae todo
    python extraer_datasets.py --dry-run     # muestra qué haría sin ejecutar
    python extraer_datasets.py --solo MLAAD  # extrae solo una base de datos
    python extraer_datasets.py --reset CLAVE # borra un marcador para forzar re-extracción
    python extraer_datasets.py --listar-marcadores

Los marcadores de extracción completada se guardan en D:\.marcadores_extraccion\
Para forzar la re-extracción de un archivo específico, borra su .done o usa --reset <CLAVE>.
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

BASE          = Path("D:/")
SIN_EXTRAER   = BASE / "Bases de Datos sin extraer"
EXTRAIDAS     = BASE / "Bases de Datos extraidas"

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
# Sistema de marcadores — reanudabilidad granular
# ---------------------------------------------------------------------------

MARKERS_DIR = BASE / ".marcadores_extraccion"


def ya_listo(clave):
    return (MARKERS_DIR / f"{clave}.done").exists()


def marcar_listo(clave, dry_run=False):
    if dry_run:
        return
    MARKERS_DIR.mkdir(parents=True, exist_ok=True)
    (MARKERS_DIR / f"{clave}.done").touch()
    log(f"  Marcando listo: {clave}")


def promover_si_existe(clave, dest, marcador_interno=None, dry_run=False):
    """
    Si ya hay marcador → saltar.
    Si no hay marcador pero la carpeta parece completa → promover y saltar.
    Devuelve True si se debe saltar.
    """
    if ya_listo(clave):
        return True
    carpeta_ok = (
        (dest / marcador_interno).exists() if marcador_interno
        else (dest.exists() and any(dest.iterdir()))
    )
    if carpeta_ok:
        log(f"  Carpeta ya existe, promoviendo a marcador: {clave}")
        marcar_listo(clave, dry_run)
        return True
    return False


def resetear_marcador(clave):
    marker = MARKERS_DIR / f"{clave}.done"
    if marker.exists():
        marker.unlink()
        print(f"Marcador eliminado: {clave}")
    else:
        print(f"No existe marcador para: {clave}")


# ---------------------------------------------------------------------------
# Helpers de extracción
# ---------------------------------------------------------------------------

def extraer_zip(src, dest, dry_run=False):
    log(f"ZIP: {src.name}  ->  {dest}")
    if dry_run:
        return True
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
    return True


def extraer_tar(src, dest, dry_run=False):
    log(f"TAR: {src.name}  ->  {dest}")
    if dry_run:
        return True
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
    return True


def extraer_split_tar(partes, dest, dry_run=False):
    """Concatena archivos split (.tar.gzaa, .tar.gzab, ...) y extrae en Python puro."""
    nombre_base = partes[0].name[:-2]  # quita "aa"
    log(f"SPLIT TAR ({len(partes)} partes): {nombre_base}* -> {dest}")
    if dry_run:
        return True

    dest.mkdir(parents=True, exist_ok=True)

    import io

    class ConcatStream(io.RawIOBase):
        def __init__(self, rutas):
            self._rutas = iter(rutas)
            self._fh = None
            self._abrir_siguiente()

        def _abrir_siguiente(self):
            if self._fh:
                self._fh.close()
                self._fh = None
            ruta = next(self._rutas, None)
            if ruta:
                self._fh = open(ruta, "rb")

        def readinto(self, b):
            while self._fh:
                n = self._fh.readinto(b)
                if n:
                    return n
                self._abrir_siguiente()
            return 0

        def readable(self):
            return True

    log("  Concatenando y extrayendo (puede tardar varios minutos)...")
    t0 = time.time()
    stream = io.BufferedReader(ConcatStream(sorted(partes)), buffer_size=1024 * 1024)
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as t:
            t.extractall(dest, filter="data")
        elapsed = time.time() - t0
        log(f"  Completado en {elapsed:.0f}s")
        return True
    except Exception as e:
        log(f"  ERROR al extraer archivos split: {e}", "ERROR")
        return False


# ---------------------------------------------------------------------------
# Tareas por base de datos
# ---------------------------------------------------------------------------

def extraer_ASVspoof(dry_run):
    src_dir  = SIN_EXTRAER / "ASVspoof"
    dest_dir = EXTRAIDAS   / "ASVspoof"
    log("=== ASVspoof ===")

    # ASVspoof2019 LA y PA (ZIP)
    for nombre_zip, subcarpeta in [("LA.zip", "LA"), ("PA.zip", "PA")]:
        src = src_dir / nombre_zip
        if not src.exists():
            log(f"  No encontrado: {nombre_zip}", "WARN")
            continue
        clave = f"ASVspoof_2019_{subcarpeta}"
        dest = dest_dir / "ASVspoof2019"
        if promover_si_existe(clave, dest / subcarpeta, dry_run=dry_run):
            log(f"  Ya extraido: {nombre_zip}, saltando")
        else:
            if extraer_zip(src, dest, dry_run):
                marcar_listo(clave, dry_run)

    # ASVspoof5 flac_*.tar (pueden ser split: flac_T_aa.tar, flac_T_ab.tar, ...)
    # Agrupa por prefijo (flac_T, flac_D, flac_E)
    flac_tars = sorted(src_dir.glob("flac_*.tar"))
    prefijos = {}
    for t in flac_tars:
        # stem: "flac_T_aa" → prefijo "flac_T"; stem: "flac_T" → prefijo "flac_T"
        partes = t.stem.split("_")
        prefijo = "_".join(partes[:2])  # "flac_T", "flac_D", "flac_E"
        prefijos.setdefault(prefijo, []).append(t)

    dest_asv5 = dest_dir / "ASVspoof5"
    for prefijo, archivos in sorted(prefijos.items()):
        tipo = prefijo.split("_")[1]  # T, D, E
        nombre_carpeta = {"T": "train", "D": "dev", "E": "eval"}.get(tipo, tipo)
        clave = f"ASVspoof_5_{prefijo}"
        if promover_si_existe(clave, dest_asv5 / nombre_carpeta, dry_run=dry_run):
            log(f"  Ya extraido: {prefijo}, saltando")
        else:
            if len(archivos) == 1:
                ok = extraer_tar(archivos[0], dest_asv5, dry_run)
            else:
                ok = extraer_split_tar(archivos, dest_asv5, dry_run)
            if ok:
                marcar_listo(clave, dry_run)

    # Protocolos ASVspoof5
    proto_gz = src_dir / "ASVspoof5_protocols.tar.gz"
    if proto_gz.exists():
        clave = "ASVspoof_5_protocols"
        if promover_si_existe(clave, dest_asv5 / "_protocols_asv5", dry_run=dry_run):
            log("  Protocolos ASVspoof5 ya extraidos, saltando")
        else:
            if extraer_tar(proto_gz, dest_asv5, dry_run):
                marcar_listo(clave, dry_run)

    # Metadatos 2019/2021
    meta_gz = src_dir / "ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz"
    if meta_gz.exists():
        clave = "ASVspoof_meta_2019_2021"
        dest = dest_dir / "ASVspoof2019"
        if promover_si_existe(clave, dest / "ASVspoof2019_2021_VCTK_VCC_MetaInfo", dry_run=dry_run):
            log("  Metadatos 2019/2021 ya extraidos, saltando")
        else:
            if extraer_tar(meta_gz, dest, dry_run):
                marcar_listo(clave, dry_run)

    # ASVspoof2021 LA eval
    la21 = src_dir / "ASVspoof2021_LA_eval.tar.gz"
    if la21.exists():
        clave = "ASVspoof_2021_LA_eval"
        dest = dest_dir / "ASVspoof2021" / "LA"
        if promover_si_existe(clave, dest, dry_run=dry_run):
            log("  Ya extraido: ASVspoof2021_LA_eval, saltando")
        else:
            if extraer_tar(la21, dest_dir / "ASVspoof2021", dry_run):
                marcar_listo(clave, dry_run)
    else:
        log("  No encontrado: ASVspoof2021_LA_eval.tar.gz", "WARN")

    # ASVspoof2021 PA eval (múltiples partes — cada parte tiene su propio marcador)
    pa_parts = sorted(src_dir.glob("ASVspoof2021_PA_eval_part*.tar.gz"))
    if pa_parts:
        dest_pa = dest_dir / "ASVspoof2021" / "PA"
        clave_conjunto = "ASVspoof_2021_PA_eval_completo"
        if promover_si_existe(clave_conjunto, dest_pa, dry_run=dry_run):
            log("  Ya extraido: ASVspoof2021_PA_eval, saltando")
        else:
            log(f"  ASVspoof2021 PA eval: {len(pa_parts)} partes")
            todas_ok = True
            for p in pa_parts:
                clave_parte = f"ASVspoof_2021_PA_eval_{p.stem}"
                if ya_listo(clave_parte):
                    log(f"  Ya extraida parte: {p.name}, saltando")
                    continue
                if extraer_tar(p, dest_dir / "ASVspoof2021", dry_run):
                    marcar_listo(clave_parte, dry_run)
                else:
                    todas_ok = False
            if todas_ok:
                marcar_listo(clave_conjunto, dry_run)


def extraer_MLAAD(dry_run):
    log("=== MLAAD ===")
    fake_dir = EXTRAIDAS / "MLAAD" / "fake"
    if fake_dir.exists():
        count = sum(1 for f in fake_dir.rglob("*") if f.is_file())
        log(f"  fake/ ya extraida ({count} archivos). Sin mas archivos comprimidos en MLAAD.")
    else:
        log("  Carpeta fake/ no encontrada", "WARN")
    log("  Los audios reales (bonafide) de MLAAD estan en M-AILABS")


def extraer_MAILABS(dry_run):
    log("=== M-AILABS (bonafide de MLAAD) ===")
    src_dir  = SIN_EXTRAER / "M-AILABS"
    dest_dir = EXTRAIDAS   / "M-AILABS"
    for tgz in sorted(src_dir.glob("*.tgz")):
        lang = tgz.stem  # "en_UK", "es_ES", etc.
        clave = f"MAILABS_{lang}"
        if promover_si_existe(clave, dest_dir / lang, dry_run=dry_run):
            log(f"  Ya extraido: {tgz.name}, saltando")
        else:
            if extraer_tar(tgz, dest_dir, dry_run):
                marcar_listo(clave, dry_run)


def extraer_CodecFake(dry_run):
    log("=== CodecFake ===")
    shards_dir = (SIN_EXTRAER / "CodecFake" /
                  "rogertseng___codec_fake" / "default" / "0.0.0" /
                  "ae224c9cff0e7a19edf885a670c2f7821711aed9")
    output_dir = EXTRAIDAS / "CodecFake" / "train"

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

    script = Path(__file__).parent / "extraer_codecfake.py"
    if script.exists():
        log("  Lanzando extraer_codecfake.py ...")
        subprocess.run([sys.executable, str(script)], check=True)
    else:
        log(f"  Script no encontrado: {script}. Ejecutalo manualmente.", "WARN")


def extraer_SONAR(dry_run):
    log("=== SONAR ===")
    # Los ZIPs de SONAR conviven con las carpetas extraídas en EXTRAIDAS/SONAR/
    d = EXTRAIDAS / "SONAR"

    zips = {
        "SONAR_dataset.zip": d / "SONAR_core",
        "generated_audio.zip": d / "generated_audio",
        "LibriSeVoc.zip": d / "LibriSeVoc",
    }
    for nombre, dest in zips.items():
        src = d / nombre
        if not src.exists():
            log(f"  No encontrado: {nombre}", "WARN")
            continue
        clave = f"SONAR_{Path(nombre).stem}"
        if promover_si_existe(clave, dest, dry_run=dry_run):
            log(f"  Ya extraido: {nombre}, saltando")
        else:
            if extraer_zip(src, d, dry_run):
                marcar_listo(clave, dry_run)

    # LJSpeech (tar.bz2)
    ljspeech_src = d / "LJSpeech-1.1.tar.bz2"
    if ljspeech_src.exists():
        clave = "SONAR_LJSpeech-1.1"
        if promover_si_existe(clave, d / "LJSpeech", dry_run=dry_run):
            log("  Ya extraido: LJSpeech-1.1.tar.bz2, saltando")
        else:
            if extraer_tar(ljspeech_src, d, dry_run):
                marcar_listo(clave, dry_run)
    else:
        log("  No encontrado: LJSpeech-1.1.tar.bz2", "WARN")

    # In-the-Wild
    itw_zip = d / "release_in_the_wild.zip"
    itw_dest = d / "In_the_Wild"
    if itw_zip.exists():
        clave = "SONAR_InTheWild"
        if promover_si_existe(clave, itw_dest, dry_run=dry_run):
            log("  Ya extraido: release_in_the_wild.zip, saltando")
        else:
            if extraer_zip(itw_zip, itw_dest, dry_run):
                marcar_listo(clave, dry_run)
    else:
        log("  No encontrado: release_in_the_wild.zip", "WARN")


def extraer_SpoofCeleb(dry_run):
    log("=== SpoofCeleb ===")
    src_dir  = SIN_EXTRAER / "SpoofCeleb"
    dest_dir = EXTRAIDAS   / "SpoofCeleb"

    clave = "SpoofCeleb"
    # La estructura interna del tar es spoofceleb/spoofceleb/flac/
    inner_flac = dest_dir / "spoofceleb" / "spoofceleb" / "flac"
    if promover_si_existe(clave, inner_flac, dry_run=dry_run):
        log("  SpoofCeleb ya extraido (carpeta flac/ existe), saltando")
        return

    partes = sorted(src_dir.glob("spoofceleb.tar.gz*"))
    if not partes:
        log("  No se encontraron partes spoofceleb.tar.gz*", "WARN")
        return

    log(f"  {len(partes)} partes encontradas")
    if extraer_split_tar(partes, dest_dir, dry_run):
        marcar_listo(clave, dry_run)


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
    parser.add_argument("--reset", metavar="CLAVE",
                        help="Borra el marcador de una extraccion para forzar re-extraccion")
    parser.add_argument("--listar-marcadores", action="store_true",
                        help="Muestra todos los marcadores de extraccion completada")
    args = parser.parse_args()

    if args.listar_marcadores:
        if MARKERS_DIR.exists():
            marcadores = sorted(MARKERS_DIR.glob("*.done"))
            if marcadores:
                print(f"Marcadores en {MARKERS_DIR}:")
                for m in marcadores:
                    print(f"  {m.stem}")
            else:
                print("No hay marcadores aun.")
        else:
            print("No hay marcadores aun.")
        return

    if args.reset:
        resetear_marcador(args.reset)
        return

    if args.dry_run:
        log("*** MODO DRY-RUN: no se extraera nada ***")

    log(f"Inicio — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Directorio base: {BASE}")
    log(f"Sin extraer:     {SIN_EXTRAER}")
    log(f"Extraidas:       {EXTRAIDAS}")
    log(f"Marcadores en:   {MARKERS_DIR}")

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
