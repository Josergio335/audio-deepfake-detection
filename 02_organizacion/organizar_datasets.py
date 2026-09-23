#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
organizar_datasets.py  v1.0
Organizacion, extraccion y estadisticas de datasets para tesis de deteccion
de audio deepfake.

REQUISITOS:
    ffmpeg (incluye ffprobe): https://ffmpeg.org/download.html
        Windows/Chocolatey:  choco install ffmpeg
        Windows/winget:      winget install Gyan.FFmpeg
    pip install tqdm   (opcional)

ESTRUCTURA RESULTANTE:
    D:/Bases de Datos sin extraer/
        ASVspoof/   M-AILABS/   SONAR/   SpoofCeleb/
        CodecFake/  MLAAD/   <- _REFERENCIA.txt (sin comprimido)

    D:/Bases de Datos extraidas/
        ASVspoof/
            ASVspoof2019/LA/{train|dev|eval}/en/{bonafide|spoof}/
            ASVspoof2019/PA/{train|dev|eval}/en/{bonafide|spoof}/
            ASVspoof2021/LA/eval/en/sin_etiqueta/
            ASVspoof2021/PA/eval/en/sin_etiqueta/
            ASVspoof5/{train|dev|eval}/multilingual/{bonafide|spoof}/
        MLAAD/fake/{idioma}/{modelo}/
        M-AILABS/{idioma}/
        CodecFake/datos/
        SONAR/
            In_the_Wild/en/{bonafide|spoof}/
            SONAR_core/en/{bonafide|spoof/{modelo}}/
            generated_audio/{en|ja}/{modelo}/
            LibriSeVoc/en/{modelo}/
            LJSpeech/
        SpoofCeleb/{train|development|evaluation}/en/{bonafide|spoof}/

    D:/duraciones.csv            <- mediciones incrementales (reanudable)
    D:/resumen_estadisticas.txt  <- tabla final
    D:/organizacion_log.txt      <- log

FASES:
    1  setup    Verifica ffprobe
    2  mover    Mueve comprimidos a "sin extraer"
    3  extraer  Extrae y organiza en "extraidas"
    4  medir    Mide duraciones con ffprobe (incremental, reanudable)
    5  resumen  Genera tabla final

USO:
    python organizar_datasets.py                  # todas las fases
    python organizar_datasets.py --fases 3,4,5    # fases especificas
    python organizar_datasets.py --fases 5        # solo tabla final
    python organizar_datasets.py --dry-run        # simula sin ejecutar
"""

import os, sys, shutil, tarfile, zipfile, csv, subprocess, argparse, time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# =============================================================================
# CONSTANTES
# =============================================================================
BASE            = Path("D:/")
DIR_SIN_EXTRAER = BASE / "Bases de Datos sin extraer"
DIR_EXTRAIDAS   = BASE / "Bases de Datos extraidas"
CSV_DURACIONES  = BASE / "duraciones.csv"
RESUMEN_TXT     = BASE / "resumen_estadisticas.txt"
LOG_FILE        = BASE / "organizacion_log.txt"
AUDIO_EXT       = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}
WORKERS         = max(1, os.cpu_count() - 1)   # fase 3 (CPU-bound)
WORKERS_IO      = os.cpu_count()               # fase 4 (I/O-bound: ffprobe)

IDIOMAS = {
    "am":"Amharico",      "ar":"Arabe",          "bg":"Bulgaro",
    "bn":"Bengali",       "cs":"Checo",           "da":"Danes",
    "de":"Aleman",        "el":"Griego",          "en":"Ingles",
    "es":"Espanol",       "et":"Estonio",         "fa":"Persa",
    "fi":"Finlandes",     "fr":"Frances",         "ga":"Gaelico irlandes",
    "ha":"Hausa",         "he":"Hebreo",          "hi":"Hindi",
    "hr":"Croata",        "hu":"Hungaro",         "id":"Indonesio",
    "ig":"Igbo",          "it":"Italiano",        "ja":"Japones",
    "jv":"Javanes",       "kn":"Canares",         "ko":"Coreano",
    "lb":"Luxemburgues",  "lt":"Lituano",         "lv":"Leton",
    "ml":"Malabar",       "mr":"Marati",          "ms":"Malayo",
    "mt":"Maltes",        "nl":"Holandes",        "no":"Noruego",
    "pl":"Polaco",        "pt":"Portugues",       "ro":"Rumano",
    "ru":"Ruso",          "si":"Cingales",        "sk":"Eslovaco",
    "sl":"Esloveno",      "sv":"Sueco",           "sw":"Suajili",
    "ta":"Tamil",         "th":"Tailandes",       "tk":"Turcomano",
    "tr":"Turco",         "uk":"Ucraniano",       "ur":"Urdu",
    "vi":"Vietnamita",    "yo":"Yoruba",          "zh-cn":"Chino mandarin",
    "multilingual":"Multilingue (MLS: en/de/nl/fr/es/it/pt/pl)",
    "sin_etiqueta":"Sin etiqueta (ver notas al pie)",
    "desconocido":"Desconocido",
}

def nombre_idioma(c):
    return IDIOMAS.get(c, c)

# =============================================================================
# LOGGING Y UTILIDADES
# =============================================================================
def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def fmt_dur(seg):
    seg = max(0, int(seg))
    return f"{seg//3600:02d}:{(seg%3600)//60:02d}:{seg%60:02d}"

def fmt_num(n):
    return f"{n:,}"

def ya_existe(path, marcador=None):
    p = Path(path)
    if marcador:
        return (p / marcador).exists()
    if p.is_dir():
        return p.exists() and any(p.iterdir())
    return p.exists()

# =============================================================================
# EXTRACCION (bajo nivel)
# =============================================================================
def _extraer_zip(src, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as z:
        items = z.infolist()
        total = len(items)
        for i, m in enumerate(items, 1):
            z.extract(m, dest)
            if i % 5000 == 0 or i == total:
                print(f"    {i}/{total}...", end="\r")
    print()

def _extraer_tar(src, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src) as t:
        items = t.getmembers()
        total = len(items)
        for i, m in enumerate(items, 1):
            t.extract(m, dest, filter="data")
            if i % 5000 == 0 or i == total:
                print(f"    {i}/{total}...", end="\r")
    print()

def _extraer_split_tar(partes, dest):
    dest.mkdir(parents=True, exist_ok=True)
    log(f"    Concatenando {len(partes)} partes...")
    cat = subprocess.Popen(
        ["cat"] + [str(p) for p in sorted(partes)],
        stdout=subprocess.PIPE)
    tar = subprocess.Popen(
        ["tar", "-xzf", "-", "-C", str(dest)],
        stdin=cat.stdout)
    cat.stdout.close()
    tar.wait()
    cat.wait()
    if tar.returncode != 0:
        log("    ERROR en extraccion split", "ERROR")

def copiar_audio(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)

def leer_protocolo_19(proto_file):
    mapa = {}
    with open(proto_file, encoding="utf-8") as f:
        for linea in f:
            p = linea.strip().split()
            if len(p) >= 5:
                mapa[p[1]] = p[4]
    return mapa

def leer_protocolo_asv5(tsv_file):
    mapa = {}
    with open(tsv_file, encoding="utf-8") as f:
        for linea in f:
            p = linea.strip().split()
            if len(p) >= 9:
                mapa[p[1]] = {"label": p[8], "speaker": p[0]}
    return mapa

# =============================================================================
# FASE 1: SETUP
# =============================================================================
def fase_setup(dry_run=False):
    log("=" * 70)
    log("FASE 1: Verificacion de requisitos")
    log("=" * 70)
    ok = True
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, timeout=5)
        log("  [OK] ffprobe encontrado")
    except FileNotFoundError:
        log("  [FALTA] ffprobe no encontrado", "ERROR")
        print("\n  Instala ffmpeg:")
        print("    choco install ffmpeg")
        print("    winget install Gyan.FFmpeg")
        print("    https://ffmpeg.org/download.html\n")
        ok = False
    try:
        import tqdm
        log("  [OK] tqdm disponible")
    except ImportError:
        log("  [WARN] tqdm no instalado (opcional): pip install tqdm", "WARN")
    if not ok and not dry_run:
        sys.exit(1)

# =============================================================================
# FASE 2: MOVER COMPRIMIDOS
# =============================================================================
def fase_mover(dry_run=False):
    log("=" * 70)
    log("FASE 2: Mover archivos comprimidos a \'Bases de Datos sin extraer\'")
    log("=" * 70)

    tareas = [
        (BASE / "ASVspoof", "ASVspoof", [
            "LA.zip", "PA.zip",
            "ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz",
            "ASVspoof2021_LA_eval.tar.gz",
            "ASVspoof2021_PA_eval_part*.tar.gz",
            "ASVspoof5_protocols.tar.gz",
            "flac_T_*.tar", "flac_D_*.tar", "flac_E_*.tar",
            "LICENSE.txt", "README.txt",
        ]),
        (BASE / "M-AILABS", "M-AILABS", ["*.tgz"]),
        (BASE / "SONAR", "SONAR", [
            "SONAR_dataset.zip", "generated_audio.zip",
            "LibriSeVoc.zip", "LJSpeech-1.1.tar.bz2",
            "LICENSE", "datasheet.pdf",
        ]),
        (BASE / "SpoofCeleb", "SpoofCeleb", [
            "spoofceleb.tar.gz*", "README.md", ".gitattributes",
        ]),
    ]

    total_mov = 0
    for origen, subdir, globs in tareas:
        destino = DIR_SIN_EXTRAER / subdir
        if not origen.exists():
            log(f"  Origen no encontrado: {origen}", "WARN")
            continue
        if not dry_run:
            destino.mkdir(parents=True, exist_ok=True)
        for patron in globs:
            for archivo in sorted(origen.glob(patron)):
                dest_a = destino / archivo.name
                if dest_a.exists():
                    continue
                log(f"  {subdir}/{archivo.name}")
                if not dry_run:
                    shutil.move(str(archivo), str(dest_a))
                total_mov += 1

    # In-the-Wild: ZIP anidado en cache HF
    itw_src = (BASE / "SONAR" / "In the Wild" / "mueller91___in-the-wild" /
               "datasets--mueller91--In-The-Wild" / "snapshots" /
               "eee168f92c367f8c82ff2cf42b6f61e362fd6211" /
               "release_in_the_wild.zip")
    itw_dest = DIR_SIN_EXTRAER / "SONAR" / "release_in_the_wild.zip"
    if itw_src.exists() and not itw_dest.exists():
        log("  SONAR/release_in_the_wild.zip")
        if not dry_run:
            (DIR_SIN_EXTRAER / "SONAR").mkdir(parents=True, exist_ok=True)
            shutil.copy2(itw_src, itw_dest)
        total_mov += 1

    for db, ruta in [("CodecFake", BASE / "CodecFake"), ("MLAAD", BASE / "MLAAD")]:
        ref = DIR_SIN_EXTRAER / db / "_REFERENCIA.txt"
        if not ref.exists() and not dry_run:
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(
                f"{db}: datos en {ruta}\n"
                "Descargado en formato descomprimido; no existe archivo comprimido.\n",
                encoding="utf-8")
        log(f"  Referencia: sin_extraer/{db}/_REFERENCIA.txt")

    log(f"  Total movidos: {total_mov} archivos")

# =============================================================================
# FASE 3: EXTRAER Y ORGANIZAR
# =============================================================================

def _organizar_19(temp_inner, dest_base, sistema, dry_run):
    proto_dir = temp_inner / f"ASVspoof2019_{sistema}_cm_protocols"
    if not proto_dir.exists():
        log(f"    Protocolos no encontrados: {proto_dir}", "WARN")
        return
    mapa = {}
    for pf in proto_dir.glob("*.txt"):
        n = pf.stem.lower()
        split = "train" if "train" in n else ("dev" if "dev" in n else "eval")
        for stem, label in leer_protocolo_19(pf).items():
            mapa[stem] = (label, split)
    audios = list(temp_inner.rglob("*.flac")) + list(temp_inner.rglob("*.wav"))
    log(f"    {len(audios)} audios | {len(mapa)} en protocolo")
    for audio in audios:
        stem = audio.stem
        if stem in mapa:
            label, split = mapa[stem]
        else:
            s = "_T_" if "_T_" in stem else ("_D_" if "_D_" in stem else "_E_")
            split = {"_T_": "train", "_D_": "dev", "_E_": "eval"}.get(s, "eval")
            label = "desconocido"
        if not dry_run:
            copiar_audio(audio, dest_base / split / "en" / label / audio.name)


def fase_extraer_asvspoof(dry_run=False):
    log("  [ASVspoof]")
    src = DIR_SIN_EXTRAER / "ASVspoof"

    for zip_name, subdir, sistema in [
        ("LA.zip", "ASVspoof2019/LA", "LA"),
        ("PA.zip", "ASVspoof2019/PA", "PA"),
    ]:
        src_z = src / zip_name
        dest  = DIR_EXTRAIDAS / "ASVspoof" / subdir
        if not src_z.exists():
            log(f"    {zip_name} no encontrado", "WARN"); continue
        if ya_existe(dest, "train"):
            log(f"    ASVspoof2019/{sistema} ya organizado, saltando"); continue
        log(f"    Extrayendo {zip_name}...")
        temp = dest / "_temp"
        if not dry_run:
            _extraer_zip(src_z, temp)
            _organizar_19(temp / sistema, dest, sistema, dry_run)
            shutil.rmtree(temp, ignore_errors=True)

    src_la21 = src / "ASVspoof2021_LA_eval.tar.gz"
    dest_la21 = DIR_EXTRAIDAS / "ASVspoof" / "ASVspoof2021" / "LA" / "eval" / "en" / "sin_etiqueta"
    if src_la21.exists():
        if ya_existe(dest_la21):
            log("    ASVspoof2021/LA ya organizado, saltando")
        else:
            log("    Extrayendo ASVspoof2021 LA eval...")
            log("    NOTA: etiquetas bonafide/spoof no publicadas para ASVspoof2021")
            if not dry_run:
                temp = dest_la21.parent / "_temp_la21"
                try:
                    _extraer_tar(src_la21, temp)
                    for a in temp.rglob("*.flac"):
                        copiar_audio(a, dest_la21 / a.name)
                    shutil.rmtree(temp, ignore_errors=True)
                except Exception as e:
                    log(f"    [ERROR] ASVspoof2021_LA_eval.tar.gz: {e}", "ERROR")
                    log("    Descarga el archivo nuevamente y vuelve a ejecutar", "ERROR")
    else:
        log("    ASVspoof2021_LA_eval.tar.gz no encontrado", "WARN")

    pa21 = sorted(src.glob("ASVspoof2021_PA_eval_part*.tar.gz"))
    dest_pa21 = DIR_EXTRAIDAS / "ASVspoof" / "ASVspoof2021" / "PA" / "eval" / "en" / "sin_etiqueta"
    if pa21:
        if ya_existe(dest_pa21):
            log("    ASVspoof2021/PA ya organizado, saltando")
        else:
            log(f"    Extrayendo ASVspoof2021 PA eval ({len(pa21)} partes)...")
            log("    NOTA: etiquetas bonafide/spoof no publicadas para ASVspoof2021")
            if not dry_run:
                progreso = dest_pa21.parent / "_partes_pa21.txt"
                ya_ext = set(progreso.read_text(encoding="utf-8").splitlines()) if progreso.exists() else set()
                errores = []
                for parte in pa21:
                    if parte.name in ya_ext:
                        log(f"      -> {parte.name} (ya extraida, saltando)"); continue
                    log(f"      -> {parte.name}")
                    temp_p = dest_pa21.parent / f"_temp_{parte.stem}"
                    try:
                        _extraer_tar(parte, temp_p)
                        for a in temp_p.rglob("*.flac"):
                            copiar_audio(a, dest_pa21 / a.name)
                        shutil.rmtree(temp_p, ignore_errors=True)
                        with open(progreso, "a", encoding="utf-8") as pf:
                            pf.write(parte.name + "\n")
                    except Exception as e:
                        log(f"      [ERROR] {parte.name}: {e} -- saltando parte corrupta", "ERROR")
                        errores.append(parte.name)
                        shutil.rmtree(temp_p, ignore_errors=True)
                progreso.unlink(missing_ok=True)
                if errores:
                    log(f"    ADVERTENCIA: {len(errores)} partes con error: {errores}", "WARN")
                    log("    Reemplaza los archivos corruptos y vuelve a ejecutar", "WARN")

    dest5 = DIR_EXTRAIDAS / "ASVspoof" / "ASVspoof5"
    _splits_ok = all(
        ya_existe(dest5 / s / "multilingual") and any((dest5 / s / "multilingual").rglob("*.flac"))
        for s in ("train", "dev", "eval")
    )
    if _splits_ok:
        log("    ASVspoof5 ya organizado, saltando")
    else:
        proto_gz  = src / "ASVspoof5_protocols.tar.gz"
        proto_dir = src / "_protocols_asv5"
        if proto_gz.exists() and not ya_existe(proto_dir, "ASVspoof5.train.tsv"):
            log("    Extrayendo protocolos ASVspoof5...")
            if not dry_run:
                _extraer_tar(proto_gz, proto_dir)
        mapa5 = {}
        if not dry_run and proto_dir.exists():
            for tsv_name, split in [
                ("ASVspoof5.train.tsv", "train"),
                ("ASVspoof5.dev.track_1.tsv", "dev"),
                ("ASVspoof5.eval.track_1.tsv", "eval"),
            ]:
                tsv = proto_dir / tsv_name
                if tsv.exists():
                    for stem, info in leer_protocolo_asv5(tsv).items():
                        mapa5[stem] = {**info, "split": split}
            log(f"    ASVspoof5 protocolo: {len(mapa5)} entradas")
        nota = (
            "ASVspoof5 basado en Multilingual LibriSpeech (MLS).\n"
            "Idiomas: en de nl fr es it pt pl\n"
            "Mapeo speaker->idioma requiere metadata MLS: https://www.openslr.org/94/\n"
        )
        grupos = {
            "train": sorted(src.glob("flac_T_*.tar")),
            "dev":   sorted(src.glob("flac_D_*.tar")),
            "eval":  sorted(src.glob("flac_E_*.tar")),
        }
        for split, tars in grupos.items():
            if not tars:
                log(f"    ASVspoof5 {split}: sin tars", "WARN"); continue
            split_dest = dest5 / split / "multilingual"
            if ya_existe(split_dest) and any(split_dest.rglob("*.flac")):
                log(f"    ASVspoof5 {split}: ya organizado, saltando"); continue
            temp = dest5 / f"_temp_{split}"
            log(f"    ASVspoof5 {split}: {len(tars)} tars...")
            if not dry_run:
                progreso = dest5 / f"_tars_{split}.txt"
                ya_ext = set(progreso.read_text(encoding="utf-8").splitlines()) if progreso.exists() else set()
                errores = []
                for tar in tars:
                    if tar.name in ya_ext:
                        log(f"      -> {tar.name} (ya extraido, saltando)"); continue
                    log(f"      -> {tar.name}")
                    temp_t = dest5 / f"_temp_{split}_{tar.stem}"
                    try:
                        _extraer_tar(tar, temp_t)
                        for audio in temp_t.rglob("*.flac"):
                            label = mapa5.get(audio.stem, {}).get("label", "desconocido")
                            copiar_audio(audio, dest5 / split / "multilingual" / label / audio.name)
                        shutil.rmtree(temp_t, ignore_errors=True)
                        with open(progreso, "a", encoding="utf-8") as pf:
                            pf.write(tar.name + "\n")
                    except Exception as e:
                        log(f"      [ERROR] {tar.name}: {e} -- saltando tar corrupto", "ERROR")
                        errores.append(tar.name)
                        shutil.rmtree(temp_t, ignore_errors=True)
                nota_p = dest5 / split / "multilingual" / "NOTA_IDIOMA.txt"
                nota_p.parent.mkdir(parents=True, exist_ok=True)
                nota_p.write_text(nota, encoding="utf-8")
                if errores:
                    log(f"    ADVERTENCIA: {len(errores)} tars con error en {split}: {errores}", "WARN")
                progreso.unlink(missing_ok=True)
        log("    ASVspoof5 organizado")


def fase_extraer_mlaad(dry_run=False):
    log("  [MLAAD]")
    src  = DIR_SIN_EXTRAER / "MLAAD" / "fake"
    dest = DIR_EXTRAIDAS / "MLAAD" / "fake"
    if not src.exists():
        log("    MLAAD/fake no encontrado", "WARN"); return
    if ya_existe(dest) and any(dest.iterdir()):
        log("    MLAAD ya organizado, saltando"); return
    log("    Copiando fake/{idioma}/{modelo}/...")
    if not dry_run:
        for lang_dir in sorted(src.iterdir()):
            if not lang_dir.is_dir(): continue
            for model_dir in sorted(lang_dir.iterdir()):
                if not model_dir.is_dir(): continue
                dd = dest / lang_dir.name / model_dir.name
                dd.mkdir(parents=True, exist_ok=True)
                for f in model_dir.iterdir():
                    if f.suffix.lower() in AUDIO_EXT or f.name == "meta.csv":
                        t = dd / f.name
                        if not t.exists():
                            shutil.copy2(f, t)
    log("    MLAAD organizado")


def fase_extraer_mailabs(dry_run=False):
    log("  [M-AILABS]")
    src_dir = DIR_SIN_EXTRAER / "M-AILABS"
    errores = []
    for tgz in sorted(src_dir.glob("*.tgz")):
        lang = tgz.stem
        dest = DIR_EXTRAIDAS / "M-AILABS" / lang
        if ya_existe(dest):
            log(f"    {lang}: ya extraido, saltando"); continue
        log(f"    Extrayendo {tgz.name}...")
        if not dry_run:
            try:
                _extraer_tar(tgz, DIR_EXTRAIDAS / "M-AILABS")
                log(f"    {lang}: OK")
            except Exception as e:
                log(f"    [ERROR] {tgz.name}: {e} -- archivo corrupto, saltando", "ERROR")
                errores.append(tgz.name)
        else:
            log(f"    {lang}: OK")
    if errores:
        log(f"    ADVERTENCIA: {len(errores)} archivos con error: {errores}", "WARN")
        log("    Descarga los archivos corruptos y vuelve a ejecutar", "WARN")


def fase_extraer_codecfake(dry_run=False):
    log("  [CodecFake]")
    # Detectar snapshot disponible dinamicamente (no depende de hash hardcodeado)
    _cf_base = DIR_SIN_EXTRAER / "CodecFake" / "datasets--rogertseng--CodecFake" / "snapshots"
    src = None
    if _cf_base.exists():
        for _snap in sorted(_cf_base.iterdir()):
            # Los archivos pueden estar en el snapshot directamente o en snapshot/data/
            _candidate = _snap / "data" if (_snap / "data").exists() else _snap
            if _candidate.exists():
                src = _candidate; break
    dest = DIR_EXTRAIDAS / "CodecFake" / "datos"
    if src is None:
        log("    CodecFake: directorio no encontrado", "WARN"); return
    _cf_ext = {".parquet", ".arrow"}
    if ya_existe(dest) and any(f for f in dest.iterdir() if f.suffix in _cf_ext):
        log("    CodecFake ya organizado, saltando"); return
    log("    Copiando archivos de datos CodecFake (Ingles/VCTK)...")
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        copiados = 0
        for f in sorted(src.iterdir()):
            if f.suffix in _cf_ext:
                d = dest / f.name
                if not d.exists(): shutil.copy2(f, d); copiados += 1
        log(f"    {copiados} archivos copiados")
        (dest / "NOTA.txt").write_text(
            "CodecFake: corpus VCTK (Ingles puro).\n"
            "Formato Arrow/Parquet (HuggingFace Datasets).\n"
            "Columnas: audio, label, speaker_id, codec_name.\n"
            "label=real -> bonafide; otro valor -> spoof (codec re-sintetizado).\n"
            "Total: 707,872 muestras.\n", encoding="utf-8")
    log("    CodecFake organizado")


def fase_extraer_sonar(dry_run=False):
    log("  [SONAR]")
    src = DIR_SIN_EXTRAER / "SONAR"

    s1 = src / "SONAR_dataset.zip"
    d1 = DIR_EXTRAIDAS / "SONAR" / "SONAR_core"
    if s1.exists() and not ya_existe(d1):
        log("    Extrayendo SONAR_dataset.zip...")
        if not dry_run:
            temp = d1 / "_temp"
            try:
                _extraer_zip(s1, temp)
                inner = temp / "SONAR_dataset"
                for sub in inner.iterdir():
                    if sub.name in ("seedtts.csv", "README.md"): continue
                    dd = (d1 / "en" / "bonafide" if sub.name == "real_samples"
                          else d1 / "en" / "spoof" / sub.name)
                    if sub.is_dir():
                        dd.mkdir(parents=True, exist_ok=True)
                        for f in sub.iterdir():
                            if f.suffix.lower() in AUDIO_EXT:
                                copiar_audio(f, dd / f.name)
                shutil.rmtree(temp, ignore_errors=True)
            except Exception as e:
                log(f"    [ERROR] SONAR_dataset.zip: {e}", "ERROR")
                shutil.rmtree(temp, ignore_errors=True)
    elif ya_existe(d1): log("    SONAR_core ya organizado, saltando")
    else: log("    SONAR_dataset.zip no encontrado", "WARN")

    s2 = src / "release_in_the_wild.zip"
    d2 = DIR_EXTRAIDAS / "SONAR" / "In_the_Wild"
    if s2.exists() and not ya_existe(d2):
        log("    Extrayendo In-the-Wild (meta.csv -> bonafide/spoof)...")
        if not dry_run:
            temp = d2 / "_temp"
            try:
                _extraer_zip(s2, temp)
                inner = temp / "release_in_the_wild"
                meta = {}
                with open(inner / "meta.csv", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        meta[row["file"]] = row["label"]
                for audio in inner.glob("*.wav"):
                    label = meta.get(audio.name, "desconocido")
                    copiar_audio(audio, d2 / "en" / label / audio.name)
                shutil.rmtree(temp, ignore_errors=True)
            except Exception as e:
                log(f"    [ERROR] release_in_the_wild.zip: {e}", "ERROR")
                shutil.rmtree(temp, ignore_errors=True)
    elif ya_existe(d2): log("    In-the-Wild ya organizado, saltando")
    else: log("    release_in_the_wild.zip no encontrado", "WARN")

    s3 = src / "generated_audio.zip"
    d3 = DIR_EXTRAIDAS / "SONAR" / "generated_audio"
    if s3.exists() and not ya_existe(d3):
        log("    Extrayendo generated_audio.zip...")
        if not dry_run:
            temp = d3 / "_temp"
            try:
                _extraer_zip(s3, temp)
                for mod in (temp / "generated_audio").iterdir():
                    if not mod.is_dir(): continue
                    lang = "ja" if "jsut" in mod.name else "en"
                    dd = d3 / lang / mod.name
                    dd.mkdir(parents=True, exist_ok=True)
                    for f in mod.iterdir():
                        if f.suffix.lower() in AUDIO_EXT:
                            copiar_audio(f, dd / f.name)
                shutil.rmtree(temp, ignore_errors=True)
            except Exception as e:
                log(f"    [ERROR] generated_audio.zip: {e}", "ERROR")
                shutil.rmtree(temp, ignore_errors=True)
    elif ya_existe(d3): log("    generated_audio ya organizado, saltando")

    s4 = src / "LibriSeVoc.zip"
    d4 = DIR_EXTRAIDAS / "SONAR" / "LibriSeVoc" / "en"
    if s4.exists() and not ya_existe(d4):
        log("    Extrayendo LibriSeVoc.zip...")
        if not dry_run:
            temp = d4.parent / "_temp"
            try:
                _extraer_zip(s4, temp)
                for mod in (temp / "LibriSeVoc").iterdir():
                    if not mod.is_dir(): continue
                    dd = d4 / mod.name
                    dd.mkdir(parents=True, exist_ok=True)
                    for f in mod.iterdir():
                        if f.suffix.lower() in AUDIO_EXT:
                            copiar_audio(f, dd / f.name)
                shutil.rmtree(temp, ignore_errors=True)
            except Exception as e:
                log(f"    [ERROR] LibriSeVoc.zip: {e}", "ERROR")
                shutil.rmtree(temp, ignore_errors=True)
    elif ya_existe(d4): log("    LibriSeVoc ya organizado, saltando")

    s5 = src / "LJSpeech-1.1.tar.bz2"
    d5 = DIR_EXTRAIDAS / "SONAR" / "LJSpeech"
    if s5.exists() and not ya_existe(d5):
        log("    Extrayendo LJSpeech-1.1.tar.bz2...")
        if not dry_run:
            try:
                _extraer_tar(s5, d5)
            except Exception as e:
                log(f"    [ERROR] LJSpeech-1.1.tar.bz2: {e}", "ERROR")
                shutil.rmtree(d5, ignore_errors=True)
    elif ya_existe(d5): log("    LJSpeech ya organizado, saltando")

    log("    SONAR organizado")


def fase_extraer_spoofceleb(dry_run=False):
    log("  [SpoofCeleb]")
    dest = DIR_EXTRAIDAS / "SpoofCeleb"
    _sc_ok = all(
        ya_existe(dest / s / "en") and any((dest / s / "en").rglob("*.flac"))
        for s in ("train", "development", "evaluation")
    )
    if _sc_ok:
        log("    SpoofCeleb ya organizado, saltando"); return

    src_ext = BASE / "SpoofCeleb" / "spoofceleb" / "spoofceleb"
    partes  = sorted((DIR_SIN_EXTRAER / "SpoofCeleb").glob("spoofceleb.tar.gz*"))

    if src_ext.exists():
        log("    Usando SpoofCeleb ya extraido en D:\\SpoofCeleb\\")
        flac_base = src_ext / "flac"
        meta_dir  = src_ext / "metadata"
    elif partes:
        log(f"    Extrayendo desde {len(partes)} partes split...")
        temp = dest / "_temp"
        if not dry_run: _extraer_split_tar(partes, temp)
        flac_base = temp / "spoofceleb" / "flac"
        meta_dir  = temp / "spoofceleb" / "metadata"
    else:
        log("    SpoofCeleb: no se encontro fuente", "WARN"); return

    if dry_run: return

    for csv_name, split in [("train.csv", "train"),
                              ("development.csv", "development"),
                              ("evaluation.csv", "evaluation")]:
        meta_csv = meta_dir / csv_name
        if not meta_csv.exists():
            log(f"    Metadata no encontrada: {csv_name}", "WARN"); continue
        log(f"    Organizando {split}...")
        with open(meta_csv, encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        for fila in filas:
            rel   = fila["file"]
            atq   = rel.split("/")[0]
            label = "bonafide" if atq == "a00" else "spoof"
            a_src = flac_base / split / rel
            a_dst = dest / split / "en" / label / Path(rel).name
            if a_src.exists(): copiar_audio(a_src, a_dst)
        log(f"    {split}: OK")

    tmp = dest / "_temp"
    if tmp.exists(): shutil.rmtree(tmp, ignore_errors=True)
    log("    SpoofCeleb organizado")


def fase_extraer(dry_run=False):
    log("=" * 70)
    log("FASE 3: Extraccion y organizacion en \'Bases de Datos extraidas\'")
    log("=" * 70)
    DIR_EXTRAIDAS.mkdir(parents=True, exist_ok=True)
    for nombre, func in [
        ("ASVspoof",   fase_extraer_asvspoof),
        ("MLAAD",      fase_extraer_mlaad),
        ("M-AILABS",   fase_extraer_mailabs),
        ("CodecFake",  fase_extraer_codecfake),
        ("SONAR",      fase_extraer_sonar),
        ("SpoofCeleb", fase_extraer_spoofceleb),
    ]:
        try:
            func(dry_run)
        except Exception as e:
            log(f"  ERROR en {nombre}: {e}", "ERROR")
            import traceback; traceback.print_exc()

# =============================================================================
# FASE 4: MEDIR DURACIONES
# =============================================================================
def _ffprobe_dur(path_str):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path_str],
            capture_output=True, text=True, timeout=30)
        txt = r.stdout.strip()
        return float(txt) if txt else 0.0
    except Exception:
        return 0.0

def _parsear_ruta(path_str):
    """Extrae (db, version, split, idioma, label) segun la estructura de cada dataset."""
    try:
        rel = Path(path_str).relative_to(DIR_EXTRAIDAS)
        p   = rel.parts
        db  = p[0] if p else "-"

        if db == "ASVspoof":
            version = p[1] if len(p) > 1 else "-"
            if version == "ASVspoof5":
                # ASVspoof/ASVspoof5/{split}/multilingual/{label}/file
                split  = p[2] if len(p) > 2 else "-"
                idioma = p[3] if len(p) > 3 else "-"
                label  = p[4] if len(p) > 4 else "-"
            else:
                # ASVspoof/ASVspoof201X/{LA|PA}/{split}/{idioma}/{label}/file
                split  = p[3] if len(p) > 3 else "-"
                idioma = p[4] if len(p) > 4 else "-"
                label  = p[5] if len(p) > 5 else "-"
            return (db, version, split, idioma, label)

        elif db == "M-AILABS":
            # M-AILABS/{lang_tag}/by_book/{gender}/{speaker}/...
            lang_tag = p[1] if len(p) > 1 else "-"
            idioma   = lang_tag.split("_")[0] if "_" in lang_tag else lang_tag
            return (db, lang_tag, "-", idioma, "bonafide")

        elif db == "MLAAD":
            # MLAAD/fake/{lang}/{model}/file
            idioma = p[2] if len(p) > 2 else "-"
            return (db, "fake", "-", idioma, "spoof")

        elif db == "SONAR":
            sub = p[1] if len(p) > 1 else "-"
            if sub in ("In_the_Wild", "SONAR_core"):
                idioma = p[2] if len(p) > 2 else "-"
                label  = p[3] if len(p) > 3 else "-"
                return (db, sub, "-", idioma, label)
            elif sub in ("generated_audio", "LibriSeVoc"):
                idioma = p[2] if len(p) > 2 else "-"
                return (db, sub, "-", idioma, "spoof")
            elif sub == "LJSpeech":
                return (db, sub, "-", "en", "bonafide")
            else:
                return (db, sub, "-", "-", "-")

        elif db == "SpoofCeleb":
            # SpoofCeleb/{split}/en/{bonafide|spoof}/file
            split  = p[1] if len(p) > 1 else "-"
            idioma = p[2] if len(p) > 2 else "-"
            label  = p[3] if len(p) > 3 else "-"
            return (db, "-", split, idioma, label)

        elif db == "CodecFake":
            # CodecFake/train/{genuine|spoof}/file.wav  — basado en VCTK (ingles)
            split  = p[1] if len(p) > 1 else "-"
            folder = p[2] if len(p) > 2 else "-"
            label  = "bonafide" if folder == "genuine" else "spoof"
            return (db, "-", split, "en", label)

        else:
            return (db,
                    p[1] if len(p) > 1 else "-",
                    p[2] if len(p) > 2 else "-",
                    p[3] if len(p) > 3 else "-",
                    p[4] if len(p) > 4 else "-")
    except Exception:
        return ("-", "-", "-", "-", "-")

def fase_medir(dry_run=False):
    log("=" * 70)
    log("FASE 4: Medicion de duraciones (ffprobe paralelo, reanudable)")
    log("=" * 70)

    if dry_run:
        n = sum(1 for p in DIR_EXTRAIDAS.rglob("*")
                if p.is_file() and p.suffix.lower() in AUDIO_EXT)
        log(f"  [dry-run] {fmt_num(n)} archivos serian medidos")
        return

    ya_medidos = set()
    if CSV_DURACIONES.exists():
        with open(CSV_DURACIONES, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ya_medidos.add(row["ruta"])
        log(f"  Reanudando: {fmt_num(len(ya_medidos))} ya medidos")

    pendientes = [
        str(p) for p in DIR_EXTRAIDAS.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXT
        and str(p) not in ya_medidos
    ]
    log(f"  {fmt_num(len(pendientes))} archivos pendientes (workers: {WORKERS_IO})")
    if not pendientes:
        log("  Nada pendiente de medir"); return

    modo = "a" if CSV_DURACIONES.exists() else "w"
    with open(CSV_DURACIONES, modo, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if modo == "w":
            writer.writerow(["ruta","db","version","split","idioma","label","duracion_seg"])
        t0 = time.time()
        n = 0
        total = len(pendientes)
        with ProcessPoolExecutor(max_workers=WORKERS_IO) as pool:
            futuros = {pool.submit(_ffprobe_dur, p): p for p in pendientes}
            for fut in as_completed(futuros):
                p = futuros[fut]
                dur = fut.result()
                writer.writerow([p, *_parsear_ruta(p), f"{dur:.3f}"])
                n += 1
                if n % 500 == 0:
                    f.flush()
                    elapsed = time.time() - t0
                    vel = n / elapsed if elapsed > 0 else 1
                    eta = (total - n) / vel
                    print(f"  {n}/{total}  {vel:.0f}/s  ETA {fmt_dur(eta)}   ", end="\r")
    print()
    log(f"  Completado. CSV: {CSV_DURACIONES}")

# =============================================================================
# FASE 5: RESUMEN
# =============================================================================
def fase_resumen(dry_run=False):
    log("=" * 70)
    log("FASE 5: Generando tabla resumen de estadisticas")
    log("=" * 70)

    if not CSV_DURACIONES.exists():
        log("  duraciones.csv no encontrado. Ejecuta la fase 4 primero.", "WARN")
        return

    # Indices O(n): una sola pasada re-parseando 'ruta' para obtener valores correctos
    log("  Leyendo y procesando CSV...")
    idx_db_tipo  = defaultdict(lambda: [0, 0.0])
    idx_db_idm   = defaultdict(lambda: [0, 0.0])
    idx_tipo     = defaultdict(lambda: [0, 0.0])
    idx_idm      = defaultdict(lambda: [0, 0.0])
    idx_detalle  = defaultdict(lambda: [0, 0.0])
    total_c, total_d = 0, 0.0

    with open(CSV_DURACIONES, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ruta = row["ruta"]
                # Saltar archivos que estaban en carpetas temporales
                if "_temp" in ruta or "\\_temp\\" in ruta:
                    continue
                dur  = float(row["duracion_seg"])
                db, ver, spl, idm, lbl = _parsear_ruta(ruta)
                if db == "-": continue
                # Normalizar label a los valores conocidos
                if lbl not in ("bonafide", "spoof", "sin_etiqueta", "desconocido"):
                    lbl = "spoof" if db in ("MLAAD",) else lbl
                idx_db_tipo[(db, lbl)][0]          += 1;  idx_db_tipo[(db, lbl)][1]          += dur
                idx_db_idm[(db, idm)][0]           += 1;  idx_db_idm[(db, idm)][1]           += dur
                idx_tipo[lbl][0]                   += 1;  idx_tipo[lbl][1]                   += dur
                idx_idm[idm][0]                    += 1;  idx_idm[idm][1]                    += dur
                idx_detalle[(db,ver,spl,idm,lbl)][0] += 1
                idx_detalle[(db,ver,spl,idm,lbl)][1] += dur
                total_c += 1; total_d += dur
            except (ValueError, KeyError):
                pass
    log(f"  {fmt_num(total_c)} registros procesados")

    dbs     = sorted({k[0] for k in idx_db_tipo})
    idiomas = sorted({k[1] for k in idx_db_idm})
    tipos   = sorted(idx_tipo.keys())
    S = "=" * 80
    s = "-" * 80

    lineas = [
        S,
        "RESUMEN DE BASES DE DATOS - TESIS DETECCION DE AUDIO DEEPFAKE",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        S,
    ]

    # ── 1. POR BASE DE DATOS ─────────────────────────────────────────────────
    lineas += ["", S, "1. METRICAS POR BASE DE DATOS", S]
    for db in dbs:
        db_c = sum(v[0] for k,v in idx_db_tipo.items() if k[0]==db)
        db_d = sum(v[1] for k,v in idx_db_tipo.items() if k[0]==db)
        lineas += ["", f"  [{db}]  {fmt_num(db_c)} archivos   {fmt_dur(db_d)} total"]
        lineas.append(s)

        lineas.append(f"  {'TIPO':<22} {'ARCHIVOS':>12}   {'DURACION':>12}")
        lineas.append(f"  {'-'*22} {'-'*12}   {'-'*12}")
        for tipo in tipos:
            c, d = idx_db_tipo.get((db, tipo), [0, 0.0])
            if c: lineas.append(f"  {tipo:<22} {fmt_num(c):>12}   {fmt_dur(d):>12}")

        lineas.append("")
        lineas.append(f"  {'IDIOMA':<35} {'ARCHIVOS':>12}   {'DURACION':>12}")
        lineas.append(f"  {'-'*35} {'-'*12}   {'-'*12}")
        for idm in idiomas:
            c, d = idx_db_idm.get((db, idm), [0, 0.0])
            if c:
                lineas.append(f"  {nombre_idioma(idm)[:34]:<35} {fmt_num(c):>12}   {fmt_dur(d):>12}")

    # ── 2. TOTAL GENERAL ─────────────────────────────────────────────────────
    lineas += ["", S, "2. TOTAL GENERAL", S]
    lineas.append(f"  Total: {fmt_num(total_c)} archivos   {fmt_dur(total_d)}")
    lineas.append("")

    lineas.append(f"  {'TIPO':<22} {'ARCHIVOS':>12}   {'DURACION':>12}")
    lineas.append(f"  {'-'*22} {'-'*12}   {'-'*12}")
    for tipo in tipos:
        c, d = idx_tipo[tipo]
        if c: lineas.append(f"  {tipo:<22} {fmt_num(c):>12}   {fmt_dur(d):>12}")

    lineas.append("")
    lineas.append(f"  {'IDIOMA':<35} {'ARCHIVOS':>12}   {'DURACION':>12}")
    lineas.append(f"  {'-'*35} {'-'*12}   {'-'*12}")
    for idm in idiomas:
        c, d = idx_idm[idm]
        if c:
            lineas.append(f"  {nombre_idioma(idm)[:34]:<35} {fmt_num(c):>12}   {fmt_dur(d):>12}")

    # ── 3. TABLA DETALLADA ───────────────────────────────────────────────────
    lineas += ["", S, "3. TABLA DETALLADA (DB / VERSION / SPLIT / IDIOMA / TIPO)", S]
    COL = "{:<18}{:<20}{:<14}{:<28}{:<16}{:>10}  {:>12}"
    lineas.append(COL.format("BASE DE DATOS","VERSION/SUB-DB","SPLIT","IDIOMA","TIPO","ARCHIVOS","DURACION"))
    lineas.append(s)
    for clave in sorted(idx_detalle.keys()):
        db, ver, spl, idm, tipo = clave
        cnt, dur = idx_detalle[clave]
        lineas.append(COL.format(db, ver[:19], spl,
            nombre_idioma(idm)[:27], tipo, fmt_num(cnt), fmt_dur(dur)))

    # ── 4. LEYENDA Y NOTAS ───────────────────────────────────────────────────
    lineas += ["", S, "4. LEYENDA DE CODIGOS DE IDIOMA", s]
    for idm in idiomas:
        lineas.append(f"  {idm:<14} = {nombre_idioma(idm)}")

    lineas += [
        "", S, "5. NOTAS IMPORTANTES", s,
        "  [1] ASVspoof2021 LA/PA: Etiquetas bonafide/spoof no publicadas oficialmente.",
        "      Organizados como sin_etiqueta. Ver: https://www.asvspoof.org/",
        "  [2] ASVspoof5: Basado en MLS (8 idiomas). Mapeo speaker->idioma requiere",
        "      metadata MLS adicional: https://www.openslr.org/94/",
        "      Todos los archivos agrupados bajo 'multilingual'.",
        "  [3] CodecFake: Formato Arrow (HuggingFace). Ingles/VCTK. ~707,872 muestras.",
        "      Duraciones no medibles con ffprobe sobre archivos Arrow.",
        "  [4] MLAAD: Solo contiene audio deepfake (spoof). Bonafide -> ver M-AILABS.",
        "  [5] M-AILABS: Solo contiene audio bonafide (real). Deepfake -> ver MLAAD.",
        "  [6] SONAR/generated_audio y LibriSeVoc: clasificados como spoof.",
        "      SONAR/In_the_Wild y SONAR_core: incluyen bonafide y spoof segun meta.csv.",
        "      SONAR/LJSpeech: clasificado como bonafide.",
        S,
    ]

    texto = "\n".join(lineas)
    print("\n" + texto)
    RESUMEN_TXT.write_text(texto, encoding="utf-8")
    log(f"  Resumen guardado en: {RESUMEN_TXT}")
# =============================================================================
# MAIN
# =============================================================================
FASES_MAP = {
    "1": ("setup",   fase_setup),
    "2": ("mover",   fase_mover),
    "3": ("extraer", fase_extraer),
    "4": ("medir",   fase_medir),
    "5": ("resumen", fase_resumen),
}

def main():
    parser = argparse.ArgumentParser(
        description="Organizacion de datasets deepfake audio para tesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python organizar_datasets.py                 # todas las fases\n"
            "  python organizar_datasets.py --fases 3,4,5   # extraer+medir+resumir\n"
            "  python organizar_datasets.py --fases 5       # solo tabla final\n"
            "  python organizar_datasets.py --dry-run       # simula sin cambios\n"
        )
    )
    parser.add_argument("--fases", default="1,2,3,4,5",
                        help="Fases a ejecutar separadas por coma (default: 1,2,3,4,5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula sin realizar cambios en disco")
    args = parser.parse_args()

    selec = [f.strip() for f in args.fases.split(",")]
    inv   = [f for f in selec if f not in FASES_MAP]
    if inv:
        print(f"Fases invalidas: {inv}. Validas: {list(FASES_MAP.keys())}")
        sys.exit(1)

    if args.dry_run:
        log("*** MODO DRY-RUN: no se realizaran cambios en disco ***")

    log(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Fases: {selec}  |  dry-run: {args.dry_run}")
    t0 = time.time()

    for num in selec:
        nombre, func = FASES_MAP[num]
        try:
            func(args.dry_run)
        except KeyboardInterrupt:
            log("Interrumpido. El progreso esta guardado en duraciones.csv", "WARN")
            sys.exit(0)
        except Exception as e:
            log(f"ERROR en fase {num} ({nombre}): {e}", "ERROR")
            import traceback; traceback.print_exc()

    log(f"Tiempo total: {fmt_dur(time.time() - t0)}")


if __name__ == "__main__":
    main()
