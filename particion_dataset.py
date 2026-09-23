"""
particion_dataset.py
Genera los manifests de entrenamiento para Exp-1, Exp-2 y Exp-3.

Diseno:
    Exp-1 subset Exp-2 subset Exp-3  (subconjuntos anidados)
    Se selecciona Exp-3 primero; Exp-1 y Exp-2 se derivan de el.

    Cap por (idioma, clase):
        Exp-1: 18h  -- bottleneck: ruso spoof (~18h)
        Exp-2: 32h  -- bottleneck: polaco bonafide (~32h)
        Exp-3: 46h  -- bottleneck: italiano spoof (~46h)

    7 idiomas: en, de, fr, es, it, pl, ru
    Datasets de entrenamiento: ASVspoof (split=train), M-AILABS, MLAAD, CodecFake

Salida:
    D:/Particiones/Exp1/manifest.csv
    D:/Particiones/Exp2/manifest.csv
    D:/Particiones/Exp3/manifest.csv
    D:/Particiones/resumen_particiones.txt

Cada manifest tiene columnas: ruta, db, idioma, label, duracion_seg

Uso:
    python particion_dataset.py            # genera los 3 manifests
    python particion_dataset.py --dry-run  # muestra resumen sin escribir archivos
"""

import csv
import random
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

CSV_DURACIONES = Path(r"D:\Scripts Extraccion\duraciones.csv")
OUTPUT_DIR     = Path(r"D:\Particiones")

SEED = 42

IDIOMAS_CORE = {"en", "de", "fr", "es", "it", "pl", "ru"}

TRAIN_DBS = {"ASVspoof", "M-AILABS", "MLAAD", "CodecFake"}

CAPS = {
    "Exp1": 18.0,
    "Exp2": 32.0,
    "Exp3": 46.0,
}

# Mapeo de codigos de idioma de M-AILABS a ISO-639-1
MAILABS_LANG = {
    "de_DE": "de",
    "fr_FR": "fr",
    "es_ES": "es",
    "en_US": "en",
    "en_UK": "en",
    "it_IT": "it",
    "pl_PL": "pl",
    "ru_RU": "ru",
}

# ---------------------------------------------------------------------------
# Parseo de ruta (independiente, sin importar organizar_datasets)
# ---------------------------------------------------------------------------

def parsear_ruta(ruta):
    """
    Extrae (db, split, idioma, label) directamente desde la ruta del archivo.
    Devuelve None si la ruta no puede ser clasificada.

    Estructuras conocidas:
        ASVspoof:  .../ASVspoof/ASVspoof2019/LA/{split}/{idioma}/{label}/file
        CodecFake: .../CodecFake/train/{genuine|spoof}/file
        MLAAD:     .../MLAAD/{real|fake}/{lang}/{tts}/file
        M-AILABS:  .../M-AILABS/{lang_code}/by_book/.../file  (siempre bonafide)
    """
    parts = ruta.replace("/", "\\").split("\\")

    if "ASVspoof" in parts:
        i = parts.index("ASVspoof")
        # partes: [ASVspoof, version, protocol(LA/PA), split, idioma, label, file]
        if len(parts) < i + 7:
            return None
        protocol = parts[i + 2]  # LA o PA
        if protocol != "LA":     # solo protocolo LA (voz sintetica); PA = replay attacks
            return None
        split  = parts[i + 3]   # train / dev / eval
        idioma = parts[i + 4]   # en
        label  = parts[i + 5]   # bonafide / spoof
        if label not in ("bonafide", "spoof"):
            return None
        return ("ASVspoof", split, idioma, label)

    if "CodecFake" in parts:
        i = parts.index("CodecFake")
        # partes: [CodecFake, train, genuine|spoof, file]
        if len(parts) < i + 4:
            return None
        split  = parts[i + 1]   # train
        folder = parts[i + 2]   # genuine / spoof
        label  = "bonafide" if folder == "genuine" else "spoof"
        return ("CodecFake", split, "en", label)

    if "MLAAD" in parts:
        i = parts.index("MLAAD")
        # partes: [MLAAD, real|fake, lang, tts_model, file]
        if len(parts) < i + 4:
            return None
        carpeta = parts[i + 1]  # real / fake
        idioma  = parts[i + 2]  # en, de, fr, es, it, pl, ru, ...
        label   = "bonafide" if carpeta == "real" else "spoof"
        return ("MLAAD", "train", idioma, label)

    for j, p in enumerate(parts):
        if "M-AILABS" in p:
            # partes: [..., M-AILABS, lang_code, by_book, ...]
            if j + 1 >= len(parts):
                return None
            lang_code = parts[j + 1]
            idioma = MAILABS_LANG.get(lang_code)
            if idioma is None:
                return None
            return ("M-AILABS", "train", idioma, "bonafide")

    return None


# ---------------------------------------------------------------------------
# Leer y filtrar candidatos
# ---------------------------------------------------------------------------

def leer_candidatos():
    print(f"Leyendo {CSV_DURACIONES} ...")
    candidatos = []
    omitidos = defaultdict(int)

    with open(CSV_DURACIONES, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dur = float(row["duracion_seg"])
            if dur <= 0:
                omitidos["duracion_cero"] += 1
                continue

            db_csv = row["db"].strip()
            if db_csv not in TRAIN_DBS:
                omitidos["db_excluida"] += 1
                continue

            parsed = parsear_ruta(row["ruta"])
            if parsed is None:
                omitidos["ruta_no_reconocida"] += 1
                continue

            db, split, idioma, label = parsed

            # ASVspoof: solo split train
            if db == "ASVspoof" and split != "train":
                omitidos["asvspoof_no_train"] += 1
                continue

            # Solo idiomas core
            if idioma not in IDIOMAS_CORE:
                omitidos[f"idioma_excluido:{idioma}"] += 1
                continue

            candidatos.append({
                "ruta":    row["ruta"],
                "db":      db,
                "idioma":  idioma,
                "label":   label,
                "dur_seg": dur,
            })

    print(f"  {len(candidatos):,} archivos candidatos")
    if omitidos:
        for razon, n in sorted(omitidos.items()):
            print(f"  omitidos [{razon}]: {n:,}")
    return candidatos


# ---------------------------------------------------------------------------
# Agrupar y mezclar con semilla fija
# ---------------------------------------------------------------------------

def agrupar_y_mezclar(candidatos):
    grupos = defaultdict(list)
    for c in candidatos:
        grupos[(c["idioma"], c["label"])].append(c)

    rng = random.Random(SEED)
    for key in grupos:
        rng.shuffle(grupos[key])

    return grupos


# ---------------------------------------------------------------------------
# Seleccionar hasta el cap (para Exp-3)
# ---------------------------------------------------------------------------

def seleccionar(grupos, cap_horas):
    seleccionados = []
    cap_seg = cap_horas * 3600
    for (lang, lbl) in sorted(grupos):
        acum = 0.0
        for a in grupos[(lang, lbl)]:
            if acum >= cap_seg:
                break
            seleccionados.append(a)
            acum += a["dur_seg"]
    return seleccionados


# ---------------------------------------------------------------------------
# Derivar subconjunto anidado preservando orden de Exp-3
# ---------------------------------------------------------------------------

def subconjunto(archivos_base, cap_horas):
    acum = defaultdict(float)
    cap_seg = cap_horas * 3600
    sub = []
    for a in archivos_base:
        key = (a["idioma"], a["label"])
        if acum[key] < cap_seg:
            sub.append(a)
            acum[key] += a["dur_seg"]
    return sub


# ---------------------------------------------------------------------------
# Calcular resumen textual
# ---------------------------------------------------------------------------

def calcular_resumen(archivos, nombre, cap):
    por_idioma_label = defaultdict(lambda: [0, 0.0])
    por_db           = defaultdict(lambda: [0, 0.0])

    for a in archivos:
        k = (a["idioma"], a["label"])
        por_idioma_label[k][0] += 1
        por_idioma_label[k][1] += a["dur_seg"] / 3600
        por_db[a["db"]][0]     += 1
        por_db[a["db"]][1]     += a["dur_seg"] / 3600

    total_h = sum(a["dur_seg"] for a in archivos) / 3600

    lines = []
    lines.append(f"\n{'='*62}")
    lines.append(f"  {nombre}  (cap: {cap}h por idioma/clase)")
    lines.append(f"{'='*62}")
    lines.append(f"  Total archivos : {len(archivos):,}")
    lines.append(f"  Total horas    : {total_h:.1f}h")

    lines.append(f"\n  Por idioma y clase:")
    lines.append(f"  {'Idioma':<8} {'Clase':<10} {'Archivos':>10} {'Horas':>8}")
    lines.append(f"  {'-'*40}")
    for (lang, lbl) in sorted(por_idioma_label):
        n, h = por_idioma_label[(lang, lbl)]
        lines.append(f"  {lang:<8} {lbl:<10} {n:>10,} {h:>8.1f}h")

    lines.append(f"\n  Por dataset:")
    lines.append(f"  {'Dataset':<15} {'Archivos':>10} {'Horas':>8}")
    lines.append(f"  {'-'*36}")
    for db in sorted(por_db):
        n, h = por_db[db]
        lines.append(f"  {db:<15} {n:>10,} {h:>8.1f}h")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Escribir manifest CSV
# ---------------------------------------------------------------------------

def escribir_manifest(archivos, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ruta", "db", "idioma", "label", "duracion_seg"]
        )
        writer.writeheader()
        for a in archivos:
            writer.writerow({
                "ruta":         a["ruta"],
                "db":           a["db"],
                "idioma":       a["idioma"],
                "label":        a["label"],
                "duracion_seg": f"{a['dur_seg']:.3f}",
            })
    print(f"  Escrito: {path}  ({len(archivos):,} archivos)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genera manifests de entrenamiento para Exp-1, Exp-2, Exp-3"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Muestra resumen sin escribir archivos"
    )
    args = parser.parse_args()

    print("=" * 62)
    print("  PARTICION DE DATASET  --  Exp-1 / Exp-2 / Exp-3")
    print("=" * 62)
    if args.dry_run:
        print("  *** DRY-RUN: no se escribiran archivos ***")
    print(f"  Semilla: {SEED}")
    print()

    candidatos = leer_candidatos()
    if not candidatos:
        print("ERROR: No se encontraron candidatos. Revisa CSV_DURACIONES y los filtros.")
        return

    grupos = agrupar_y_mezclar(candidatos)

    print(f"\nGenerando Exp-3 (cap={CAPS['Exp3']}h/clase) ...")
    exp3 = seleccionar(grupos, CAPS["Exp3"])

    print(f"Generando Exp-2 (cap={CAPS['Exp2']}h/clase, subconjunto de Exp-3) ...")
    exp2 = subconjunto(exp3, CAPS["Exp2"])

    print(f"Generando Exp-1 (cap={CAPS['Exp1']}h/clase, subconjunto de Exp-2) ...")
    exp1 = subconjunto(exp2, CAPS["Exp1"])

    resumen  = calcular_resumen(exp1, "Exp-1  -- bottleneck: ruso spoof",     CAPS["Exp1"])
    resumen += calcular_resumen(exp2, "Exp-2  -- bottleneck: polaco bonafide", CAPS["Exp2"])
    resumen += calcular_resumen(exp3, "Exp-3  -- bottleneck: italiano spoof",  CAPS["Exp3"])
    resumen += f"\n\nGenerado : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    resumen += f"Semilla  : {SEED}\n"

    print(resumen)

    if not args.dry_run:
        escribir_manifest(exp1, OUTPUT_DIR / "Exp1" / "manifest.csv")
        escribir_manifest(exp2, OUTPUT_DIR / "Exp2" / "manifest.csv")
        escribir_manifest(exp3, OUTPUT_DIR / "Exp3" / "manifest.csv")

        resumen_path = OUTPUT_DIR / "resumen_particiones.txt"
        resumen_path.write_text(resumen, encoding="utf-8")
        print(f"  Resumen: {resumen_path}")

    print("\nParticion completada.")


if __name__ == "__main__":
    main()
