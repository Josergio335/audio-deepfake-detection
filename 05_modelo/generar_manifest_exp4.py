#!/usr/bin/env python3
"""
Genera el manifest para Experimento 4 (Exp-4):

Estrategia:
  - 7 idiomas originales (de, en, es, fr, it, pl, ru): tope de 1800 archivos por idioma y clase
  - 9 idiomas nuevos de CommonVoice (bonafide) + MLAAD (spoof): tope de 2400 archivos por idioma y clase
  - Los topes son por numero de archivos, como aproximacion a ~3 h; no se mide la duracion
    (duracion_seg queda en 0 para los idiomas nuevos)
  - Extiende ~/embeddings/indice.csv con las entradas nuevas
  - Genera indice_test_zeroshot_exp4.csv (zero-shot sin los 9 idiomas movidos a train)

Idiomas nuevos en train (9): ja, ko, hi, id, ar, he, bg, cs, fi
  → tipológicamente diversos: asiático-este, sur-asiático, austronesio, semítico, eslavo, uralico

Idiomas que permanecen como zero-shot (9 CV): ro, nl, lt, hu, et, fa, da, el, pt
"""

import csv
import random
import sys
from pathlib import Path
from collections import defaultdict

random.seed(42)

HOME = Path.home()

# Rutas de entrada
EXP3_MANIFEST    = HOME / "particiones/Exp3/manifest.csv"
INDICE_TRAIN     = HOME / "embeddings/indice.csv"
INDICE_ZEROSHOT  = HOME / "embeddings_test/indice_test_zeroshot_v2.csv"

# Rutas de salida
OUT_DIR          = HOME / "particiones/Exp4"
OUT_MANIFEST     = OUT_DIR / "manifest.csv"
OUT_ZEROSHOT_EXP4 = HOME / "embeddings_test/indice_test_zeroshot_exp4.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuración de idiomas ──────────────────────────────────────────────────
IDIOMAS_ORIG   = {"de", "en", "es", "fr", "it", "pl", "ru"}
IDIOMAS_NUEVOS = {"ja", "ko", "hi", "id", "ar", "he", "bg", "cs", "fi"}

# Topes por clase (como proxy de 3h)
# M-AILABS avg ~6s/archivo  → 3h ≈ 1800 archivos
# CommonVoice avg ~4-5s     → 3h ≈ 2400 archivos
# MLAAD spoof avg ~4-5s     → 3h ≈ 2400 archivos
TOPE_ORIG  = 1800
TOPE_NUEVO = 2400

# ── 1. Leer Exp3 manifest, filtrar idiomas originales con tope ────────────────
print("=" * 60)
print("Paso 1: Filtrando idiomas originales del manifest Exp3")
print("=" * 60)

grupos = defaultdict(list)
total_exp3 = 0
with open(EXP3_MANIFEST, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        total_exp3 += 1
        if row["idioma"] in IDIOMAS_ORIG:
            grupos[(row["idioma"], row["label"])].append(row)

filas_exp4 = []
for (idioma, label), filas in sorted(grupos.items()):
    random.shuffle(filas)
    sel = filas[:TOPE_ORIG]
    filas_exp4.extend(sel)
    print(f"  {idioma:3s} / {label:8s}: {len(sel):5d} seleccionados de {len(filas):6d}")

print(f"\n  Subtotal idiomas originales: {len(filas_exp4)} archivos")

# ── 2. Leer zeroshot_v2, extraer idiomas nuevos ───────────────────────────────
print("\n" + "=" * 60)
print("Paso 2: Extrayendo idiomas nuevos de zeroshot_v2")
print("=" * 60)

grupos_nuevos = defaultdict(list)
all_zeroshot_rows = []

with open(INDICE_ZEROSHOT, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames_zeroshot = reader.fieldnames
    for row in reader:
        all_zeroshot_rows.append(row)
        lang = row.get("idioma", "")
        if lang in IDIOMAS_NUEVOS:
            grupos_nuevos[(lang, row["label"])].append(row)

nuevas_filas_indice = []
n_nuevos = 0

# Verificar qué idiomas/labels están disponibles
print("  Disponibles en zeroshot_v2 para idiomas nuevos:")
for (idioma, label), filas in sorted(grupos_nuevos.items()):
    random.shuffle(filas)
    sel = filas[:TOPE_NUEVO]
    print(f"  {idioma:3s} / {label:8s}: {len(sel):5d} seleccionados de {len(filas):6d}")
    n_nuevos += len(sel)

    for row in sel:
        # Entrada para el manifest Exp4
        filas_exp4.append({
            "ruta": row["ruta_original"],
            "db": row["db"],
            "idioma": row["idioma"],
            "label": row["label"],
            "duracion_seg": "0"
        })
        # Entrada para extender indice.csv
        nuevas_filas_indice.append({
            "ruta_original": row["ruta_original"],
            "hash": row["hash"],
            "wav2vec2bert_pt": row["wav2vec2bert_pt"],
            "wavlm_pt": row["wavlm_pt"],
            "whisper_pt": row["whisper_pt"]
        })

print(f"\n  Subtotal idiomas nuevos: {n_nuevos} archivos")
print(f"\n  TOTAL Exp4 manifest: {len(filas_exp4)} archivos")

# ── 3. Escribir manifest Exp4 ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Paso 3: Escribiendo manifest Exp4")
print("=" * 60)

with open(OUT_MANIFEST, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["ruta", "db", "idioma", "label", "duracion_seg"])
    writer.writeheader()
    writer.writerows(filas_exp4)
print(f"  Escrito: {OUT_MANIFEST}")
print(f"  Total filas: {len(filas_exp4)}")

# ── 4. Verificar duplicados antes de extender indice.csv ─────────────────────
print("\n" + "=" * 60)
print("Paso 4: Extendiendo indice.csv con entradas nuevas")
print("=" * 60)

# Leer hashes existentes para evitar duplicados
print("  Leyendo hashes existentes en indice.csv...")
hashes_existentes = set()
with open(INDICE_TRAIN, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        hashes_existentes.add(row["hash"])
print(f"  {len(hashes_existentes)} entradas existentes")

# Filtrar solo entradas verdaderamente nuevas
nuevas_sin_dup = [r for r in nuevas_filas_indice if r["hash"] not in hashes_existentes]
duplicados = len(nuevas_filas_indice) - len(nuevas_sin_dup)
print(f"  {len(nuevas_filas_indice)} nuevas → {duplicados} duplicados → {len(nuevas_sin_dup)} a añadir")

with open(INDICE_TRAIN, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["ruta_original", "hash", "wav2vec2bert_pt", "wavlm_pt", "whisper_pt"])
    writer.writerows(nuevas_sin_dup)
print(f"  Indice extendido: {len(nuevas_sin_dup)} entradas añadidas")

# ── 5. Generar zeroshot_exp4.csv ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Paso 5: Generando indice_test_zeroshot_exp4.csv")
print("=" * 60)

# Excluir los hashes que ya se usaron en train (para evitar data leakage)
hashes_en_train = {r["hash"] for r in nuevas_filas_indice}

count_zs = 0
count_excl = 0
with open(OUT_ZEROSHOT_EXP4, "w", encoding="utf-8", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames_zeroshot)
    writer.writeheader()
    for row in all_zeroshot_rows:
        if row.get("idioma", "") not in IDIOMAS_NUEVOS:
            writer.writerow(row)
            count_zs += 1
        else:
            count_excl += 1

print(f"  Excluidos (movidos a train): {count_excl} archivos")
print(f"  Zero-shot Exp4: {count_zs} archivos")
print(f"  Escrito: {OUT_ZEROSHOT_EXP4}")

# ── 6. Resumen final ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"  Manifest Exp4:         {OUT_MANIFEST}")
print(f"  Zero-shot Exp4 index:  {OUT_ZEROSHOT_EXP4}")
print(f"  Entradas nuevas en indice.csv: {len(nuevas_sin_dup)}")
print()
print("Próximos pasos:")
print("  1. Editar train.py: añadir en MANIFEST_DIRS la clave 4:")
print("        4: Path.home() / 'particiones/Exp4',")
print("  2. Editar evaluate.py: añadir 'zeroshot_exp4' a choices (opcional)")
print("  3. Entrenar: python train.py --exp 4")
print("  4. Evaluar: python evaluate.py --escenario zeroshot_exp4 --modelo <checkpoint>")
