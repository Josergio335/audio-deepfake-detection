"""
filtrar_manifests.py
Actualiza los manifests de Exp1, Exp2 y Exp3 eliminando entradas
cuyos archivos procesados no existen en D:/Bases de Datos procesadas/.

Esto cubre dos casos:
    - Archivos vacios tras VAD (menos de 0.5s de voz)
    - Cualquier otro archivo que haya fallado en el preprocesamiento

Los manifests originales se renombran con sufijo _original como respaldo.

Uso:
    python filtrar_manifests.py --dry-run   # muestra cuantos se excluirian
    python filtrar_manifests.py             # actualiza los manifests
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict

PARTICIONES_DIR = Path(r"D:\Particiones")
INPUT_BASE      = Path(r"D:\Bases de Datos extraidas")
OUTPUT_BASE     = Path(r"D:\Bases de Datos procesadas")
EXPERIMENTOS    = ["Exp1", "Exp2", "Exp3"]

def ruta_procesada(ruta_original):
    p = Path(ruta_original)
    try:
        rel = p.relative_to(INPUT_BASE)
    except ValueError:
        rel = Path(p.name)
    return OUTPUT_BASE / rel.with_suffix(".wav")

def filtrar_manifest(exp, dry_run):
    manifest_path = PARTICIONES_DIR / exp / "manifest.csv"
    if not manifest_path.exists():
        print(f"  [{exp}] manifest no encontrado, omitiendo")
        return

    filas_ok = []
    excluidos = defaultdict(int)

    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            procesado = ruta_procesada(row["ruta"])
            if procesado.exists():
                filas_ok.append(row)
            else:
                excluidos[row["db"]] += 1

    total_orig = len(filas_ok) + sum(excluidos.values())
    print(f"\n  [{exp}]")
    print(f"    Original  : {total_orig:,} archivos")
    print(f"    Excluidos : {sum(excluidos.values()):,}")
    for db, n in sorted(excluidos.items()):
        print(f"      {db}: {n:,}")
    print(f"    Resultante: {len(filas_ok):,} archivos")

    if dry_run:
        return

    # Respaldar original
    backup = manifest_path.with_suffix(".csv_original")
    if not backup.exists():
        manifest_path.rename(backup)
    else:
        manifest_path.unlink()

    # Escribir manifest filtrado
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filas_ok)

    print(f"    Escrito: {manifest_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("  FILTRADO DE MANIFESTS — excluir archivos sin procesar")
    print("=" * 62)
    if args.dry_run:
        print("  *** DRY-RUN: no se modificaran los manifests ***")

    for exp in EXPERIMENTOS:
        filtrar_manifest(exp, args.dry_run)

    print("\nFiltrado completado.")

if __name__ == "__main__":
    main()
