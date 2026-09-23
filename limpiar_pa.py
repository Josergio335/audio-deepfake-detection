"""
limpiar_pa.py
Elimina de D:/Bases de Datos procesadas/ los archivos que provienen
del protocolo PA de ASVspoof (replay attacks), que no deben estar
en el conjunto de entrenamiento.

Uso:
    python limpiar_pa.py --dry-run   # muestra cuantos archivos borraria
    python limpiar_pa.py             # elimina los archivos
"""

import argparse
from pathlib import Path

OUTPUT_BASE = Path(r"D:\Bases de Datos procesadas")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra cuantos archivos borraria sin eliminar nada")
    args = parser.parse_args()

    print("Buscando archivos PA en D:\\Bases de Datos procesadas\\ASVspoof ...")

    asvspoof_dir = OUTPUT_BASE / "ASVspoof"
    if not asvspoof_dir.exists():
        print("No se encontro la carpeta ASVspoof en el directorio procesado.")
        return

    pa_files = list(asvspoof_dir.rglob("*"))
    pa_files = [f for f in pa_files if f.is_file() and "\\PA\\" in str(f)]

    print(f"  Archivos PA encontrados: {len(pa_files):,}")

    if args.dry_run or not pa_files:
        if not pa_files:
            print("No hay archivos PA que eliminar.")
        return

    eliminados = 0
    for f in pa_files:
        try:
            f.unlink()
            eliminados += 1
        except Exception as e:
            print(f"  Error eliminando {f}: {e}")

    # Eliminar carpetas PA vacias
    for d in sorted(asvspoof_dir.rglob("*"), reverse=True):
        if d.is_dir() and "\\PA\\" in str(d) or d.name == "PA":
            try:
                d.rmdir()
            except OSError:
                pass  # no esta vacia, ignorar

    print(f"  Eliminados: {eliminados:,} archivos")

if __name__ == "__main__":
    main()
