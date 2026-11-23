"""
Añade filas bonafide de Mozilla Common Voice al manifest de zero-shot existente.

NO modifica las filas de spoof (MLAAD) ya presentes.
Solo añade idiomas que no están en los 7 de entrenamiento.
Tope: MAX_POR_IDIOMA clips por idioma (semilla 42 para reproducibilidad).

Uso:
    python añadir_bonafide_zeroshot.py               # preview de cuánto añadiría
    python añadir_bonafide_zeroshot.py --aplicar     # escribe el manifest actualizado
    python añadir_bonafide_zeroshot.py --aplicar --tope 1000  # tope distinto
"""

import argparse
import csv
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
MANIFEST_ZEROSHOT = Path("D:/Particiones/manifest_test_zeroshot.csv")
COMMONVOICE_DIR   = Path("D:/Bases de Datos extraidas/CommonVoice")
MAX_POR_IDIOMA    = 2400
MIN_POR_IDIOMA    = 100   # excluir idiomas con menos clips que esto
SEMILLA           = 42

# Idiomas de entrenamiento — no se añaden como zero-shot
IDIOMAS_TRAIN = {"en", "de", "fr", "es", "it", "pl", "ru"}

# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aplicar", action="store_true",
                        help="Escribe el manifest actualizado (sin esto solo muestra preview)")
    parser.add_argument("--tope", type=int, default=MAX_POR_IDIOMA,
                        help=f"Clips máximos por idioma (default: {MAX_POR_IDIOMA})")
    args = parser.parse_args()

    rng = random.Random(SEMILLA)

    # 1. Leer manifest existente
    with open(MANIFEST_ZEROSHOT, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        filas_existentes = list(reader)
        campos = reader.fieldnames

    idiomas_ya_en_manifest = {r["idioma"] for r in filas_existentes}
    print(f"Manifest actual: {len(filas_existentes):,} filas, {len(idiomas_ya_en_manifest)} idiomas")

    # 2. Recorrer carpetas de Common Voice
    nuevas_filas = []
    resumen = []

    for lang_dir in sorted(COMMONVOICE_DIR.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name

        # Saltar idiomas de entrenamiento
        if lang in IDIOMAS_TRAIN:
            print(f"  [{lang}] IGNORADO (idioma de entrenamiento)")
            continue

        # Buscar clips (pueden estar en clips/ o audios/)
        clips_dir = lang_dir / "clips"
        if not clips_dir.exists():
            clips_dir = lang_dir / "audios"
        if not clips_dir.exists():
            print(f"  [{lang}] Sin carpeta de clips, saltando")
            continue

        archivos = sorted(clips_dir.glob("*.mp3"))
        if not archivos:
            print(f"  [{lang}] 0 clips, saltando")
            continue

        # Comprobar si ya hay bonafide de este idioma en el manifest
        bonafide_existente = sum(
            1 for r in filas_existentes
            if r["idioma"] == lang and r["label"] == "bonafide"
        )
        if bonafide_existente > 0:
            print(f"  [{lang}] Ya tiene {bonafide_existente} bonafide en el manifest, saltando")
            continue

        # Excluir idiomas con muy pocos clips
        if len(archivos) < MIN_POR_IDIOMA:
            print(f"  [{lang}] Solo {len(archivos)} clips — insuficiente, excluyendo")
            continue

        # Muestrear hasta tope
        muestra = archivos if len(archivos) <= args.tope else rng.sample(archivos, args.tope)

        for ruta in muestra:
            nuevas_filas.append({
                "ruta":     str(ruta),
                "db":       "CommonVoice",
                "idioma":   lang,
                "label":    "bonafide",
                "escenario": "zeroshot",
            })

        resumen.append((lang, len(archivos), len(muestra)))
        print(f"  [{lang}] {len(archivos):,} disponibles → {len(muestra):,} añadidos")

    print(f"\nTotal nuevas filas bonafide: {len(nuevas_filas):,}")
    print(f"Idiomas nuevos: {len(resumen)}")

    if not args.aplicar:
        print("\n[DRY-RUN] Usa --aplicar para escribir el manifest actualizado.")
        return

    if not nuevas_filas:
        print("Nada que añadir.")
        return

    # 3. Escribir manifest actualizado (filas existentes + nuevas al final)
    todas = filas_existentes + nuevas_filas
    with open(MANIFEST_ZEROSHOT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(todas)

    print(f"\nManifest actualizado: {len(todas):,} filas totales")
    print(f"Guardado en: {MANIFEST_ZEROSHOT}")


if __name__ == "__main__":
    main()
