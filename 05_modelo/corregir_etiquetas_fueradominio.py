"""
corregir_etiquetas_fueradominio.py
Corrige in situ las etiquetas de SONAR (LibriSeVoc e In_the_Wild) en el indice de
test del escenario fuera de dominio.

Contexto: la version anterior de generar_manifests_test.py marcaba como bonafide todo
LibriSeVoc e In_the_Wild, incluidos los vocoders (diffwave, melgan, parallel_wave_gan,
wavegrad, wavenet, wavernn) y la carpeta spoof. Este script aplica sobre el indice ya
extraido (~/embeddings_test/indice_test_fueradominio.csv) la misma regla que ahora usa
generar_manifests_test.etiqueta_sonar, sin reextraer embeddings.

Deja un respaldo del indice original y es idempotente: si ya esta corregido, no cambia nada.

Uso:
    python corregir_etiquetas_fueradominio.py
"""

import csv
import shutil
from pathlib import Path

INDICE = Path.home() / "embeddings_test" / "indice_test_fueradominio.csv"
RESPALDO = INDICE.with_name(INDICE.name + ".bak_etiquetas_erroneas")

BONAFIDE = {"gt", "wavs", "bona-fide", "bonafide"}
SUBCONJUNTOS_AFECTADOS = {"LibriSeVoc", "In_the_Wild"}


def main():
    if not RESPALDO.exists():
        shutil.copy2(INDICE, RESPALDO)
        print(f"Respaldo creado: {RESPALDO}")
    else:
        print(f"Respaldo ya existente (no se sobrescribe): {RESPALDO}")

    with open(INDICE, encoding="utf-8") as f:
        lector = csv.DictReader(f)
        campos = lector.fieldnames
        filas = list(lector)

    cambios = 0
    for r in filas:
        partes = set(r["ruta_original"].replace("\\", "/").split("/"))
        if r["db"].upper() == "SONAR" and partes & SUBCONJUNTOS_AFECTADOS:
            nuevo = "bonafide" if partes & BONAFIDE else "spoof"
            if nuevo != r["label"]:
                r["label"] = nuevo
                cambios += 1

    with open(INDICE, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)

    print(f"Etiquetas corregidas: {cambios:,}  (esperado en la primera ejecucion: 7,763)")


if __name__ == "__main__":
    main()
