"""
generar_manifests_test.py
Genera los 3 manifests de prueba para la evaluación del modelo.

Escenarios:
    1. En distribución   → ASVspoof 2019 LA dev + eval  (~20,000 archivos)
    2. Zero-shot         → MLAAD 33 idiomas no vistos   (~80,000 archivos, ~2,400/idioma)
    3. Fuera de dominio  → SONAR + SpoofCeleb           (~31,500 archivos)

Uso:
    python generar_manifests_test.py
    python generar_manifests_test.py --base D:/Bases de Datos procesadas

Salida:
    SALIDA_DIR/manifest_test_indistribucion.csv
    SALIDA_DIR/manifest_test_zeroshot.csv
    SALIDA_DIR/manifest_test_fueradominio.csv
"""

import argparse
import csv
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# AJUSTAR según entorno
BASE_PROCESADA = Path(r"D:\Bases de Datos procesadas")  # raíz de los audios procesados (MLAAD zero-shot)
BASE_EXTRAIDA  = Path(r"D:\Bases de Datos extraidas")   # raíz de los audios originales (ASVspoof dev/eval, SONAR, SpoofCeleb)
SALIDA_DIR     = Path(r"D:\Particiones")                 # dónde se guardan los manifests de test

SEMILLA        = 42
MAX_ZEROSHOT_POR_IDIOMA = 2400   # máximo de archivos por idioma en zero-shot
MAX_FUERADOMINIO        = 31500  # máximo total para SONAR + SpoofCeleb

# Idiomas usados en entrenamiento (excluir del zero-shot)
IDIOMAS_TRAIN = {"en", "de", "fr", "es", "it", "pl", "ru"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def listar_wavs(directorio: Path) -> list[Path]:
    """Lista recursivamente todos los .wav y .flac bajo un directorio."""
    if not directorio.exists():
        return []
    return list(directorio.rglob("*.wav")) + list(directorio.rglob("*.flac"))


def ruta_a_fila(ruta: Path, db: str, idioma: str, label: str, escenario: str) -> dict:
    return {
        "ruta":      str(ruta),
        "db":        db,
        "idioma":    idioma,
        "label":     label,
        "escenario": escenario,
    }


def escribir_manifest(filas: list[dict], ruta_salida: Path):
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ruta", "db", "idioma", "label", "escenario"])
        writer.writeheader()
        writer.writerows(filas)
    print(f"  -> {ruta_salida.name}: {len(filas):,} archivos")


# ---------------------------------------------------------------------------
# Escenario 1: En distribución — ASVspoof 2019 LA dev + eval
# ---------------------------------------------------------------------------

def generar_indistribucion(base: Path) -> list[dict]:
    """
    ASVspoof 2019 LA dev + eval desde Bases de Datos extraidas.
    Estructura: base/ASVspoof/ASVspoof2019/LA/{dev,eval}/en/{bonafide,spoof}/
    Se excluye la carpeta 'desconocido' (838 archivos sin etiqueta).
    """
    filas = []
    raiz = base / "ASVspoof" / "ASVspoof2019" / "LA"

    for split in ["dev", "eval"]:
        for label in ["bonafide", "spoof"]:
            carpeta = raiz / split / "en" / label
            for wav in listar_wavs(carpeta):
                filas.append(ruta_a_fila(wav, "ASVspoof2019", "en", label, "indistribucion"))

    print(f"  En distribución: {len(filas):,} archivos encontrados")
    return filas


# ---------------------------------------------------------------------------
# Escenario 2: Zero-shot — MLAAD idiomas no vistos
# ---------------------------------------------------------------------------

def generar_zeroshot(base: Path, rng: random.Random) -> list[dict]:
    """
    Spoof: MLAAD idiomas no vistos (base/MLAAD/fake/{idioma}/), máx MAX_ZEROSHOT_POR_IDIOMA por idioma.
    Bonafide: M-AILABS (BASE_PROCESADA/M-AILABS/), muestreado para igualar el total de spoof.
    Nota: el bonafide es de idiomas de entrenamiento — se documenta en la tesis.
    """
    filas_spoof = []
    raiz_mlaad = base / "MLAAD" / "fake"

    if not raiz_mlaad.exists():
        print(f"  ADVERTENCIA: no se encontró {raiz_mlaad}")
        return filas_spoof

    for carpeta_idioma in sorted(raiz_mlaad.iterdir()):
        if not carpeta_idioma.is_dir():
            continue
        idioma = carpeta_idioma.name
        if idioma in IDIOMAS_TRAIN:
            continue

        wavs = listar_wavs(carpeta_idioma)
        if not wavs:
            continue

        muestra = rng.sample(wavs, min(len(wavs), MAX_ZEROSHOT_POR_IDIOMA))
        for wav in muestra:
            filas_spoof.append(ruta_a_fila(wav, "MLAAD", idioma, "spoof", "zeroshot"))

    idiomas_unicos = len({f["idioma"] for f in filas_spoof})
    print(f"  Zero-shot spoof : {len(filas_spoof):,} archivos, {idiomas_unicos} idiomas")

    # Bonafide de M-AILABS para balancear (mismo número que spoof)
    filas_bonafide = []
    raiz_mailabs = base / "M-AILABS"
    if raiz_mailabs.exists():
        todos_bonafide = listar_wavs(raiz_mailabs)
        n_bonafide = len(filas_spoof)
        muestra_bonafide = rng.sample(todos_bonafide, min(len(todos_bonafide), n_bonafide))
        for wav in muestra_bonafide:
            filas_bonafide.append(ruta_a_fila(wav, "M-AILABS", "multi", "bonafide", "zeroshot"))
        print(f"  Zero-shot bonafide: {len(filas_bonafide):,} archivos (M-AILABS, idiomas entrenamiento)")
    else:
        print(f"  ADVERTENCIA: no se encontró M-AILABS en {raiz_mailabs}")

    return filas_spoof + filas_bonafide


# ---------------------------------------------------------------------------
# Escenario 3: Fuera de dominio — SONAR + SpoofCeleb
# ---------------------------------------------------------------------------

def etiqueta_sonar(partes) -> str | None:
    """
    Etiqueta de un archivo de SONAR segun su subconjunto y subcarpeta.
      LJSpeech        : grabaciones reales                        -> bonafide
      LibriSeVoc      : carpeta 'gt' = real; el resto son vocoders -> bonafide / spoof
      In_the_Wild     : carpetas 'bona-fide' / 'spoof'
      SONAR_core      : carpetas 'bonafide' / 'spoof'
      generated_audio : audio generado                            -> spoof
    Devuelve None si el subconjunto o la carpeta no tienen regla (no se asume clase).
    """
    p = set(partes)
    if "LJSpeech" in p:
        return "bonafide"
    if "LibriSeVoc" in p:
        return "bonafide" if "gt" in p else "spoof"
    if "In_the_Wild" in p or "SONAR_core" in p:
        if p & {"bonafide", "bona-fide"}:
            return "bonafide"
        if "spoof" in p:
            return "spoof"
        return None
    if "generated_audio" in p:
        return "spoof"
    return None


def generar_fueradominio(base: Path, rng: random.Random) -> list[dict]:
    """
    SONAR y SpoofCeleb desde Bases de Datos extraidas.
    SONAR estructura: base/SONAR/{generated_audio,In_the_Wild,...}/en/{spoof,bonafide}/
    SpoofCeleb estructura: base/SpoofCeleb/train/en/spoof/
    Muestreo proporcional al número de archivos de cada base.
    """
    filas_sonar = []
    filas_spoofceleb = []

    # SONAR — la etiqueta depende del subconjunto y de su subcarpeta
    raiz_sonar = base / "SONAR"
    sin_etiqueta = 0
    for wav in listar_wavs(raiz_sonar):
        label = etiqueta_sonar(wav.parts)
        if label is None:
            sin_etiqueta += 1
            continue
        filas_sonar.append(ruta_a_fila(wav, "SONAR", "en", label, "fueradominio"))
    if sin_etiqueta:
        print(f"  ADVERTENCIA: {sin_etiqueta:,} archivos de SONAR sin regla de etiqueta (excluidos)")

    # SpoofCeleb
    raiz_spoofceleb = base / "SpoofCeleb"
    for wav in listar_wavs(raiz_spoofceleb):
        filas_spoofceleb.append(ruta_a_fila(wav, "SpoofCeleb", "en", "spoof", "fueradominio"))

    total = len(filas_sonar) + len(filas_spoofceleb)
    print(f"  Fuera de dominio encontrados: SONAR={len(filas_sonar):,} | SpoofCeleb={len(filas_spoofceleb):,}")

    # Muestreo proporcional si supera el máximo
    if total > MAX_FUERADOMINIO:
        prop_sonar = len(filas_sonar) / total
        n_sonar = int(MAX_FUERADOMINIO * prop_sonar)
        n_spoofceleb = MAX_FUERADOMINIO - n_sonar
        filas_sonar = rng.sample(filas_sonar, min(n_sonar, len(filas_sonar)))
        filas_spoofceleb = rng.sample(filas_spoofceleb, min(n_spoofceleb, len(filas_spoofceleb)))

    filas = filas_sonar + filas_spoofceleb
    print(f"  Fuera de dominio seleccionados: {len(filas):,}")
    return filas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genera los 3 manifests de prueba para evaluación del modelo"
    )
    parser.add_argument("--base", type=Path, default=BASE_PROCESADA,
                        help="Ruta raíz de los audios procesados")
    parser.add_argument("--salida", type=Path, default=SALIDA_DIR,
                        help="Directorio donde se guardan los manifests de test")
    args = parser.parse_args()

    rng = random.Random(SEMILLA)

    print("=" * 60)
    print("GENERANDO MANIFESTS DE TEST")
    print(f"Base: {args.base}")
    print(f"Salida: {args.salida}")
    print("=" * 60)

    print("\n[1/3] En distribución (ASVspoof 2019 LA dev + eval)...")
    filas_ind = generar_indistribucion(BASE_EXTRAIDA)
    escribir_manifest(filas_ind, args.salida / "manifest_test_indistribucion.csv")

    print("\n[2/3] Zero-shot (MLAAD idiomas no vistos)...")
    filas_zs = generar_zeroshot(BASE_EXTRAIDA, rng)
    escribir_manifest(filas_zs, args.salida / "manifest_test_zeroshot.csv")

    print("\n[3/3] Fuera de dominio (SONAR + SpoofCeleb)...")
    filas_fd = generar_fueradominio(BASE_EXTRAIDA, rng)
    escribir_manifest(filas_fd, args.salida / "manifest_test_fueradominio.csv")

    total = len(filas_ind) + len(filas_zs) + len(filas_fd)
    print(f"\nTotal archivos de test: {total:,}")
    print("Listo.")


if __name__ == "__main__":
    main()
