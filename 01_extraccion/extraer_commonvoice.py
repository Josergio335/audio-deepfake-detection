"""
Extraccion y organizacion de Mozilla Common Voice para el escenario zero-shot.

Fuente:  D:/Bases de Datos sin extraer/Mozilla Data Collective/  (y la raiz de sin extraer)
Destino: D:/Bases de Datos extraidas/CommonVoice/{lang}/clips/*.mp3
         + D:/Bases de Datos extraidas/CommonVoice/{lang}/validated.tsv

Solo extrae los clips que aparecen en validated.tsv (bonafide confirmado por votos).
El script es REANUDABLE: cada idioma queda marcado con un archivo .done y se salta
en ejecuciones posteriores. Si un idioma falla a la mitad, borra su .done y repite.

Uso:
    python extraer_commonvoice.py               # procesa todo lo encontrado
    python extraer_commonvoice.py --solo ja     # solo un idioma
    python extraer_commonvoice.py --dry-run     # muestra qué haría sin extraer
    python extraer_commonvoice.py --reset ja    # fuerza re-extracción de un idioma
    python extraer_commonvoice.py --listar      # muestra estado de cada idioma
"""

import tarfile
import argparse
import csv
import shutil
import logging
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE        = Path("D:/")
FUENTES     = [
    BASE / "Bases de Datos sin extraer" / "Mozilla Data Collective",
    BASE / "Bases de Datos sin extraer",   # el et.tar.gz suelto
]
DESTINO     = BASE / "Bases de Datos extraidas" / "CommonVoice"
MARCADORES  = BASE / ".marcadores_extraccion" / "CommonVoice"
LOG_FILE    = DESTINO / "extraccion_commonvoice.log"

DESTINO.mkdir(parents=True, exist_ok=True)
MARCADORES.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def buscar_tars():
    """Devuelve lista de (ruta_tar, lang) para todos los cv-corpus encontrados."""
    encontrados = []
    for carpeta in FUENTES:
        if not carpeta.exists():
            continue
        for f in sorted(carpeta.glob("*-corpus-*.tar.gz")):
            # nombre: {id}-{cv|sps}-corpus-{version}-{fecha}-{lang}.tar.gz
            sin_ext = f.name.replace(".tar.gz", "")   # quita ambas extensiones
            partes = sin_ext.split("-")
            lang = partes[-1]   # último segmento antes de .tar.gz
            encontrados.append((f, lang))
    return encontrados


def marcador(lang):
    return MARCADORES / f"{lang}.done"


def ya_hecho(lang):
    return marcador(lang).exists()


def marcar_hecho(lang, n_clips):
    marcador(lang).write_text(
        f"completado={datetime.now().isoformat()}  clips={n_clips}\n",
        encoding="utf-8",
    )


def leer_validated_tsv(tf, lang):
    """Extrae validated.tsv del tar y devuelve el conjunto de nombres de archivo validados."""
    candidatos = [
        f"cv-corpus",   # prefijo del path interno
    ]
    for member in tf.getmembers():
        if member.name.endswith(f"{lang}/validated.tsv"):
            f = tf.extractfile(member)
            if f is None:
                return set()
            csv.field_size_limit(10_000_000)
            reader = csv.DictReader(f.read().decode("utf-8").splitlines(), delimiter="\t")
            return {row["path"] for row in reader}
    return set()


# ---------------------------------------------------------------------------
# Extracción de un idioma
# ---------------------------------------------------------------------------

def extraer_idioma(tar_path: Path, lang: str, dry_run=False):
    destino_clips = DESTINO / lang / "clips"
    destino_tsv   = DESTINO / lang / "validated.tsv"

    if ya_hecho(lang):
        log.info(f"[{lang}] Ya completado, saltando.")
        return

    log.info(f"[{lang}] Abriendo {tar_path.name} ...")
    if dry_run:
        log.info(f"[{lang}] [DRY-RUN] Se extraerían clips validados a {destino_clips}")
        return

    destino_clips.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            todos = tf.getmembers()

            # Detectar tipo de corpus por estructura interna
            es_sps = any("/audios/" in m.name for m in todos)

            if es_sps:
                # sps-corpus: carpeta audios/, sin validated.tsv — todos son bonafide
                validados = set()
                miembros_clips = [m for m in todos if "/audios/" in m.name and m.name.endswith(".mp3")]
                log.info(f"[{lang}] sps-corpus detectado — {len(miembros_clips)} clips en audios/")
            else:
                # cv-corpus: carpeta clips/ + validated.tsv
                validados = leer_validated_tsv(tf, lang)
                if not validados:
                    log.warning(f"[{lang}] validated.tsv vacío o no encontrado; extrayendo todos los clips.")

                # Extraer validated.tsv al destino
                for member in todos:
                    if member.name.endswith(f"{lang}/validated.tsv"):
                        member_data = tf.extractfile(member)
                        if member_data:
                            destino_tsv.write_bytes(member_data.read())
                            log.info(f"[{lang}] validated.tsv guardado ({len(validados)} filas)")
                        break

                miembros_clips = [
                    m for m in todos
                    if f"/{lang}/clips/" in m.name and m.name.endswith(".mp3")
                ]
            clips_total = len(miembros_clips)

            clips_ok = 0
            clips_salteados = 0
            clips_total = len(miembros_clips)

            for member in miembros_clips:
                nombre = Path(member.name).name   # solo el filename.mp3
                destino_archivo = destino_clips / nombre

                # Si hay lista de validados, filtrar
                if validados and nombre not in validados:
                    continue

                # Reanudable: saltar si ya existe
                if destino_archivo.exists():
                    clips_salteados += 1
                    continue

                data = tf.extractfile(member)
                if data:
                    contenido = data.read()
                    for intento in range(3):
                        try:
                            destino_archivo.write_bytes(contenido)
                            clips_ok += 1
                            break
                        except PermissionError:
                            import time as _time
                            _time.sleep(1 + intento)
                    else:
                        log.warning(f"[{lang}] No se pudo escribir {nombre} tras 3 intentos, saltando.")

                if (clips_ok + clips_salteados) % 500 == 0 and (clips_ok + clips_salteados) > 0:
                    log.info(f"[{lang}]  ... {clips_ok} nuevos, {clips_salteados} ya existían")

        total_en_disco = len(list(destino_clips.glob("*.mp3")))
        log.info(f"[{lang}] LISTO — {clips_ok} extraídos, {clips_salteados} ya existían, "
                 f"{total_en_disco} clips en disco de {clips_total} en el archivo")
        marcar_hecho(lang, total_en_disco)

    except Exception as e:
        log.error(f"[{lang}] ERROR: {e}")
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extrae Common Voice para zero-shot bonafide")
    parser.add_argument("--solo",    metavar="LANG", help="Procesar solo este idioma (ej: ja)")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar qué haría sin extraer")
    parser.add_argument("--reset",   metavar="LANG", help="Borrar marcador y forzar re-extracción")
    parser.add_argument("--listar",  action="store_true", help="Mostrar estado de cada idioma")
    args = parser.parse_args()

    tars = buscar_tars()
    if not tars:
        log.error("No se encontraron archivos cv-corpus-*.tar.gz en las rutas de búsqueda.")
        return

    if args.reset:
        m = marcador(args.reset)
        if m.exists():
            m.unlink()
            log.info(f"Marcador de '{args.reset}' eliminado. Se re-extraerá en la siguiente ejecución.")
        else:
            log.info(f"No había marcador para '{args.reset}'.")
        return

    if args.listar:
        print(f"\n{'Idioma':<8} {'Archivo':<55} {'Estado'}")
        print("-" * 80)
        for tar_path, lang in tars:
            estado = "✓ hecho" if ya_hecho(lang) else "pendiente"
            clips_dir = DESTINO / lang / "clips"
            n = len(list(clips_dir.glob("*.mp3"))) if clips_dir.exists() else 0
            print(f"{lang:<8} {tar_path.name:<55} {estado}  ({n} clips en disco)")
        return

    for tar_path, lang in tars:
        if args.solo and lang != args.solo:
            continue
        extraer_idioma(tar_path, lang, dry_run=args.dry_run)

    log.info("Extracción finalizada.")


if __name__ == "__main__":
    main()
