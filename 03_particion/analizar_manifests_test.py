"""
analizar_manifests_test.py
Analiza los manifests de los 4 escenarios de prueba y genera:
  - Manifests enriquecidos con duracion_seg (si no la tienen)
  - resumen_particiones_test.txt  (mismo formato que resumen_particiones.txt)

Escenarios cubiertos:
  1. En distribución  → manifest_test_indistribucion.csv
  2. Zero-shot        → manifest_test_zeroshot_v2.csv  (servidor, rutas Linux)
                        o manifest_test_zeroshot.csv    (local, rutas Windows)
  3. Fuera de dominio → manifest_test_fueradominio.csv
  4. Compresión       → manifest_test_compresion_{codec}_{kbps}kbps.csv (6 variantes)

Uso:
    python analizar_manifests_test.py                    # todos los manifests en PARTICIONES_DIR
    python analizar_manifests_test.py --dry-run          # solo resumen, no enriquece manifests
    python analizar_manifests_test.py --manifest ruta.csv  # un manifest específico

Funciona tanto en Windows (rutas D:\\...) como en Linux (rutas /home/...).
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    PARTICIONES_DIR = Path(r"D:\Particiones")
    OUTPUT_DIR      = PARTICIONES_DIR
    CSV_DURACIONES  = Path(r"D:\Scripts Extraccion\duraciones.csv")
    # (prefijo_windows, base_linux, cambiar_extension_a_wav) — no usado en Windows
    WIN_PREFIXES = [
        (r"D:\Bases de Datos procesadas\\",           Path.home() / "datos/Bases de Datos procesadas", False),
        (r"D:\Bases de Datos extraidas\CommonVoice\\", Path.home() / "datos/Bases de Datos procesadas/CommonVoice", True),
        (r"D:\Bases de Datos extraidas\\",             Path.home() / "datos/Bases de Datos procesadas", True),
    ]
else:
    PARTICIONES_DIR = Path.home() / "particiones"
    OUTPUT_DIR      = PARTICIONES_DIR
    CSV_DURACIONES  = Path.home() / "datos/duraciones.csv"
    # (prefijo_windows, base_linux, cambiar_extension_a_wav)
    WIN_PREFIXES = [
        (r"D:\Bases de Datos procesadas\\",           Path.home() / "datos/Bases de Datos procesadas", False),
        (r"D:\Bases de Datos extraidas\CommonVoice\\", Path.home() / "datos/Bases de Datos procesadas/CommonVoice", True),
        (r"D:\Bases de Datos extraidas\\",             Path.home() / "datos/Bases de Datos procesadas", True),
    ]

# ---------------------------------------------------------------------------
# Cargar lookup de duraciones pre-calculadas
# ---------------------------------------------------------------------------

def cargar_duraciones_lookup(csv_path: Path) -> dict:
    """
    Carga duraciones.csv como lookup: nombre_sin_extension → duracion_seg.
    Permite encontrar rápidamente la duración por nombre de archivo.
    """
    if not csv_path.exists():
        print(f"  [!] duraciones.csv no encontrado en {csv_path}. Se medirá desde audio.")
        return {}
    lookup = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nombre = Path(row["ruta"]).stem  # nombre sin extensión
            try:
                lookup[nombre] = float(row["duracion_seg"])
            except (KeyError, ValueError):
                pass
    print(f"  duraciones.csv cargado: {len(lookup):,} entradas.")
    return lookup

# ---------------------------------------------------------------------------
# Resolución de rutas (Windows → Linux si es necesario)
# ---------------------------------------------------------------------------

def resolver_ruta(ruta_str: str) -> Path:
    """Convierte ruta del manifest a ruta real en este sistema."""
    if ruta_str.startswith("/"):
        return Path(ruta_str)
    if not IS_WINDOWS:
        for win_pref, linux_base, a_wav in WIN_PREFIXES:
            if ruta_str.startswith(win_pref):
                rel = ruta_str[len(win_pref):].replace("\\", "/")
                ruta = linux_base / rel
                if a_wav:
                    ruta = ruta.with_suffix(".wav")
                return ruta
    return Path(ruta_str)

# ---------------------------------------------------------------------------
# Medir duración de un archivo de audio (con lookup primero)
# ---------------------------------------------------------------------------

def medir_duracion(ruta: Path, lookup: dict) -> float:
    """
    Devuelve duración en segundos.
    Estrategia: 1) lookup por stem (rápido), 2) soundfile.info(), 3) torchaudio.
    """
    # 1. Lookup por nombre de archivo sin extensión (cubre WAV, FLAC y MP3 del mismo stem)
    stem = ruta.stem
    if stem in lookup:
        return lookup[stem]

    # 2. soundfile (lee solo el header)
    try:
        import soundfile as sf
        info = sf.info(str(ruta))
        return info.frames / info.samplerate
    except Exception:
        pass

    # 3. torchaudio como último recurso
    try:
        import torchaudio
        info = torchaudio.info(str(ruta))
        return info.num_frames / info.sample_rate
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# Leer manifest y enriquecer con duracion_seg
# ---------------------------------------------------------------------------

def leer_manifest_con_duracion(manifest_path: Path, lookup: dict,
                               verbose: bool = True) -> list[dict]:
    """
    Lee el manifest. Si ya tiene duracion_seg, la usa.
    Si no, primero consulta el lookup (duraciones.csv), luego mide desde audio.
    Devuelve lista de dicts con: ruta, db, idioma, label, duracion_seg, + extras.
    """
    filas = []
    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows_raw = list(reader)

    tiene_duracion = "duracion_seg" in fieldnames
    total = len(rows_raw)

    if verbose:
        print(f"  Leyendo {total:,} filas de {manifest_path.name} ...", end="", flush=True)
        if not tiene_duracion:
            print(f" (lookup + audio si necesario)")
        else:
            print(f" (usando duracion_seg del manifest)")

    sin_lookup = errores = 0
    for row in rows_raw:
        if tiene_duracion:
            dur = float(row["duracion_seg"])
        else:
            ruta = resolver_ruta(row["ruta"])
            stem = ruta.stem
            if stem in lookup:
                dur = lookup[stem]
            else:
                sin_lookup += 1
                dur = medir_duracion(ruta, lookup)
                if dur == 0.0:
                    errores += 1

        filas.append({
            "ruta":         row["ruta"],
            "db":           row.get("db", ""),
            "idioma":       row.get("idioma", ""),
            "label":        row.get("label", ""),
            "escenario":    row.get("escenario", manifest_path.stem.replace("manifest_test_", "")),
            "codec":        row.get("codec", ""),
            "kbps":         row.get("kbps", ""),
            "split_orig":   row.get("split_original", ""),
            "duracion_seg": dur,
        })

    if verbose:
        msg = f"  OK — {total:,} filas"
        if sin_lookup:
            msg += f" ({sin_lookup:,} medidas desde audio)"
        if errores:
            msg += f" ⚠ {errores} sin duración"
        print(msg)

    return filas

# ---------------------------------------------------------------------------
# Escribir manifest enriquecido
# ---------------------------------------------------------------------------

def escribir_manifest_enriquecido(filas: list[dict], ruta_original: Path):
    """Escribe el manifest con duracion_seg añadida."""
    # Determinar columnas: las originales más duracion_seg
    with open(ruta_original, encoding="utf-8") as f:
        fieldnames_orig = csv.DictReader(f).fieldnames or []

    if "duracion_seg" not in fieldnames_orig:
        fieldnames = list(fieldnames_orig) + ["duracion_seg"]
    else:
        fieldnames = list(fieldnames_orig)

    out_path = ruta_original  # sobrescribe el manifest original añadiendo duracion_seg
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for fila in filas:
            row = {k: fila.get(k, "") for k in fieldnames}
            row["duracion_seg"] = f"{fila['duracion_seg']:.3f}"
            writer.writerow(row)

    print(f"  Enriquecido: {out_path.name}  ({len(filas):,} filas)")

# ---------------------------------------------------------------------------
# Calcular y formatear resumen (mismo estilo que particion_dataset.py)
# ---------------------------------------------------------------------------

def calcular_resumen_test(filas: list[dict], nombre: str, extra_col: str = None) -> str:
    """
    Genera resumen textual por idioma/clase (y opcionalmente por codec/kbps o db).
    extra_col: 'codec_kbps' para compresión, 'db' para fuera de dominio, None para los demás.
    """
    por_idioma_label = defaultdict(lambda: [0, 0.0])
    por_db           = defaultdict(lambda: [0, 0.0])
    por_extra        = defaultdict(lambda: [0, 0.0])

    for a in filas:
        k = (a["idioma"], a["label"])
        por_idioma_label[k][0] += 1
        por_idioma_label[k][1] += a["duracion_seg"] / 3600
        por_db[a["db"]][0] += 1
        por_db[a["db"]][1] += a["duracion_seg"] / 3600
        if extra_col == "codec_kbps" and a["codec"]:
            ck = f"{a['codec']}_{a['kbps']}kbps"
            por_extra[ck][0] += 1
            por_extra[ck][1] += a["duracion_seg"] / 3600

    total_h = sum(a["duracion_seg"] for a in filas) / 3600

    lines = []
    lines.append(f"\n{'='*62}")
    lines.append(f"  {nombre}")
    lines.append(f"{'='*62}")
    lines.append(f"  Total archivos : {len(filas):,}")
    lines.append(f"  Total horas    : {total_h:.1f}h")

    lines.append(f"\n  Por idioma y clase:")
    lines.append(f"  {'Idioma':<8} {'Clase':<10} {'Archivos':>10} {'Horas':>8}")
    lines.append(f"  {'-'*40}")
    for (lang, lbl) in sorted(por_idioma_label):
        n, h = por_idioma_label[(lang, lbl)]
        lines.append(f"  {lang:<8} {lbl:<10} {n:>10,} {h:>8.1f}h")
    total_n = sum(v[0] for v in por_idioma_label.values())
    total_hh = sum(v[1] for v in por_idioma_label.values())
    lines.append(f"  {'TOTAL':<19} {total_n:>10,} {total_hh:>8.1f}h")

    lines.append(f"\n  Por dataset:")
    lines.append(f"  {'Dataset':<20} {'Archivos':>10} {'Horas':>8}")
    lines.append(f"  {'-'*41}")
    for db in sorted(por_db):
        n, h = por_db[db]
        lines.append(f"  {db:<20} {n:>10,} {h:>8.1f}h")

    if extra_col == "codec_kbps" and por_extra:
        lines.append(f"\n  Por codec y bitrate:")
        lines.append(f"  {'Codec/Kbps':<20} {'Archivos':>10} {'Horas':>8}")
        lines.append(f"  {'-'*41}")
        for ck in sorted(por_extra):
            n, h = por_extra[ck]
            lines.append(f"  {ck:<20} {n:>10,} {h:>8.1f}h")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Detectar manifests disponibles
# ---------------------------------------------------------------------------

def detectar_manifests(dir_path: Path) -> list[Path]:
    """Devuelve todos los manifest_test_*.csv del directorio."""
    manifests = sorted(dir_path.glob("manifest_test_*.csv"))
    # Excluir versiones _con_duracion (ya enriquecidas)
    return [m for m in manifests if "_con_duracion" not in m.name]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analiza manifests de test y genera resumen de composición"
    )
    parser.add_argument("--manifest", type=Path,
                        help="Analizar solo este manifest (por defecto: todos en PARTICIONES_DIR)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo genera resumen, no escribe manifests enriquecidos")
    parser.add_argument("--dir", type=Path, default=PARTICIONES_DIR,
                        help=f"Directorio de manifests (default: {PARTICIONES_DIR})")
    args = parser.parse_args()

    print("=" * 62)
    print("  ANÁLISIS DE MANIFESTS DE TEST")
    print("=" * 62)
    if args.dry_run:
        print("  *** DRY-RUN: no se escribirán manifests enriquecidos ***")
    print(f"  Directorio : {args.dir}")
    print()

    if args.manifest:
        manifests = [args.manifest]
    else:
        manifests = detectar_manifests(args.dir)
        if not manifests:
            print(f"ERROR: No se encontraron manifest_test_*.csv en {args.dir}")
            return
        print(f"  Manifests encontrados: {len(manifests)}")
        for m in manifests:
            print(f"    {m.name}")
        print()

    # Cargar lookup de duraciones pre-calculadas (una sola vez, cubre todos los manifests)
    lookup = cargar_duraciones_lookup(CSV_DURACIONES)

    resumen_total = ""
    resumen_total += f"RESUMEN MANIFESTS DE TEST\n"
    resumen_total += f"Generado : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    for manifest_path in manifests:
        nombre = manifest_path.stem.replace("manifest_test_", "").replace("_", " ").upper()

        # Detectar si es escenario de compresión
        es_compresion = "compresion" in manifest_path.name

        filas = leer_manifest_con_duracion(manifest_path, lookup, verbose=True)
        if not filas:
            print(f"  [!] {manifest_path.name}: vacío, saltando.")
            continue

        extra_col = "codec_kbps" if es_compresion else None
        resumen = calcular_resumen_test(filas, nombre, extra_col=extra_col)
        print(resumen)
        resumen_total += resumen + "\n"

        if not args.dry_run:
            # Verificar si ya tiene duracion_seg
            with open(manifest_path, encoding="utf-8") as f:
                tiene = "duracion_seg" in (csv.DictReader(f).fieldnames or [])
            if not tiene:
                escribir_manifest_enriquecido(filas, manifest_path)

    # Escribir resumen
    out_txt = OUTPUT_DIR / "resumen_particiones_test.txt"
    if not args.dry_run:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(resumen_total)
        print(f"\n  Resumen guardado en: {out_txt}")
    else:
        print(f"\n[DRY-RUN] Resumen no guardado (usa sin --dry-run para guardar).")


if __name__ == "__main__":
    main()
