"""
generar_degradaciones.py
Genera EN EL SERVIDOR versiones degradadas (MP3 y perdida de paquetes, PLR) de una muestra
estratificada del test zero-shot multilingue, a partir del audio ya procesado, sin subir datos.
Pensado para el Resultado 7 (robustez frente a la degradacion en varios idiomas).

Condiciones (--condiciones):
    mp3_<kbps>   compresion MP3 a ese bitrate medio (codificada con soundfile/libsndfile;
                 el nivel de compresion se CALIBRA midiendo el bitrate real obtenido)
    plr_<pct>    perdida de paquetes: tramas de 20 ms anuladas al azar con ese porcentaje

Siempre se genera tambien la referencia "zs_limpio": la misma muestra sin degradar, que
reutiliza los embeddings ya extraidos (mismo hash que el indice original).

Uso:
    python generar_degradaciones.py --dry-run                   # calibra MP3 y muestra la muestra
    python generar_degradaciones.py                             # mp3_32 y plr_10 por defecto
    python generar_degradaciones.py --condiciones mp3_32 mp3_64 plr_5 plr_10 plr_20

Salida:
    ~/datos/degradaciones/<condicion>/<idioma>/<label>/<hash>.{mp3,wav}
    ~/particiones/manifest_test_zs_<condicion>.csv   (rutas absolutas; listo para
        python extraer_embeddings_test.py --manifest <ese archivo>)
    ~/particiones/muestra_zs.csv                     (la muestra, para reproducibilidad)
"""

import argparse
import csv
import hashlib
import random
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

HOME = Path.home()
EMBEDDINGS_TEST_DIR = HOME / "embeddings_test"
SALIDA_AUDIO = HOME / "datos" / "degradaciones"
MANIFESTS_DIR = HOME / "particiones"
SR = 16000
TRAMA_MS = 20

WIN_PREFIXES = [
    ("D:\\Bases de Datos procesadas\\", HOME / "datos/Bases de Datos procesadas", False),
    ("D:\\Bases de Datos extraidas\\CommonVoice\\", HOME / "datos/Bases de Datos procesadas/CommonVoice", True),
    ("D:\\Bases de Datos extraidas\\", HOME / "datos/Bases de Datos procesadas", True),
]


def resolver(ruta_win: str) -> Path:
    if ruta_win.startswith("/"):
        return Path(ruta_win)
    for pref, base, a_wav in WIN_PREFIXES:
        if ruta_win.startswith(pref):
            p = base / ruta_win[len(pref):].replace("\\", "/")
            return p.with_suffix(".wav") if a_wav else p
    return Path(ruta_win)


def md5(texto: str) -> str:
    return hashlib.md5(texto.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Muestra
# ---------------------------------------------------------------------------

def leer_base(base: str):
    indice = EMBEDDINGS_TEST_DIR / f"indice_test_{base}.csv"
    filas, faltantes = [], 0
    with open(indice, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            origen = resolver(r["ruta_original"])
            if origen.exists():
                filas.append({"ruta_original": r["ruta_original"], "origen": origen,
                              "label": r["label"], "db": r["db"], "idioma": r["idioma"]})
            else:
                faltantes += 1
    print(f"Base {base}: {len(filas):,} archivos con audio ({faltantes:,} sin audio)")
    return filas


def muestrear(filas, n_bona_idioma: int, n_spoof_total: int, semilla: int):
    """Bonafide: n por idioma. Spoof: cuota igual por idioma hasta n_spoof_total."""
    rng = random.Random(semilla)
    por_grupo = defaultdict(list)
    for fila in filas:
        por_grupo[(fila["idioma"], fila["label"])].append(fila)

    muestra = []
    for (idioma, label), lst in sorted(por_grupo.items()):
        if label == "bonafide":
            lst = sorted(lst, key=lambda x: x["ruta_original"])
            rng.shuffle(lst)
            muestra.extend(lst[:n_bona_idioma])

    idiomas_spoof = sorted({i for (i, l) in por_grupo if l == "spoof"})
    cuota = max(1, n_spoof_total // max(1, len(idiomas_spoof)))
    for idioma in idiomas_spoof:
        lst = sorted(por_grupo[(idioma, "spoof")], key=lambda x: x["ruta_original"])
        rng.shuffle(lst)
        muestra.extend(lst[:cuota])
    return muestra


# ---------------------------------------------------------------------------
# Degradaciones
# ---------------------------------------------------------------------------

def leer_audio(ruta: Path) -> np.ndarray:
    audio, sr = sf.read(str(ruta), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise ValueError(f"frecuencia de muestreo {sr} != {SR}: {ruta}")
    return audio


def _fijar_nivel_mp3(f, nivel: float) -> None:
    """Intenta fijar bitrate constante y nivel de compresion en libsndfile (best effort)."""
    try:
        f.compression_level = nivel
        return
    except Exception:
        pass
    try:
        ffi, lib = sf._ffi, sf._snd
        modo = ffi.new("int*", 0)  # SF_BITRATE_MODE_CONSTANT
        lib.sf_command(f._file, 0x1305, modo, 4)          # SFC_SET_BITRATE_MODE
        valor = ffi.new("double*", float(nivel))
        lib.sf_command(f._file, 0x1301, valor, 8)         # SFC_SET_COMPRESSION_LEVEL
    except Exception:
        pass


def escribir_mp3(destino: Path, audio: np.ndarray, nivel):
    destino.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(destino), "w", samplerate=SR, channels=1,
                      format="MP3", subtype="MPEG_LAYER_III") as f:
        if nivel is not None:
            _fijar_nivel_mp3(f, nivel)
        f.write(audio)


def aplicar_plr(audio: np.ndarray, porcentaje: float, semilla: int) -> np.ndarray:
    n = int(SR * TRAMA_MS / 1000)
    rng = np.random.default_rng(semilla)
    perdidas = rng.random(int(np.ceil(len(audio) / n))) < porcentaje / 100.0
    salida = audio.copy()
    for i in np.flatnonzero(perdidas):
        salida[i * n:(i + 1) * n] = 0.0
    return salida


def calibrar_mp3(muestra, kbps_objetivo: float, n_audios: int = 8):
    """Elige el nivel de compresion cuyo bitrate medio se acerca mas al objetivo."""
    audios = []
    for fila in muestra[:200]:
        try:
            a = leer_audio(fila["origen"])
            if len(a) > SR:
                audios.append(a)
        except Exception:
            continue
        if len(audios) >= n_audios:
            break
    if not audios:
        raise RuntimeError("no se pudo leer ningun audio para calibrar")

    resultados = []
    with tempfile.TemporaryDirectory() as tmp:
        for nivel in [None] + [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]:
            kbps = []
            for i, a in enumerate(audios):
                dst = Path(tmp) / f"c_{i}.mp3"
                escribir_mp3(dst, a, nivel)
                kbps.append(dst.stat().st_size * 8 / (len(a) / SR) / 1000)
            resultados.append((nivel, float(np.mean(kbps))))
    print("Calibracion MP3 (nivel -> bitrate medio medido):")
    for nivel, k in resultados:
        print(f"   nivel={'por defecto' if nivel is None else nivel}: {k:6.1f} kbps")
    mejor = min(resultados, key=lambda x: abs(x[1] - kbps_objetivo))
    return mejor, resultados


def _tarea(args):
    origen, destino, cond, nivel, semilla = args
    try:
        audio = leer_audio(Path(origen))
        destino = Path(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        if cond.startswith("mp3_"):
            escribir_mp3(destino, audio, nivel)
        else:
            sf.write(str(destino), aplicar_plr(audio, float(cond.split("_")[1]), semilla),
                     SR, subtype="PCM_16")
        return True
    except Exception as e:
        return f"{origen}: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def escribir_manifest(ruta: Path, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ruta", "db", "idioma", "label", "escenario", "ruta_origen"])
        w.writerows(filas)


def main():
    ap = argparse.ArgumentParser(description="Genera degradaciones multilingues en el servidor")
    ap.add_argument("--base", default="zeroshot_exp4")
    ap.add_argument("--condiciones", nargs="+", default=["mp3_32", "plr_10"])
    ap.add_argument("--n-bona-idioma", type=int, default=300, dest="n_bona")
    ap.add_argument("--n-spoof-total", type=int, default=2700, dest="n_spoof")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tolerancia", type=float, default=0.15,
                    help="desvio relativo maximo aceptado entre el bitrate objetivo y el medido")
    ap.add_argument("--dry-run", action="store_true", help="solo calibra y resume la muestra")
    args = ap.parse_args()

    filas = leer_base(args.base)
    muestra = muestrear(filas, args.n_bona, args.n_spoof, args.semilla)
    por = defaultdict(int)
    for m in muestra:
        por[m["label"]] += 1
    n_idiomas = len({m["idioma"] for m in muestra})
    print(f"Muestra: {len(muestra):,} archivos ({dict(por)}) en {n_idiomas} idiomas")

    niveles = {}
    for cond in args.condiciones:
        if cond.startswith("mp3_"):
            objetivo = float(cond.split("_")[1])
            (nivel, medido), _ = calibrar_mp3(muestra, objetivo)
            desvio = abs(medido - objetivo) / objetivo
            etiqueta_nivel = "por defecto" if nivel is None else nivel
            print(f"  {cond}: nivel elegido {etiqueta_nivel} -> {medido:.1f} kbps (objetivo {objetivo:g}, desvio {desvio:.0%})")
            if desvio > args.tolerancia:
                sys.exit(f"ERROR: no se logra {objetivo:g} kbps con este codificador "
                         f"(mejor {medido:.1f}). No se genera {cond}.")
            niveles[cond] = nivel
        elif not cond.startswith("plr_"):
            sys.exit(f"Condicion no reconocida: {cond} (use mp3_<kbps> o plr_<pct>)")

    if args.dry_run:
        print("--dry-run: no se genera audio.")
        return

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    escribir_manifest(MANIFESTS_DIR / "muestra_zs.csv",
                      [(m["ruta_original"], m["db"], m["idioma"], m["label"], "muestra", str(m["origen"]))
                       for m in muestra])

    # Referencia sin degradar: reutiliza los embeddings existentes (mismo hash)
    escribir_manifest(MANIFESTS_DIR / "manifest_test_zs_limpio.csv",
                      [(m["ruta_original"], m["db"], m["idioma"], m["label"], "zs_limpio", str(m["origen"]))
                       for m in muestra])

    comandos = ["python extraer_embeddings_test.py --manifest ~/particiones/manifest_test_zs_limpio.csv"]
    for cond in args.condiciones:
        ext = "mp3" if cond.startswith("mp3_") else "wav"
        escenario = "zs_" + (cond + "kbps" if cond.startswith("mp3_") else cond.replace("plr_", "plr"))
        tareas, filas_manifest = [], []
        for m in muestra:
            destino = SALIDA_AUDIO / cond / m["idioma"] / m["label"] / f"{md5(m['ruta_original'])}.{ext}"
            semilla = int(md5(m["ruta_original"])[:8], 16)
            tareas.append((str(m["origen"]), str(destino), cond, niveles.get(cond), semilla))
            filas_manifest.append((str(destino), m["db"], m["idioma"], m["label"], escenario, str(m["origen"])))

        print(f"\nGenerando {cond}: {len(tareas):,} archivos ...")
        errores = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(_tarea, tareas, chunksize=32):
                if res is not True:
                    errores.append(res)
        print(f"  listos: {len(tareas) - len(errores):,} | errores: {len(errores):,}")
        for e in errores[:5]:
            print("   ", e)
        escribir_manifest(MANIFESTS_DIR / f"manifest_test_{escenario}.csv", filas_manifest)
        comandos.append(f"python extraer_embeddings_test.py --manifest ~/particiones/manifest_test_{escenario}.csv")

    print("\nSiguiente paso (extraer embeddings; reanudable):")
    for c in comandos:
        print("   ", c)


if __name__ == "__main__":
    main()
