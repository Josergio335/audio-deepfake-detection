"""
analisis_r8.py
Analisis estadistico para el Resultado 8 (comparacion con lineas base) y t-DCF del Resultado 7.
Solo usa numpy y trabaja sobre los scores por archivo ya guardados:
  - nuestros modelos: ~/evaluaciones/scores/<escenario>/<corrida>.csv   (exportar_scores.py)
  - baselines:        ~/entrenamientos_baseline/<corrida>/eval_<escenario>_scores.csv (eval_baseline.py)

Secciones:
  1. EER y AUROC con intervalo de confianza del 95 % por bootstrap, por escenario y modelo
     (en indistribucion tambien solo sobre ASVspoof 2019 LA `eval`).
  2. Diferencia pareada de EER respecto al primer modelo de --ours, con IC 95 % (mismos
     archivos remuestreados en ambos modelos).
  3. min t-DCF normalizado en ASVspoof 2019 LA `eval` (formula oficial del reto, con los
     puntajes del ASV del paquete LA) para nuestros modelos.
  4. EER por idioma en zeroshot_exp4 y su dispersion (medida de sesgo idiomatico).

Uso:
    python analisis_r8.py \
        --ours "Exp-3=exp3_20261119_143040" "Exp-4 A=exp4_20261119_125609" \
        --baselines "LFCC-LCNN=lfcc_lcnn_exp4_20261121_005925" \
                    "RawNet2 simpl.=rawnet2_exp4_20261121_005928" \
                    "AASIST simpl.=aasist_exp4_20261121_005931" \
        --asv ~/datos/asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt --reps 300
"""

import argparse
import csv
from pathlib import Path

import numpy as np

HOME = Path.home()
EMB_TEST = HOME / "embeddings_test"
SCORES_DIR = HOME / "evaluaciones" / "scores"
BASE_DIR = HOME / "entrenamientos_baseline"
OUT_DIR = HOME / "evaluaciones" / "analisis_r8"
ESCENARIOS = ["indistribucion", "fueradominio", "compresion_mp3_32kbps", "zeroshot_exp4"]

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Metricas (score alto = mas spoof, salvo en las funciones de t-DCF)
# ---------------------------------------------------------------------------

def roc(pos, neg):
    """AUROC y EER con pos=spoof y neg=bonafide."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(-s, kind="stable")
    s, y = s[o], y[o]
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    ult = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tpr = np.r_[0.0, tps[ult] / len(pos)]
    fpr = np.r_[0.0, fps[ult] / len(neg)]
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float(_trapz(tpr, fpr)), float((fpr[i] + fnr[i]) / 2)


def det_cm(bona, spoof):
    """Curva DET; los scores altos significan 'bonafide'. Umbrales como en el codigo oficial."""
    todos = np.sort(np.concatenate([bona, spoof]))
    umbrales = np.concatenate([[todos[0] - 1e-3], todos])
    b, s = np.sort(bona), np.sort(spoof)
    pmiss = np.searchsorted(b, umbrales, side="left") / len(b)
    pfa = 1 - np.searchsorted(s, umbrales, side="left") / len(s)
    return pmiss, pfa, umbrales


def leer_asv(ruta: Path):
    tar, non, spoof = [], [], []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            p = linea.split()
            if len(p) < 3:
                continue
            {"target": tar, "nontarget": non, "spoof": spoof}[p[1]].append(float(p[2]))
    return np.array(tar), np.array(non), np.array(spoof)


def tasas_asv(tar, non, spoof):
    pmiss, pfa, th = det_cm(tar, non)
    i = int(np.argmin(np.abs(pmiss - pfa)))
    umbral = th[i]
    return {
        "eer_asv": float((pmiss[i] + pfa[i]) / 2),
        "Pfa_asv": float(np.mean(non >= umbral)),
        "Pmiss_asv": float(np.mean(tar < umbral)),
        "Pmiss_spoof_asv": float(np.mean(spoof < umbral)),
    }


def min_tdcf(bona_cm, spoof_cm, asv):
    """min t-DCF normalizado (modelo de costos de ASVspoof 2019 LA)."""
    Pspoof = 0.05
    Ptar, Pnon = (1 - Pspoof) * 0.99, (1 - Pspoof) * 0.01
    Cmiss_asv, Cfa_asv, Cmiss_cm, Cfa_cm = 1, 10, 1, 10
    C1 = Ptar * (Cmiss_cm - Cmiss_asv * asv["Pmiss_asv"]) - Pnon * Cfa_asv * asv["Pfa_asv"]
    C2 = Cfa_cm * Pspoof * (1 - asv["Pmiss_spoof_asv"])
    if C1 < 0 or C2 < 0:
        raise ValueError("costos negativos: revisar los puntajes del ASV")
    pmiss, pfa, _ = det_cm(bona_cm, spoof_cm)
    return float(np.min((C1 * pmiss + C2 * pfa) / min(C1, C2)))


# ---------------------------------------------------------------------------
# Carga de scores
# ---------------------------------------------------------------------------

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cargar_nuestro(esc, corrida):
    d = {"ruta": [], "y": [], "idioma": [], "s": []}
    with open(SCORES_DIR / esc / f"{corrida}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d["ruta"].append(r["ruta_original"])
            d["y"].append(1 if r["label"] == "spoof" else 0)
            d["idioma"].append(r["idioma"])
            d["s"].append(float(r["logit"]))
    return {k: np.array(v) for k, v in d.items()}


def cargar_baseline(esc, corrida, idiomas):
    d = {"ruta": [], "y": [], "idioma": [], "s": []}
    with open(BASE_DIR / corrida / f"eval_{esc}_scores.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d["ruta"].append(r["ruta"])
            d["y"].append(int(r["label"]))
            d["idioma"].append(idiomas.get(r["ruta"], ""))
            d["s"].append(float(r["score_spoof"]))
    d["s"] = logit(np.array(d["s"]))
    return {k: np.array(v) for k, v in d.items()}


def mapa_idiomas(esc):
    with open(EMB_TEST / f"indice_test_{esc}.csv", encoding="utf-8") as f:
        return {r["ruta_original"]: r["idioma"] for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def ic(valores):
    lo, hi = np.percentile(valores, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_modelo(y, s, reps, rng):
    pos, neg = s[y == 1], s[y == 0]
    aucs, eers = [], []
    for _ in range(reps):
        a, e = roc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))])
        aucs.append(a)
        eers.append(e)
    return ic(aucs), ic(eers)


def bootstrap_diferencia(y, s_ref, s_otro, reps, rng):
    ip, ineg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    difs = []
    for _ in range(reps):
        idx = np.concatenate([ip[rng.integers(0, len(ip), len(ip))], ineg[rng.integers(0, len(ineg), len(ineg))]])
        yy = y[idx]
        e_ref = roc(s_ref[idx][yy == 1], s_ref[idx][yy == 0])[1]
        e_otro = roc(s_otro[idx][yy == 1], s_otro[idx][yy == 0])[1]
        difs.append(e_ref - e_otro)
    return ic(difs)


def pct(x):
    return f"{x * 100:6.2f}%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parsear(pares):
    out = []
    for p in pares:
        etiqueta, corrida = p.split("=", 1)
        out.append((etiqueta, corrida))
    return out


def informe_tdcf(a, ours, datos, rng):
    """min t-DCF normalizado sobre los archivos de LA eval, con el audio original y recomprimido a MP3.
    Los puntajes del sistema ASV de referencia son los del paquete de ASVspoof 2019 (solo cambian los del CM)."""
    tar, non, spoof_asv = leer_asv(a.asv)
    asv = tasas_asv(tar, non, spoof_asv)
    print("\n=== 3. min t-DCF normalizado, ASVspoof 2019 LA eval (limpio y comprimido) ===")
    print(f"  ASV: EER {pct(asv['eer_asv'])} | Pfa {pct(asv['Pfa_asv'])} | Pmiss {pct(asv['Pmiss_asv'])} | "
          f"Pmiss_spoof {pct(asv['Pmiss_spoof_asv'])}  (tar={len(tar):,} non={len(non):,} spoof={len(spoof_asv):,})")
    for esc, marca in (("indistribucion", "/LA/eval/"), ("compresion_mp3_32kbps", "/eval/")):
        print(f"\n  {esc}")
        for et, _ in ours:
            d = datos[(esc, et)]
            m = np.array([marca in x.replace("\\", "/") for x in d["ruta"]])
            bona, spf = -d["s"][m & (d["y"] == 0)], -d["s"][m & (d["y"] == 1)]
            v = min_tdcf(bona, spf, asv)
            boots = [min_tdcf(bona[rng.integers(0, len(bona), len(bona))], spf[rng.integers(0, len(spf), len(spf))], asv)
                     for _ in range(a.reps)]
            lo, hi = ic(boots)
            print(f"    {et:<18} min t-DCF {v:.4f} [{lo:.4f}, {hi:.4f}]   (bonafide={len(bona):,}, spoof={len(spf):,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-tdcf", action="store_true", dest="solo_tdcf",
                    help="solo calcula el t-DCF (limpio y comprimido); no escribe los demas resultados")
    ap.add_argument("--ours", nargs="+", required=True, help='"etiqueta=corrida" (scores de exportar_scores.py)')
    ap.add_argument("--baselines", nargs="*", default=[], help='"etiqueta=carpeta_del_baseline"')
    ap.add_argument("--asv", type=Path, help="ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()

    ours, bases = parsear(a.ours), parsear(a.baselines)
    rng = np.random.default_rng(a.semilla)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datos = {}   # (escenario, etiqueta) -> dict
    for esc in ESCENARIOS:
        idiomas = mapa_idiomas(esc) if esc == "zeroshot_exp4" else {}
        for et, co in ours:
            datos[(esc, et)] = cargar_nuestro(esc, co)
        for et, co in bases:
            datos[(esc, et)] = cargar_baseline(esc, co, idiomas)
    etiquetas = [e for e, _ in ours] + [e for e, _ in bases]

    if a.solo_tdcf:
        informe_tdcf(a, ours, datos, rng)
        return

    # 1. EER y AUROC con IC
    filas = []
    print(f"\n=== 1. EER y AUROC con IC 95 % (bootstrap, {a.reps} repeticiones) ===")
    for esc in ESCENARIOS:
        subconjuntos = [("", None)]
        if esc == "indistribucion":
            subconjuntos.append((" (solo LA eval)", lambda r: np.array(["/LA/eval/" in x.replace("\\", "/") for x in r])))
        for nombre_sub, filtro in subconjuntos:
            print(f"\n{esc}{nombre_sub}")
            for et in etiquetas:
                d = datos[(esc, et)]
                m = np.ones(len(d["y"]), bool) if filtro is None else filtro(d["ruta"])
                y, s = d["y"][m], d["s"][m]
                auc, eer = roc(s[y == 1], s[y == 0])
                (alo, ahi), (elo, ehi) = bootstrap_modelo(y, s, a.reps, rng)
                print(f"  {et:<18} EER {pct(eer)} [{pct(elo)}, {pct(ehi)}]   AUROC {auc:.4f} [{alo:.4f}, {ahi:.4f}]   n={len(y):,}")
                filas.append([esc + nombre_sub, et, eer, elo, ehi, auc, alo, ahi, len(y)])
    with open(OUT_DIR / "eer_auroc_ic.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["escenario", "modelo", "eer", "eer_lo", "eer_hi", "auroc", "auroc_lo", "auroc_hi", "n"])
        w.writerows(filas)

    # 2. Diferencia pareada respecto al primer modelo propio
    ref = ours[0][0]
    print(f"\n=== 2. Diferencia pareada de EER: {ref} menos cada modelo (negativo = {ref} mejor) ===")
    for esc in ESCENARIOS:
        print(f"\n{esc}")
        d_ref = datos[(esc, ref)]
        for et in etiquetas[1:]:
            d = datos[(esc, et)]
            comunes = np.intersect1d(d_ref["ruta"], d["ruta"])
            o_ref, o_d = np.argsort(d_ref["ruta"]), np.argsort(d["ruta"])
            ia = o_ref[np.searchsorted(d_ref["ruta"][o_ref], comunes)]
            ib = o_d[np.searchsorted(d["ruta"][o_d], comunes)]
            y = d_ref["y"][ia]
            e1 = roc(d_ref["s"][ia][y == 1], d_ref["s"][ia][y == 0])[1]
            e2 = roc(d["s"][ib][y == 1], d["s"][ib][y == 0])[1]
            lo, hi = bootstrap_diferencia(y, d_ref["s"][ia], d["s"][ib], a.reps, rng)
            marca = "significativa" if (lo > 0 or hi < 0) else "no significativa"
            print(f"  vs {et:<18} diferencia {e1 * 100 - e2 * 100:+7.2f} pp  IC95 [{lo * 100:+.2f}, {hi * 100:+.2f}]  ({marca})")

    # 3. t-DCF en LA eval (limpio y comprimido)
    if a.asv:
        informe_tdcf(a, ours, datos, rng)

    # 4. EER por idioma en zeroshot_exp4
    print("\n=== 4. EER por idioma en zeroshot_exp4 (idiomas con ambas clases) ===")
    por_modelo = {}
    idiomas_ambos = None
    for et in etiquetas:
        d = datos[("zeroshot_exp4", et)]
        res = {}
        for idi in sorted(set(d["idioma"])):
            m = d["idioma"] == idi
            y, s = d["y"][m], d["s"][m]
            if (y == 0).any() and (y == 1).any():
                res[idi] = roc(s[y == 1], s[y == 0])[1]
        por_modelo[et] = res
        idiomas_ambos = set(res) if idiomas_ambos is None else idiomas_ambos & set(res)
    idiomas_ambos = sorted(idiomas_ambos or [])
    print(f"  {'idioma':<8}" + "".join(f"{et[:16]:>18}" for et in etiquetas))
    for idi in idiomas_ambos:
        print(f"  {idi:<8}" + "".join(f"{pct(por_modelo[et][idi]):>18}" for et in etiquetas))
    print(f"  {'media':<8}" + "".join(f"{pct(np.mean([por_modelo[et][i] for i in idiomas_ambos])):>18}" for et in etiquetas))
    print(f"  {'desv.':<8}" + "".join(f"{pct(np.std([por_modelo[et][i] for i in idiomas_ambos], ddof=1)):>18}" for et in etiquetas))
    print(f"  {'peor':<8}" + "".join(f"{pct(max(por_modelo[et][i] for i in idiomas_ambos)):>18}" for et in etiquetas))
    with open(OUT_DIR / "eer_por_idioma.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idioma"] + etiquetas)
        for idi in idiomas_ambos:
            w.writerow([idi] + [f"{por_modelo[et][idi]:.5f}" for et in etiquetas])
    print(f"\nTablas guardadas en {OUT_DIR}")


if __name__ == "__main__":
    main()
