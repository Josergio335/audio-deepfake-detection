"""
analisis_rasa_tsne.py
Compara los embeddings ANTES y DESPUES de RASA (Resultado 6) sobre una muestra
estratificada del conjunto de VALIDACION de un experimento.

Para cada uno de los tres flujos (Wav2Vec2BERT, WavLM, Whisper):
  - toma una muestra de N audios por (idioma, clase) del conjunto de validacion
    (audios que el banco de RASA no vio al entrenar),
  - aplica el RASA entrenado (del checkpoint indicado) a sus embeddings,
  - calcula t-SNE antes y despues (coloreado por idioma y por clase),
  - calcula metricas numericas antes y despues: silueta por idioma y por clase,
    exactitud de un clasificador lineal de idioma y de clase, y la discrepancia
    de estilo (media, desviacion) entre idiomas.

Uso (en el servidor, sin GPU):
    CUDA_VISIBLE_DEVICES="" python analisis_rasa_tsne.py --exp 1 \
        --carpeta ~/entrenamientos/exp1_20261006_002845 --dry-run
    CUDA_VISIBLE_DEVICES="" python analisis_rasa_tsne.py --exp 1 \
        --carpeta ~/entrenamientos/exp1_20261006_002845

Salida, en <carpeta>/analisis_rasa/:
    muestra_embeddings.npz      embeddings antes/despues, idioma y clase
    tsne_coordenadas.csv        coordenadas t-SNE (flujo, version, x, y, idioma, clase)
    resumen_metricas.csv        metricas antes/despues por flujo
    tsne_<flujo>.png            figuras (si hay matplotlib)

Requiere scikit-learn y matplotlib:  pip install --user scikit-learn matplotlib
"""

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import ModeloDeepfake, PARTICIONES, EMBEDDINGS_DIR, asignar_split

# (nombre del embedding, atributo RASA del modelo)
FLUJOS = [
    ("wav2vec2bert", "rasa_w2v"),
    ("wavlm",        "rasa_wavlm"),
    ("whisper",      "rasa_whisp"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Muestra
# ---------------------------------------------------------------------------

def muestrear(exp, n_por_grupo, semilla, val_pct):
    """Muestra estratificada por (idioma, clase) del conjunto de validacion."""
    grupos = defaultdict(list)
    with open(PARTICIONES[exp] / "manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if asignar_split(r["ruta"], val_pct) == "val":
                grupos[(r["idioma"], r["label"])].append(r["ruta"])
    rng = random.Random(semilla)
    muestra = []
    for clave in sorted(grupos):
        rutas = grupos[clave]
        rng.shuffle(rutas)
        for ruta in rutas[:n_por_grupo]:
            muestra.append((ruta, clave[0], clave[1]))
    return muestra, {k: len(v) for k, v in grupos.items()}


def localizar_embeddings(muestra):
    """Busca en indice.csv las rutas de los .pt de cada audio de la muestra."""
    objetivo = {m[0] for m in muestra}
    filas = {}
    with open(EMBEDDINGS_DIR / "indice.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ruta_original"] in objetivo:
                filas[row["ruta_original"]] = row
    return filas


def cargar_embeddings(muestra, filas, nombre):
    tensores = []
    for ruta, _, _ in muestra:
        pt = Path(filas[ruta][f"{nombre}_pt"])
        tensores.append(torch.load(pt, map_location="cpu", weights_only=True).float())
    return torch.stack(tensores)


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def discrepancia_estilo(X, idiomas, modulo):
    """
    Distancia media entre los centroides de estilo (media, desviacion) de los
    idiomas, en unidades de dispersion de los estilos del banco.
    """
    mu, sg = X.mean(dim=1), X.std(dim=1)
    z = torch.stack([(mu - modulo.centro_mu) / modulo.escala_mu,
                     (sg - modulo.centro_sigma) / modulo.escala_sigma], dim=1).numpy()
    langs = sorted(set(idiomas))
    cent = np.array([z[np.array(idiomas) == l].mean(axis=0) for l in langs])
    d = [np.linalg.norm(cent[i] - cent[j]) for i in range(len(langs)) for j in range(i + 1, len(langs))]
    return float(np.mean(d)), float(z.std(axis=0).mean())


def metricas(X, idiomas, clases, semilla):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    P = PCA(n_components=min(50, X.shape[1]), random_state=semilla).fit_transform(
        StandardScaler().fit_transform(X))
    cv = StratifiedKFold(3, shuffle=True, random_state=semilla)
    return {
        "silueta_idioma": float(silhouette_score(P, idiomas)),
        "silueta_clase":  float(silhouette_score(P, clases)),
        "acc_idioma":     float(cross_val_score(LogisticRegression(max_iter=2000), P, idiomas, cv=cv).mean()),
        "acc_clase":      float(cross_val_score(LogisticRegression(max_iter=2000), P, clases, cv=cv).mean()),
    }


def tsne(X, semilla, perplexity):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    P = PCA(n_components=min(50, X.shape[1]), random_state=semilla).fit_transform(
        StandardScaler().fit_transform(X))
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=semilla).fit_transform(P)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def graficar(flujo, coords, idiomas, clases, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    langs = sorted(set(idiomas))
    paleta = plt.get_cmap("tab10")
    colores_idioma = {l: paleta(i % 10) for i, l in enumerate(langs)}
    colores_clase = {"bonafide": "tab:green", "spoof": "tab:red"}
    idiomas, clases = np.array(idiomas), np.array(clases)

    fig, ejes = plt.subplots(2, 2, figsize=(11, 9))
    for col, version in enumerate(["antes", "despues"]):
        xy = coords[version]
        ax = ejes[0, col]
        for l in langs:
            m = idiomas == l
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=colores_idioma[l], label=l)
        ax.set_title(f"{flujo} - {'antes' if version == 'antes' else 'después'} de RASA (por idioma)")
        ax = ejes[1, col]
        for c, color in colores_clase.items():
            m = clases == c
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=color, label=c)
        ax.set_title(f"{flujo} - {'antes' if version == 'antes' else 'después'} de RASA (por clase)")
    ejes[0, 1].legend(markerscale=3, fontsize=8, loc="best")
    ejes[1, 1].legend(markerscale=3, fontsize=8, loc="best")
    for ax in ejes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Embeddings antes/despues de RASA: t-SNE y metricas")
    ap.add_argument("--exp", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--carpeta", required=True, help="carpeta del entrenamiento (con checkpoint_mejor.pt)")
    ap.add_argument("--n-por-grupo", type=int, default=150, dest="n")
    ap.add_argument("--val-pct", type=int, default=20, dest="val_pct")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--sin-graficos", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="solo muestra la muestra y comprueba los archivos")
    args = ap.parse_args()

    carpeta = Path(args.carpeta).expanduser()
    salida = carpeta / "analisis_rasa"

    log(f"Experimento {args.exp} | carpeta {carpeta.name} | {args.n} audios por (idioma, clase), validacion {args.val_pct}%")
    muestra, disponibles = muestrear(args.exp, args.n, args.semilla, args.val_pct)
    conteo = defaultdict(int)
    for _, idioma, clase in muestra:
        conteo[(idioma, clase)] += 1
    for clave in sorted(conteo):
        log(f"  {clave[0]:>3} {clave[1]:<8} muestra {conteo[clave]:>4} de {disponibles[clave]:>6} en validacion")
    log(f"Muestra total: {len(muestra):,} audios")

    filas = localizar_embeddings(muestra)
    faltan = [r for r, _, _ in muestra if r not in filas]
    log(f"Audios con embeddings en indice.csv: {len(muestra) - len(faltan):,} (faltan {len(faltan)})")
    if args.dry_run:
        log("Dry-run terminado: no se calculo nada.")
        return
    if faltan:
        muestra = [m for m in muestra if m[0] in filas]

    idiomas = [m[1] for m in muestra]
    clases = [m[2] for m in muestra]

    log("Cargando el modelo entrenado ...")
    modelo = ModeloDeepfake()
    ck = torch.load(carpeta / "checkpoint_mejor.pt", map_location="cpu", weights_only=False)
    modelo.load_state_dict(ck["model_state"])
    modelo.eval()
    log(f"Checkpoint de la epoca {ck['epoch']} (mejor val_loss {ck['mejor_val_loss']:.4f})")

    salida.mkdir(parents=True, exist_ok=True)
    arrays = {"idioma": np.array(idiomas), "clase": np.array(clases)}
    filas_metricas, filas_tsne = [], []

    for nombre, attr in FLUJOS:
        log(f"--- {nombre} ---")
        modulo = getattr(modelo, attr)
        antes = cargar_embeddings(muestra, filas, nombre)
        with torch.no_grad():
            despues = modulo(antes)
        arrays[f"{nombre}_antes"] = antes.numpy()
        arrays[f"{nombre}_despues"] = despues.numpy()

        coords = {}
        for version, X in (("antes", antes), ("despues", despues)):
            Xn = X.numpy()
            disc, disp = discrepancia_estilo(X, idiomas, modulo)
            met = metricas(Xn, idiomas, clases, args.semilla)
            met.update({"flujo": nombre, "version": version,
                        "discrepancia_estilo_entre_idiomas": disc, "dispersion_estilo": disp})
            filas_metricas.append(met)
            log(f"  {version:>7}: silueta idioma {met['silueta_idioma']:+.3f} | clase {met['silueta_clase']:+.3f} | "
                f"acc idioma {met['acc_idioma']:.3f} | acc clase {met['acc_clase']:.3f} | discrepancia de estilo {disc:.3f}")
            t0 = time.time()
            coords[version] = tsne(Xn, args.semilla, args.perplexity)
            log(f"  t-SNE {version} listo ({time.time() - t0:.0f} s)")
            for (x, y), i, c in zip(coords[version], idiomas, clases):
                filas_tsne.append({"flujo": nombre, "version": version, "x": f"{x:.4f}", "y": f"{y:.4f}",
                                   "idioma": i, "clase": c})
        if not args.sin_graficos:
            try:
                graficar(nombre, coords, idiomas, clases, salida / f"tsne_{nombre}.png")
                log(f"  figura guardada: tsne_{nombre}.png")
            except ImportError:
                log("  matplotlib no esta instalado: se omiten las figuras (pip install --user matplotlib)")

    np.savez_compressed(salida / "muestra_embeddings.npz", **arrays)
    with open(salida / "resumen_metricas.csv", "w", newline="", encoding="utf-8") as f:
        campos = ["flujo", "version", "silueta_idioma", "silueta_clase", "acc_idioma", "acc_clase",
                  "discrepancia_estilo_entre_idiomas", "dispersion_estilo"]
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for fila in filas_metricas:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in fila.items()})
    with open(salida / "tsne_coordenadas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["flujo", "version", "x", "y", "idioma", "clase"])
        w.writeheader()
        w.writerows(filas_tsne)
    log(f"Listo. Resultados en {salida}")


if __name__ == "__main__":
    main()
