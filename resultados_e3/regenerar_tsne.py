# -*- coding: utf-8 -*-
"""Regenera las figuras t-SNE del Exp-4 con 16 colores/marcadores distintos (una por idioma)."""
import csv, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

IDIOMAS = ["ar", "bg", "cs", "de", "en", "es", "fi", "fr", "he", "hi", "id", "it", "ja", "ko", "pl", "ru"]
NOMBRES = {"ar": "árabe", "bg": "búlgaro", "cs": "checo", "de": "alemán", "en": "inglés", "es": "español",
           "fi": "finlandés", "fr": "francés", "he": "hebreo", "hi": "hindi", "id": "indonesio", "it": "italiano",
           "ja": "japonés", "ko": "coreano", "pl": "polaco", "ru": "ruso"}
# 16 colores distinguibles (tab20 sin los grises) y 4 marcadores para reforzar
COLORES = plt.get_cmap("tab20")(list(range(0, 4)) + list(range(4, 20)))
paleta = [COLORES[i] for i in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11)]
MARC = ["o", "s", "^", "D"]
COL = {l: paleta[i] for i, l in enumerate(IDIOMAS)}
MK = {l: MARC[i % 4] for i, l in enumerate(IDIOMAS)}

datos = collections.defaultdict(list)
for r in csv.DictReader(open("tsne_coordenadas.csv", encoding="utf-8")):
    datos[(r["flujo"], r["version"])].append((float(r["x"]), float(r["y"]), r["idioma"], r["clase"]))

NOM_FLUJO = {"wav2vec2bert": "Wav2Vec2BERT", "wavlm": "WavLM Large", "whisper": "Whisper Large v3"}
for flujo in ("whisper", "wavlm", "wav2vec2bert"):
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    for j, ver in enumerate(("antes", "despues")):
        pts = datos[(flujo, ver)]
        for l in IDIOMAS:
            xs = [p[0] for p in pts if p[2] == l]; ys = [p[1] for p in pts if p[2] == l]
            ax[0][j].scatter(xs, ys, s=6, c=[COL[l]], marker=MK[l], linewidths=0)
        for cl, col in (("bonafide", "#2ca02c"), ("spoof", "#d62728")):
            xs = [p[0] for p in pts if p[3] == cl]; ys = [p[1] for p in pts if p[3] == cl]
            ax[1][j].scatter(xs, ys, s=6, c=col, linewidths=0)
        t = "antes de RASA" if ver == "antes" else "después de RASA"
        ax[0][j].set_title(f"{NOM_FLUJO[flujo]} - {t} (por idioma)", fontsize=11)
        ax[1][j].set_title(f"{NOM_FLUJO[flujo]} - {t} (por clase)", fontsize=11)
        for i in (0, 1):
            ax[i][j].set_xticks([]); ax[i][j].set_yticks([])
    h = [Line2D([0], [0], marker=MK[l], color="w", markerfacecolor=COL[l], markersize=7, label=NOMBRES[l]) for l in IDIOMAS]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.045))
    h2 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7, label=n)
          for n, c in (("bonafide", "#2ca02c"), ("spoof", "#d62728"))]
    fig.legend(handles=h2, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(f"tsne_{flujo}_16colores.png", dpi=160)
    plt.close(fig)
    print("ok", flujo)
