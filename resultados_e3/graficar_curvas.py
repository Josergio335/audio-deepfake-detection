import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
runs = {"Exp-3 convergida":"exp3_20261119_143040","Exp-3 fallida":"exp3_20261119_144355",
        "Exp-4 convergida":"exp4_20261119_125609","Exp-4 fallida":"exp4_20261119_125614"}
col = {"Exp-3":"tab:blue","Exp-4":"tab:orange"}
def load(r):
    with open(f"metricas_{r}.csv") as f: return list(csv.DictReader(f))
fig, ax = plt.subplots(1,3, figsize=(15,4))
for lab, r in runs.items():
    d = load(r); e=[int(x["epoch"]) for x in d]
    ls = "--" if "fallida" in lab else "-"; c = col[lab[:5]]
    ax[0].plot(e,[float(x["train_bce"]) for x in d], ls, color=c, label=lab)
    ax[1].plot(e,[float(x["val_bce"]) for x in d], ls, color=c, label=lab)
    ax[2].plot(e,[100*float(x["val_accuracy"]) for x in d], ls, color=c, label=lab)
for a,t in zip(ax,["Pérdida BCE (entrenamiento)","Pérdida BCE (validación)","Exactitud de validación (%)"]):
    a.set_title(t); a.set_xlabel("Época"); a.grid(alpha=.3)
ax[1].set_yscale("log"); ax[2].set_ylim(88,100); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("curvas_r1.png", dpi=150)
for lab,r in runs.items():
    d=load(r); b=min(d,key=lambda x:float(x["val_total"]))
    print(lab, len(d),"épocas; mejor val_total ép.",b["epoch"],"val_acc",b["val_accuracy"],"val_bce",b["val_bce"])
