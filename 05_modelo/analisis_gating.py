"""
analisis_gating.py (R6 / pregunta del jurado: ¿aporta la fusion con Gating?)

Sobre un checkpoint ya entrenado (sin reentrenar) calcula, por escenario de test:
  - EER/AUROC de cada stream por separado (w2v, wavlm, whisper): head lineal sobre el embedding (alineado por RASA si lo hay)
  - EER/AUROC con promedio fijo de los 3 logits (g = 1/3 cada uno)
  - EER/AUROC con el Gating aprendido (el modelo tal cual)
  - Pesos g medios del Gating por escenario, por base de datos y por idioma

Uso:
  python analisis_gating.py --checkpoint ~/entrenamientos/exp4_20261119_125609/checkpoint_mejor.pt \
      --escenarios indistribucion fueradominio zeroshot_exp4 --device cuda:1
Salida: <carpeta del checkpoint>/analisis_gating.json y tabla en pantalla.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import (TestDataset, ModeloDeepfake, calcular_eer, calcular_auroc)


def collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]), torch.stack([b[3] for b in batch]),
            [b[4] for b in batch], [b[5] for b in batch])


@torch.no_grad()
def inferir_streams(modelo, loader, device):
    S, G, F, Y, DB, ID = [], [], [], [], [], []
    for e1, e2, e3, y, dbs, idiomas in tqdm(loader, desc="eval"):
        e1, e2, e3 = e1.to(device), e2.to(device), e3.to(device)
        a1, a2, a3 = modelo.rasa_w2v(e1), modelo.rasa_wavlm(e2), modelo.rasa_whisp(e3)
        s1 = modelo.head_w2v(a1).squeeze(-1)
        s2 = modelo.head_wavlm(a2).squeeze(-1)
        s3 = modelo.head_whisp(a3).squeeze(-1)
        g = modelo.gating([a1, a2, a3])                      # (B, 3)
        st = torch.stack([s1, s2, s3], dim=1)                # (B, 3)
        S.append(st.cpu().numpy())
        G.append(g.cpu().numpy())
        F.append((g * st).sum(dim=1).cpu().numpy())
        Y.append(y.numpy())
        DB.extend(dbs)
        ID.extend(idiomas)
    return (np.concatenate(S), np.concatenate(G), np.concatenate(F),
            np.concatenate(Y), np.array(DB), np.array(ID))


def metricas(y, s):
    return {"eer": calcular_eer(y, s) * 100, "auroc": calcular_auroc(y, s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--escenarios", nargs="+", default=["indistribucion", "fueradominio", "zeroshot_exp4"])
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    ck_path = Path(args.checkpoint).expanduser()
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    modelo = ModeloDeepfake(M=cfg.get("M", 160), sin_rasa=cfg.get("sin_rasa", False)).to(args.device)
    modelo.load_state_dict(ck["model_state"])
    modelo.eval()

    nombres = ["wav2vec2bert", "wavlm", "whisper"]
    resultado = {"checkpoint": str(ck_path), "escenarios": {}}
    for esc in args.escenarios:
        ds = TestDataset(esc)
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, collate_fn=collate)
        S, G, F, Y, DB, ID = inferir_streams(modelo, loader, args.device)

        r = {"n": int(len(Y)),
             "gating_aprendido": metricas(Y, F),
             "promedio_fijo": metricas(Y, S.mean(axis=1)),
             "streams": {n: metricas(Y, S[:, i]) for i, n in enumerate(nombres)},
             "g_medio": {n: float(G[:, i].mean()) for i, n in enumerate(nombres)},
             "g_std": {n: float(G[:, i].std()) for i, n in enumerate(nombres)},
             "g_medio_por_clase": {
                 c: {n: float(G[Y == v, i].mean()) for i, n in enumerate(nombres)}
                 for c, v in (("bonafide", 0), ("spoof", 1)) if (Y == v).any()},
             "g_medio_por_db": {}, "g_medio_por_idioma": {}}
        for db in np.unique(DB):
            m = DB == db
            if m.sum() >= 10:
                r["g_medio_por_db"][str(db)] = {n: float(G[m, i].mean()) for i, n in enumerate(nombres)}
        for idi in np.unique(ID):
            m = ID == idi
            if m.sum() >= 10:
                r["g_medio_por_idioma"][str(idi)] = {n: float(G[m, i].mean()) for i, n in enumerate(nombres)}
        resultado["escenarios"][esc] = r

        print(f"\n=== {esc} (n={r['n']:,}) ===")
        print(f"{'Configuracion':<22}{'EER %':>8}{'AUROC':>9}")
        for n in nombres:
            print(f"{'solo ' + n:<22}{r['streams'][n]['eer']:>8.2f}{r['streams'][n]['auroc']:>9.4f}")
        print(f"{'promedio fijo (1/3)':<22}{r['promedio_fijo']['eer']:>8.2f}{r['promedio_fijo']['auroc']:>9.4f}")
        print(f"{'Gating aprendido':<22}{r['gating_aprendido']['eer']:>8.2f}{r['gating_aprendido']['auroc']:>9.4f}")
        print("g medio: " + "  ".join(f"{n}={r['g_medio'][n]:.3f}" for n in nombres))
        print("g desv.: " + "  ".join(f"{n}={r['g_std'][n]:.3f}" for n in nombres))
        for c, gc in r["g_medio_por_clase"].items():
            print(f"g {c:<9}: " + "  ".join(f"{n}={gc[n]:.3f}" for n in nombres))

    salida = ck_path.parent / "analisis_gating.json"
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {salida}")


if __name__ == "__main__":
    main()
