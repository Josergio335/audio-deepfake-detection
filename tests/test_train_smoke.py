"""
Prueba de humo del entrenamiento (R1/R2).
Ejecuta train.py de punta a punta con datos sinteticos (embeddings aleatorios, un
manifest y un indice de juguete), en CPU y con pocas epocas. Demuestra que el
codigo instala, carga los datos, entrena los tres flujos con RASA, SERM y gating,
y deja las metricas y los checkpoints, sin necesidad de GPU ni de datos reales.
"""

import csv
import json
import sys

import pytest
import torch

import train

N = 100
DIMS = {"wav2vec2bert": 1024, "wavlm": 1024, "whisper": 1280}


@pytest.fixture
def datos_sinteticos(tmp_path, monkeypatch):
    emb_dir = tmp_path / "embeddings"
    part_dir = tmp_path / "particiones" / "Exp1"
    salida = tmp_path / "entrenamientos"
    for modelo in DIMS:
        (emb_dir / modelo).mkdir(parents=True)
    part_dir.mkdir(parents=True)

    g = torch.Generator().manual_seed(0)
    filas_manifest, filas_indice = [], []
    for i in range(N):
        clase = i % 2  # 0 = bonafide, 1 = spoof
        ruta = f"D:\\datos\\audio_{i}.wav"
        filas_manifest.append([ruta, "bonafide" if clase == 0 else "spoof", "es", "sintetica"])
        fila = [ruta, f"h{i}"]
        for modelo, d in DIMS.items():
            # clases ligeramente separables para que la perdida pueda bajar
            emb = 0.05 * torch.randn(d, generator=g) + (0.05 if clase else -0.05)
            pt = emb_dir / modelo / f"h{i}.pt"
            torch.save(emb, pt)
            fila.append(str(pt))
        filas_indice.append(fila)

    with open(part_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ruta", "label", "idioma", "db"])
        w.writerows(filas_manifest)
    with open(emb_dir / "indice.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ruta_original", "hash", "wav2vec2bert_pt", "wavlm_pt", "whisper_pt"])
        w.writerows(filas_indice)

    monkeypatch.setattr(train, "EMBEDDINGS_DIR", emb_dir)
    monkeypatch.setattr(train, "SALIDA_BASE", salida)
    monkeypatch.setattr(train, "PARTICIONES", {1: part_dir, 2: part_dir, 3: part_dir})
    return salida


def test_entrenamiento_de_punta_a_punta(datos_sinteticos, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--exp", "1", "--epochs", "3", "--bs", "16", "--workers", "0",
        "--device", "cpu", "--lambda-orth", "1.0", "--lambda-reco", "1.0",
    ])
    train.main()

    corridas = list(datos_sinteticos.glob("exp1_*"))
    assert len(corridas) == 1
    carpeta = corridas[0]
    for nombre in ("checkpoint_mejor.pt", "checkpoint_ultimo.pt", "metricas.csv", "config.json"):
        assert (carpeta / nombre).exists(), f"falta {nombre}"

    with open(carpeta / "metricas.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    assert len(filas) == 3
    for fila in filas:
        for col in ("train_bce", "train_orth", "train_reco", "val_bce", "val_accuracy"):
            assert torch.isfinite(torch.tensor(float(fila[col]))), f"{col} no es finito"
    assert float(filas[-1]["train_bce"]) < float(filas[0]["train_bce"])

    config = json.load(open(carpeta / "config.json"))
    assert config["lambda_orth"] == 1.0

    ck = torch.load(carpeta / "checkpoint_mejor.pt", map_location="cpu", weights_only=False)
    assert "model_state" in ck and "rasa_w2v.mu_bank" in ck["model_state"]


def test_split_determinista_y_proporcion():
    rutas = [f"D:\\datos\\audio_{i}.wav" for i in range(5000)]
    a = [train.asignar_split(r) for r in rutas]
    b = [train.asignar_split(r) for r in rutas]
    assert a == b
    pct_val = a.count("val") / len(a)
    assert 0.17 < pct_val < 0.23


def test_dataset_etiquetas_y_formas(datos_sinteticos):
    ds = train.EmbeddingDataset(train.PARTICIONES[1], "train")
    e1, e2, e3, y = ds[0]
    assert e1.shape == (1024,) and e2.shape == (1024,) and e3.shape == (1280,)
    assert y.item() in (0.0, 1.0)
    val = train.EmbeddingDataset(train.PARTICIONES[1], "val")
    assert len(ds) + len(val) == N


def test_modelo_forma_de_salida_y_perdidas_finitas():
    modelo = train.ModeloDeepfake().train()
    b = 8
    x = [torch.randn(b, d) * 0.1 for d in (1024, 1024, 1280)]
    score, emb_list = modelo(*x)
    assert score.shape == (b,)
    aux = modelo.perdidas_auxiliares(emb_list, torch.randint(0, 2, (b,)).float())
    for nombre, valor in aux.items():
        assert torch.isfinite(valor), nombre
