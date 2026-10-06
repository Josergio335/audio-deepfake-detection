"""
Pruebas unitarias del modulo de extraccion de caracteristicas (R5).
Usan extractores falsos: no descargan ni ejecutan los modelos fundacionales reales.
"""

import csv
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

import extraer_embeddings as ee

DIM = 8


# ---------------------------------------------------------------------------
# Utilidades de prueba
# ---------------------------------------------------------------------------

def escribir_wav(ruta, segundos=1.0, sr=16000, canales=1):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    n = int(segundos * sr)
    datos = (0.1 * np.random.randn(n, canales)).astype("float32")
    if canales == 1:
        datos = datos[:, 0]
    sf.write(str(ruta), datos, sr)


class ExtractorFalso:
    """Devuelve un vector (DIM,) que depende de la duracion del audio."""
    llamadas = 0

    def __init__(self, device):
        pass

    def extraer(self, audio):
        ExtractorFalso.llamadas += 1
        return torch.full((DIM,), float(audio.shape[1]))


class ExtractorQueFalla:
    def __init__(self, device):
        pass

    def extraer(self, audio):
        raise AssertionError("no debia procesarse: el embedding ya existia")


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Redirige las rutas del script a una carpeta temporal."""
    monkeypatch.setattr(ee, "EMBEDDINGS_DIR", tmp_path / "embeddings")
    monkeypatch.setattr(ee, "INPUT_BASE", tmp_path / "procesadas")
    monkeypatch.setattr(ee, "MANIFEST", tmp_path / "manifest.csv")
    monkeypatch.setitem(ee.EXTRACTORES, "falso", ExtractorFalso)
    ExtractorFalso.llamadas = 0
    return tmp_path


def preparar_corpus(base, n=3):
    """Crea n audios procesados y devuelve la lista (ruta_original, ruta_procesada)."""
    rutas = []
    for i in range(n):
        original = f"D:\\Bases de Datos extraidas\\MLAAD\\es\\audio_{i}.mp3"
        procesada = base / "procesadas" / "MLAAD" / "es" / f"audio_{i}.wav"
        escribir_wav(procesada, segundos=0.5 + i)
        rutas.append((original, str(procesada)))
    return rutas


# ---------------------------------------------------------------------------
# Hash y manifest
# ---------------------------------------------------------------------------

def test_hash_determinista_y_distinto_por_ruta():
    assert ee.ruta_a_hash("a/b.wav") == ee.ruta_a_hash("a/b.wav")
    assert ee.ruta_a_hash("a/b.wav") != ee.ruta_a_hash("a/c.wav")
    assert len(ee.ruta_a_hash("a/b.wav")) == 32


def test_leer_manifest_convierte_ruta_windows_a_wav(entorno):
    original = "D:\\Bases de Datos extraidas\\MLAAD\\es\\audio_0.mp3"
    with open(ee.MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ruta", "idioma"])
        w.writerow([original, "es"])
    (ruta_orig, ruta_proc), = ee.leer_manifest()
    assert ruta_orig == original
    esperada = (ee.INPUT_BASE / "MLAAD/es/audio_0").with_suffix(".wav")
    assert ruta_proc == str(esperada)


# ---------------------------------------------------------------------------
# Carga de audio
# ---------------------------------------------------------------------------

def test_cargar_audio_forma_y_frecuencia(tmp_path):
    ruta = tmp_path / "a.wav"
    escribir_wav(ruta, segundos=1.0, sr=16000)
    audio = ee.cargar_audio(str(ruta))
    assert audio.shape == (1, 16000)
    assert audio.dtype == torch.float32


def test_cargar_audio_estereo_a_mono(tmp_path):
    ruta = tmp_path / "estereo.wav"
    escribir_wav(ruta, segundos=1.0, canales=2)
    assert ee.cargar_audio(str(ruta)).shape == (1, 16000)


def test_cargar_audio_remuestrea_a_16k(tmp_path):
    if getattr(ee.torchaudio.functional, "resample", None) is None:
        pytest.skip("torchaudio no instalado")
    ruta = tmp_path / "8k.wav"
    escribir_wav(ruta, segundos=1.0, sr=8000)
    assert ee.cargar_audio(str(ruta)).shape[1] == 16000


# ---------------------------------------------------------------------------
# Pooling de los tres extractores (con procesador y modelo falsos)
# ---------------------------------------------------------------------------

class _ProcesadorFalso:
    """Devuelve entradas cuya longitud depende del audio, como los reales."""
    def __call__(self, audio_np, sampling_rate, return_tensors, **kw):
        t = max(1, len(audio_np) // 320)
        return {"input_values": torch.zeros(1, t), "input_features": torch.zeros(1, t, 4)}


class _ModeloFalso:
    def __init__(self):
        self.encoder = self._salida

    def _salida(self, *a, **k):
        t = a[0].shape[1] if a else 5
        return SimpleNamespace(last_hidden_state=torch.randn(1, t, DIM))

    def __call__(self, **inputs):
        t = inputs["input_values"].shape[1] if "input_values" in inputs else 5
        return SimpleNamespace(last_hidden_state=torch.randn(1, t, DIM))


@pytest.mark.parametrize("clase", ["ExtractorWav2Vec2BERT", "ExtractorWavLM", "ExtractorWhisper"])
def test_embedding_tiene_dimension_fija_sin_importar_la_duracion(clase):
    ext = object.__new__(getattr(ee, clase))
    ext.device = "cpu"
    ext.processor = _ProcesadorFalso()
    ext.model = _ModeloFalso()
    for segundos in (0.5, 1.0, 3.0):
        audio = torch.randn(1, int(16000 * segundos))
        emb = ext.extraer(audio)
        assert emb.shape == (DIM,)
        assert torch.isfinite(emb).all()


# ---------------------------------------------------------------------------
# Proceso por modelo: guardado, reanudacion, errores
# ---------------------------------------------------------------------------

def test_extraer_modelo_guarda_un_tensor_por_audio(entorno):
    rutas = preparar_corpus(entorno, n=3)
    ee.extraer_modelo("falso", rutas, "cpu")
    for original, _ in rutas:
        destino = ee.EMBEDDINGS_DIR / "falso" / f"{ee.ruta_a_hash(original)}.pt"
        assert destino.exists()
        assert torch.load(destino, weights_only=True).shape == (DIM,)


def test_extraccion_es_reanudable(entorno, monkeypatch):
    rutas = preparar_corpus(entorno, n=3)
    ee.extraer_modelo("falso", rutas, "cpu")
    assert ExtractorFalso.llamadas == 3
    # Segunda ejecucion: todo existe, no debe volver a procesar ni cargar el extractor
    monkeypatch.setitem(ee.EXTRACTORES, "falso", ExtractorQueFalla)
    ee.extraer_modelo("falso", rutas, "cpu")


def test_reanuda_solo_los_pendientes(entorno):
    rutas = preparar_corpus(entorno, n=3)
    ee.extraer_modelo("falso", rutas[:1], "cpu")
    ExtractorFalso.llamadas = 0
    ee.extraer_modelo("falso", rutas, "cpu")
    assert ExtractorFalso.llamadas == 2


def test_archivo_faltante_no_detiene_el_resto(entorno):
    rutas = preparar_corpus(entorno, n=3)
    rutas.insert(1, ("D:\\Bases de Datos extraidas\\MLAAD\\es\\no_existe.mp3",
                     str(entorno / "procesadas" / "no_existe.wav")))
    ee.extraer_modelo("falso", rutas, "cpu")
    pt = list((ee.EMBEDDINGS_DIR / "falso").glob("*.pt"))
    assert len(pt) == 3


def test_audio_vacio_no_genera_tensor(entorno):
    rutas = preparar_corpus(entorno, n=1)
    vacio = entorno / "procesadas" / "vacio.wav"
    sf.write(str(vacio), np.zeros(0, dtype="float32"), 16000)
    rutas.append(("D:\\Bases de Datos extraidas\\vacio.mp3", str(vacio)))
    ee.extraer_modelo("falso", rutas, "cpu")
    assert len(list((ee.EMBEDDINGS_DIR / "falso").glob("*.pt"))) == 1


def test_dry_run_no_escribe_tensores(entorno):
    rutas = preparar_corpus(entorno, n=2)
    ee.extraer_modelo("falso", rutas, "cpu", dry_run=True)
    assert list((ee.EMBEDDINGS_DIR / "falso").glob("*.pt")) == []


# ---------------------------------------------------------------------------
# Indice
# ---------------------------------------------------------------------------

def test_indice_relaciona_cada_audio_con_sus_tres_tensores(entorno):
    rutas = preparar_corpus(entorno, n=3)
    ee.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)  # en la ejecucion real la crea setup_logging
    ee.generar_indice(rutas)
    with open(ee.EMBEDDINGS_DIR / "indice.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    assert len(filas) == 3
    for fila, (original, _) in zip(filas, rutas):
        h = ee.ruta_a_hash(original)
        assert fila["ruta_original"] == original
        assert fila["hash"] == h
        for col, carpeta in (("wav2vec2bert_pt", "wav2vec2bert"),
                             ("wavlm_pt", "wavlm"), ("whisper_pt", "whisper")):
            assert fila[col] == str(ee.EMBEDDINGS_DIR / carpeta / f"{h}.pt")
