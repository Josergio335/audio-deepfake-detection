"""
Pruebas de la regla de etiquetado de SONAR (escenario fuera de dominio) y de la
semilla de entrenamiento. La regla antigua marcaba como bonafide todo LibriSeVoc e
In_the_Wild, incluidos los vocoders y la carpeta spoof.
"""

from pathlib import PureWindowsPath

import pytest
import torch

import train
from generar_manifests_test import etiqueta_sonar

RAIZ = r"D:\Bases de Datos extraidas\SONAR"


def partes(ruta):
    return PureWindowsPath(ruta).parts


@pytest.mark.parametrize("ruta, esperado", [
    (RAIZ + r"\LJSpeech\LJSpeech-1.1\wavs\LJ036-0028.wav", "bonafide"),
    (RAIZ + r"\LibriSeVoc\en\gt\1_2_3.wav", "bonafide"),
    (RAIZ + r"\LibriSeVoc\en\diffwave\1_2_3_gen.wav", "spoof"),
    (RAIZ + r"\LibriSeVoc\en\melgan\1_2_3_gen.wav", "spoof"),
    (RAIZ + r"\LibriSeVoc\en\parallel_wave_gan\1_2_3_gen.wav", "spoof"),
    (RAIZ + r"\LibriSeVoc\en\wavegrad\1_2_3_gen.wav", "spoof"),
    (RAIZ + r"\LibriSeVoc\en\wavenet\1_2_3_gen.wav", "spoof"),
    (RAIZ + r"\LibriSeVoc\en\wavernn\1_2_3_gen.wav", "spoof"),
    (RAIZ + r"\In_the_Wild\en\bona-fide\10.wav", "bonafide"),
    (RAIZ + r"\In_the_Wild\en\spoof\6869.wav", "spoof"),
    (RAIZ + r"\SONAR_core\en\bonafide\a.wav", "bonafide"),
    (RAIZ + r"\SONAR_core\en\spoof\OpenAI\echo_65.wav", "spoof"),
    (RAIZ + r"\generated_audio\en\ljspeech_hifiGAN\LJ015-0208_generated.wav", "spoof"),
])
def test_etiqueta_sonar(ruta, esperado):
    assert etiqueta_sonar(partes(ruta)) == esperado


def test_subconjunto_desconocido_no_se_asume():
    assert etiqueta_sonar(partes(RAIZ + r"\OtraBase\en\x\a.wav")) is None
    assert etiqueta_sonar(partes(RAIZ + r"\In_the_Wild\en\otra\a.wav")) is None


def test_semilla_reproducible():
    train.fijar_semilla(123)
    a = torch.rand(5)
    train.fijar_semilla(123)
    b = torch.rand(5)
    assert torch.equal(a, b)
