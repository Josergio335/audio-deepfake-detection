"""
conftest.py
Configuracion comun de las pruebas: agrega 05_modelo al path y, si no estan
instaladas librerias pesadas que solo usan los extractores reales
(transformers, torchaudio, tqdm), las reemplaza por modulos minimos para que
las pruebas corran en CI sin descargar modelos.
"""

import sys
import types
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "05_modelo"))


def _falso(nombre, **atributos):
    modulo = types.ModuleType(nombre)
    for k, v in atributos.items():
        setattr(modulo, k, v)
    sys.modules[nombre] = modulo


try:
    import transformers  # noqa: F401
except ImportError:
    class _Dummy:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise RuntimeError("transformers no esta instalado (modulo falso de pruebas)")

    _falso("transformers",
           AutoFeatureExtractor=_Dummy, Wav2Vec2BertModel=_Dummy,
           WavLMModel=_Dummy, WhisperModel=_Dummy)

try:
    import torchaudio  # noqa: F401
except ImportError:
    _falso("torchaudio", functional=types.SimpleNamespace(resample=None))

try:
    import tqdm  # noqa: F401
except ImportError:
    _falso("tqdm", tqdm=lambda it, **k: it)
