"""La ablacion sin RASA no agrega parametros de RASA y devuelve la entrada sin cambios."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_modelo"))
from train import ModeloDeepfake  # noqa: E402


def test_sin_rasa_es_identidad_y_sin_perdidas_rasa():
    m = ModeloDeepfake(sin_rasa=True)
    assert not any("rasa" in n and p.numel() > 0 for n, p in m.named_parameters())
    x1, x2, x3 = torch.randn(4, 1024), torch.randn(4, 1024), torch.randn(4, 1280)
    score, embs = m(x1, x2, x3)
    assert score.shape == (4,)
    assert torch.equal(embs[0], x1) and torch.equal(embs[2], x3)
    aux = m.perdidas_auxiliares(embs, torch.tensor([0., 1., 0., 1.]))
    assert aux["l_orth"].item() == 0.0 and aux["l_reco"].item() == 0.0


def test_con_rasa_tiene_parametros_rasa():
    m = ModeloDeepfake(sin_rasa=False)
    assert any("rasa" in n for n, _ in m.named_parameters())
