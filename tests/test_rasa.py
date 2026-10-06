"""
Pruebas unitarias del algoritmo de alineacion de estilo RASA (R6).
Datos sinteticos con la escala real de los embeddings (media ~1e-4, sigma ~0.07).
"""

import pytest
import torch

from models.rasa import RASA

D = 64
M = 16


def embeddings(b=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    escala = 0.05 + 0.1 * torch.rand(b, 1, generator=g)
    centro = 1e-4 * torch.randn(b, 1, generator=g)
    return centro + escala * torch.randn(b, D, generator=g)


def test_forma_de_salida():
    rasa = RASA(D, M=M).train()
    x = embeddings(32)
    assert rasa(x).shape == x.shape


def test_banco_se_inicializa_en_el_primer_lote_y_se_diferencia():
    rasa = RASA(D, M=M).train()
    assert rasa.inicializado.item() == 0
    assert rasa.mu_bank.std().item() == 0 or rasa.sigma_bank.std().item() == 0  # simetrico al inicio
    rasa(embeddings(256))
    assert rasa.inicializado.item() == 1
    assert rasa.mu_bank.std().item() > 0.1
    assert rasa.sigma_bank.std().item() > 0.1


def test_en_eval_no_inicializa_ni_modifica_el_banco():
    rasa = RASA(D, M=M).eval()
    antes = rasa.mu_bank.clone()
    rasa(embeddings(64))
    assert rasa.inicializado.item() == 0
    assert torch.equal(rasa.mu_bank, antes)


def test_alineacion_adain_toma_media_y_desviacion_del_estilo_de_reemplazo():
    rasa = RASA(D, M=M).train()
    x = embeddings(128)
    y = rasa(x)
    with torch.no_grad():
        mu_n, sigma_n = rasa._extraer_estilo(x)
        z_mu, z_sigma = rasa._a_z(mu_n, sigma_n)
        zt_mu, zt_sigma = rasa._proyectar_z(z_mu, z_sigma)
        mu_t, sigma_t = rasa._de_z(zt_mu, zt_sigma)
    assert torch.allclose(y.mean(dim=1, keepdim=True), mu_t, atol=1e-5)
    assert torch.allclose(y.std(dim=1, keepdim=True), sigma_t, rtol=1e-2, atol=1e-5)


def test_perdida_de_separacion_vale_uno_si_el_banco_esta_colapsado():
    rasa = RASA(D, M=M)
    rasa.mu_bank.data.zero_()
    rasa.sigma_bank.data.fill_(1.0)
    assert rasa.orthogonal_loss().item() == pytest.approx(1.0, abs=1e-6)


def test_perdida_de_separacion_baja_si_el_banco_esta_repartido():
    rasa = RASA(D, M=M)
    rasa.mu_bank.data = torch.linspace(-4, 4, M)
    rasa.sigma_bank.data = torch.linspace(4, -4, M)
    assert rasa.orthogonal_loss().item() < 0.2


def test_perdida_de_reconstruccion_no_negativa():
    rasa = RASA(D, M=M).train()
    rasa(embeddings(128))
    assert rasa.reconstruction_loss().item() >= 0


def test_reconstruccion_cercana_a_cero_si_el_banco_cubre_los_estilos():
    rasa = RASA(D, M=64).train()
    x = embeddings(64)  # M >= B: cada par se inicializa con un estilo real
    rasa(x)
    rasa.eval()
    with torch.no_grad():
        y = rasa(x)
    rasa.train()
    rasa(x)
    assert rasa.reconstruction_loss().item() < 0.5
    assert torch.isfinite(y).all()


def test_banco_no_colapsa_con_pesos_auxiliares_altos():
    torch.manual_seed(0)
    rasa = RASA(D, M=M).train()
    opt = torch.optim.Adam(rasa.parameters(), lr=1e-3)
    x = embeddings(256)
    for _ in range(200):
        opt.zero_grad()
        y = rasa(x)
        perdida = y.pow(2).mean() + 1.0 * rasa.orthogonal_loss() + 1.0 * rasa.reconstruction_loss()
        perdida.backward()
        opt.step()
    assert rasa.mu_bank.std().item() > 0.5
    assert rasa.sigma_bank.std().item() > 0.5
    assert rasa.orthogonal_loss().item() < 0.5


def test_carga_checkpoint_antiguo_sin_centro_ni_escala():
    rasa = RASA(D, M=M)
    antiguo = {"mu_bank": torch.randn(M), "sigma_bank": torch.rand(M) + 0.5}
    rasa.load_state_dict(antiguo)  # no debe lanzar error
    assert rasa.inicializado.item() == 1
    assert rasa.escala_mu.item() == 1.0
    assert torch.equal(rasa.mu_bank, antiguo["mu_bank"])


def test_estado_guardado_y_cargado_conserva_el_banco():
    a = RASA(D, M=M).train()
    a(embeddings(128))
    b = RASA(D, M=M)
    b.load_state_dict(a.state_dict())
    a.eval(); b.eval()
    x = embeddings(16, seed=3)
    assert torch.allclose(a(x), b(x))
