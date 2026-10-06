"""
RASA v1 (ORIGINAL, ARCHIVADA). No se usa en el entrenamiento.
Esta version inicializaba los 160 pares del style bank con el mismo valor (0, 1);
por simetria el bank nunca se diferencio (ver models/rasa.py para la version corregida).
Se conserva solo como referencia de los entrenamientos del 2026-10-04/05 (exp1/exp2/exp3).
"""

"""
models/rasa.py
Risk-Aware Style Alignment (RASA) — adaptado para embeddings precomputados.

El paper original aplica RASA sobre feature maps de capas superficiales del FM
(forma: B x T x D). Como aqui los embeddings ya estan promediados temporalmente
(forma: B x D), el "estilo" se captura con estadisticas escalares por muestra:
media y desviacion estandar sobre la dimension D.

El style bank aprende M vectores base (mu_b, sigma_b). Para cada embedding:
  1. Extraer estilo de entrada: mu_n = mean(x), sigma_n = std(x)
  2. Calcular similitud con cada vector base del bank
  3. Combinar bases por similitud (suma ponderada)
  4. Aplicar el estilo proyectado al embedding (AdaIN)

Perdidas adicionales durante entrenamiento:
  - L_orth: fuerza que los vectores del bank sean mutuamente ortogonales
            (maximiza la diversidad del espacio de estilos)
  - L_reco: fuerza que el bank pueda reconstruir los estilos originales
            (el bank cubre todo el espacio de estilos de la fuente)

Referencia: Yang et al. (2025) "Generalizable Audio Deepfake Detection via
Risk-Aware Style Alignment and Structural Empirical Risk Minimization". ACM MM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RASA(nn.Module):
    def __init__(self, embed_dim: int, M: int = 160):
        """
        Args:
            embed_dim: dimension del embedding de entrada (D)
            M: numero de vectores base en el style bank (paper usa M=160)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.M = M

        # Style bank: M pares (mu_b, sigma_b) aprendibles
        # mu_b y sigma_b son escalares — representan media y std de referencia
        self.mu_bank    = nn.Parameter(torch.zeros(M))       # (M,)
        self.sigma_bank = nn.Parameter(torch.ones(M))        # (M,) — inicializar en 1 para estabilidad

    def _extraer_estilo(self, x):
        """
        Extrae estadisticas de estilo del embedding.
        x: (B, D) -> mu: (B, 1), sigma: (B, 1)
        """
        mu    = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + 1e-6  # epsilon para evitar division por cero
        return mu, sigma

    def _proyectar_estilo(self, mu_n, sigma_n):
        """
        Proyecta el estilo de entrada al espacio del style bank.
        Calcula similitud con cada base y genera el estilo proyectado
        como combinacion ponderada.

        mu_n:    (B, 1)
        sigma_n: (B, 1)
        retorna: mu_tilde (B, 1), sigma_tilde (B, 1)
        """
        # Distancia al cuadrado con cada vector base — (B, M)
        d_mu    = (mu_n    - self.mu_bank.unsqueeze(0)) ** 2
        d_sigma = (sigma_n - self.sigma_bank.unsqueeze(0)) ** 2
        d       = d_mu + d_sigma  # (B, M)

        # Pesos: mas cercano = mas peso (softmax sobre distancia negativa)
        lam = torch.softmax(-d, dim=1)  # (B, M)  -- lambda en el paper

        # Estilo proyectado: suma ponderada de vectores base
        mu_tilde    = (lam * self.mu_bank.unsqueeze(0)).sum(dim=1, keepdim=True)    # (B, 1)
        sigma_tilde = (lam * self.sigma_bank.unsqueeze(0)).sum(dim=1, keepdim=True) # (B, 1)
        # Asegurar sigma positiva
        sigma_tilde = sigma_tilde.abs() + 1e-6

        return mu_tilde, sigma_tilde

    def forward(self, x):
        """
        Alinea el estilo del embedding al espacio del style bank.

        x: (B, D) embedding crudo
        retorna: (B, D) embedding con estilo alineado
        """
        mu_n, sigma_n       = self._extraer_estilo(x)
        mu_tilde, sigma_tilde = self._proyectar_estilo(mu_n, sigma_n)

        # Normalizacion de instancia + aplicacion del estilo proyectado (AdaIN)
        # F_tilde = sigma_tilde * (x - mu_n) / sigma_n + mu_tilde
        x_norm    = (x - mu_n) / sigma_n
        x_aligned = sigma_tilde * x_norm + mu_tilde

        return x_aligned

    def orthogonal_loss(self):
        """
        L_orth: penaliza similitudes entre pares de vectores base.
        Fuerza que el bank cubra un espacio de estilos diverso y no redundante.
        """
        # Normalizar vectores para coseno
        mu_n    = F.normalize(self.mu_bank.unsqueeze(1),    dim=0)  # (M, 1)
        sigma_n = F.normalize(self.sigma_bank.unsqueeze(1), dim=0)  # (M, 1)

        # Similitud coseno entre todos los pares — (M, M)
        cos_mu    = (mu_n    @ mu_n.T)
        cos_sigma = (sigma_n @ sigma_n.T)

        # Solo penalizar pares distintos (off-diagonal)
        mask = ~torch.eye(self.M, dtype=torch.bool, device=self.mu_bank.device)
        loss = cos_mu[mask].abs().mean() + cos_sigma[mask].abs().mean()
        return loss

    def reconstruction_loss(self, x):
        """
        L_reco: maximiza similitud entre el estilo original y el proyectado.
        Asegura que el bank pueda reconstruir cualquier estilo de la fuente.
        """
        mu_n, sigma_n         = self._extraer_estilo(x)
        mu_tilde, sigma_tilde = self._proyectar_estilo(mu_n, sigma_n)

        # Similitud coseno entre original y proyectado
        cos_mu    = F.cosine_similarity(mu_n,    mu_tilde,    dim=1)   # (B,)
        cos_sigma = F.cosine_similarity(sigma_n, sigma_tilde, dim=1)   # (B,)

        # Maximizar similitud = minimizar negativo
        loss = -cos_mu.mean() - cos_sigma.mean()
        return loss
