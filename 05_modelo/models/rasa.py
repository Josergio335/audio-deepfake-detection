"""
models/rasa.py
Risk-Aware Style Alignment (RASA) — adaptado para embeddings precomputados.

El paper original aplica RASA sobre feature maps de capas superficiales del FM
(forma: B x T x D). Como aqui los embeddings ya estan promediados temporalmente
(forma: B x D), el "estilo" se captura con estadisticas escalares por muestra:
media y desviacion estandar sobre la dimension D.

El style bank aprende M pares base. Para cada embedding:
  1. Extraer estilo de entrada: mu_n = mean(x), sigma_n = std(x)
  2. Calcular la cercania del estilo de entrada a cada par base del bank
  3. Combinar los pares base segun esa cercania (suma ponderada, softmax)
  4. Aplicar el estilo proyectado al embedding (AdaIN)

Cambios respecto a la version 1 (que dejaba el bank colapsado):
  - Inicializacion: la version 1 iniciaba los M pares con el mismo valor (0, 1);
    por simetria todos recibian el mismo gradiente y nunca se diferenciaban.
    Ahora, en el primer lote de entrenamiento, el bank se inicializa con estilos
    REALES muestreados de ese lote (mas un ruido pequeno).
  - Parametrizacion estandarizada: los parametros del bank (mu_bank, sigma_bank)
    se guardan en UNIDADES DE DISPERSION de los estilos de cada stream, respecto
    a un centro fijado al inicializar. Los estilos reales son numeros diminutos
    (media ~1e-4, sigma ~0.05-0.5) y el optimizador mueve cada parametro ~lr por
    paso, asi que con valores crudos el bank se alejaba de los datos en pocos pasos.
    Estilo de referencia real = centro + escala * parametro.
  - Distancias en esas mismas unidades (softmax con escala adecuada).
  - L_reco: error cuadratico entre el estilo original y el proyectado (en
    unidades estandarizadas), calculado sobre el embedding de ENTRADA.
  - L_orth: repulsion entre los pares del bank (penaliza que dos pares queden
    muy cerca) para mantener su diversidad.

Compatibilidad: los checkpoints de la version 1 y de la primera version corregida
(sin centro/escala/bandera) se siguen cargando con centro 0 y escala 1.

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
            M: numero de pares base en el style bank (paper usa M=160)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.M = M

        # Style bank en unidades estandarizadas (ver docstring del modulo).
        # Estos valores son solo de partida: se reemplazan al inicializar el bank
        # con estilos reales en el primer lote de entrenamiento.
        self.mu_bank    = nn.Parameter(torch.zeros(M))       # (M,)
        self.sigma_bank = nn.Parameter(torch.ones(M))        # (M,)

        # Centro y escala del espacio de estilo (se fijan al inicializar el bank)
        # y bandera de inicializacion. Son buffers para guardarse en el checkpoint.
        self.register_buffer("centro_mu",    torch.zeros(()))
        self.register_buffer("centro_sigma", torch.zeros(()))
        self.register_buffer("escala_mu",    torch.ones(()))
        self.register_buffer("escala_sigma", torch.ones(()))
        self.register_buffer("inicializado", torch.zeros(()))

        # Ultimo L_reco calculado en forward (solo en entrenamiento)
        self._reco = None

    # ------------------------------------------------------------------
    # Compatibilidad con checkpoints anteriores
    # ------------------------------------------------------------------
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Un checkpoint anterior no trae centro, escala ni bandera: se asume centro 0,
        # escala 1 (mismo comportamiento que entonces) y bank ya inicializado.
        for nombre, valor in (("centro_mu", 0.0), ("centro_sigma", 0.0),
                              ("escala_mu", 1.0), ("escala_sigma", 1.0),
                              ("inicializado", 1.0)):
            clave = prefix + nombre
            if clave not in state_dict:
                state_dict[clave] = torch.tensor(valor)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ------------------------------------------------------------------
    # Estilo e inicializacion
    # ------------------------------------------------------------------
    def _extraer_estilo(self, x):
        """
        Extrae estadisticas de estilo del embedding.
        x: (B, D) -> mu: (B, 1), sigma: (B, 1)
        """
        mu    = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + 1e-6  # epsilon para evitar division por cero
        return mu, sigma

    def _a_z(self, mu_n, sigma_n):
        """Estilo en unidades estandarizadas (respecto al centro y la escala del bank)."""
        return ((mu_n    - self.centro_mu)    / self.escala_mu,
                (sigma_n - self.centro_sigma) / self.escala_sigma)

    @torch.no_grad()
    def inicializar_banco(self, x):
        """
        Inicializa el bank con estilos reales de un lote (rompe la simetria) y
        fija el centro y la escala del espacio de estilo. x: (B, D).
        """
        mu, sigma = self._extraer_estilo(x)
        mu, sigma = mu.squeeze(1), sigma.squeeze(1)
        B = mu.shape[0]
        if B < 2:
            return

        self.centro_mu.fill_(float(mu.mean()))
        self.centro_sigma.fill_(float(sigma.mean()))
        self.escala_mu.fill_(float(mu.std().clamp_min(1e-8)))
        self.escala_sigma.fill_(float(sigma.std().clamp_min(1e-8)))
        self.inicializado.fill_(1.0)

        z_mu, z_sigma = self._a_z(mu, sigma)
        if B >= self.M:
            idx = torch.randperm(B, device=mu.device)[: self.M]
        else:
            idx = torch.randint(0, B, (self.M,), device=mu.device)
        ruido = 0.05  # en unidades de dispersion de los estilos
        self.mu_bank.copy_(z_mu[idx] + ruido * torch.randn(self.M, device=mu.device))
        self.sigma_bank.copy_(z_sigma[idx] + ruido * torch.randn(self.M, device=mu.device))

    # ------------------------------------------------------------------
    # Proyeccion y alineacion
    # ------------------------------------------------------------------
    def _proyectar_z(self, z_mu, z_sigma):
        """
        Proyecta el estilo (en unidades estandarizadas) al espacio del style bank.
        Pesos por cercania (softmax sobre la distancia al cuadrado negativa).
        z_mu, z_sigma: (B, 1) -> zt_mu, zt_sigma: (B, 1)
        """
        d = (z_mu    - self.mu_bank.unsqueeze(0))    ** 2 \
          + (z_sigma - self.sigma_bank.unsqueeze(0)) ** 2     # (B, M)
        lam = torch.softmax(-d, dim=1)                         # (B, M)  -- lambda en el paper
        zt_mu    = (lam * self.mu_bank.unsqueeze(0)).sum(dim=1, keepdim=True)
        zt_sigma = (lam * self.sigma_bank.unsqueeze(0)).sum(dim=1, keepdim=True)
        return zt_mu, zt_sigma

    def _de_z(self, zt_mu, zt_sigma):
        """Vuelve a las unidades originales del embedding."""
        mu_tilde    = self.centro_mu + self.escala_mu * zt_mu
        sigma_tilde = (self.centro_sigma + self.escala_sigma * zt_sigma).abs() + 1e-6  # sigma positiva
        return mu_tilde, sigma_tilde

    def forward(self, x):
        """
        Alinea el estilo del embedding al espacio del style bank.

        x: (B, D) embedding crudo
        retorna: (B, D) embedding con estilo alineado
        """
        # Primer lote de entrenamiento: inicializar el bank con estilos reales
        if self.training and self.inicializado.item() == 0:
            self.inicializar_banco(x.detach())

        mu_n, sigma_n = self._extraer_estilo(x)
        z_mu, z_sigma = self._a_z(mu_n, sigma_n)
        zt_mu, zt_sigma = self._proyectar_z(z_mu, z_sigma)
        mu_tilde, sigma_tilde = self._de_z(zt_mu, zt_sigma)

        if self.training:
            # L_reco se calcula aqui, con el embedding de ENTRADA
            self._reco = ((z_mu - zt_mu) ** 2 + (z_sigma - zt_sigma) ** 2).mean()

        # Normalizacion de instancia + aplicacion del estilo proyectado (AdaIN)
        # F_tilde = sigma_tilde * (x - mu_n) / sigma_n + mu_tilde
        x_norm    = (x - mu_n) / sigma_n
        x_aligned = sigma_tilde * x_norm + mu_tilde

        return x_aligned

    # ------------------------------------------------------------------
    # Perdidas auxiliares
    # ------------------------------------------------------------------
    def orthogonal_loss(self):
        """
        L_orth: repulsion entre los pares base. Penaliza que dos pares del bank
        queden muy cerca en el espacio de estilo (unidades estandarizadas), para
        que el bank cubra un espacio de estilos diverso y no redundante.
        Rango (0, 1]; vale ~1 si todos los pares coinciden (bank colapsado).
        """
        pm, ps = self.mu_bank, self.sigma_bank
        d2 = (pm.unsqueeze(1) - pm.unsqueeze(0)) ** 2 + (ps.unsqueeze(1) - ps.unsqueeze(0)) ** 2  # (M, M)
        mask = ~torch.eye(self.M, dtype=torch.bool, device=self.mu_bank.device)
        return torch.exp(-d2)[mask].mean()

    def reconstruction_loss(self, x=None):
        """
        L_reco: error entre el estilo original y el proyectado (unidades
        estandarizadas). Asegura que el bank pueda reconstruir los estilos de la fuente.

        Devuelve el valor calculado en el ultimo forward de entrenamiento, que usa
        el embedding de ENTRADA. El argumento x (el embedding alineado, como lo
        pasa train.py) se ignora si ya hay un valor calculado; si no lo hay (uso
        aislado, p. ej. en pruebas), se calcula a partir de x.
        """
        if self._reco is not None:
            return self._reco
        if x is None:
            raise RuntimeError("reconstruction_loss: no hay forward previo y no se paso x")
        mu_n, sigma_n   = self._extraer_estilo(x)
        z_mu, z_sigma   = self._a_z(mu_n, sigma_n)
        zt_mu, zt_sigma = self._proyectar_z(z_mu, z_sigma)
        return ((z_mu - zt_mu) ** 2 + (z_sigma - zt_sigma) ** 2).mean()
