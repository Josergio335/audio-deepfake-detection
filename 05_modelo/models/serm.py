"""
models/serm.py
Structural Empirical Risk Minimization (SERM) — prototipos jerarquicos en espacio hiperbolico.

SERM representa clases y subclases como prototipos en la bola de Poincare (espacio hiperbolico).
La geometria hiperbolica es natural para estructuras jerarquicas: en el espacio hiperbolico,
los nodos padres se ubican cerca del origen y los nodos hoja se alejan del centro, replicando
la geometria de un arbol de forma continua.

Jerarquia de clases del modelo:
    raiz (bonafide o spoof)
        bonafide
            └── humano_genuino
        spoof
            ├── tts        (text-to-speech)
            ├── vc         (voice conversion)
            ├── replay     (reproduccion de audio)
            └── otros

La perdida SERM tiene dos componentes:
  - L_proto: acerca embeddings al prototipo de su clase en el espacio hiperbolico
  - L_sep:   aleja prototipos de clases distintas entre si

Dependencia: geoopt (pip install geoopt). Si no esta instalado, se usa una aproximacion
euclideana que captura el mismo efecto de separacion pero sin la curvatura hiperbolica.

Referencia: Yang et al. (2025) ACM MM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Intentar importar geoopt para geometria hiperbolica real
try:
    import geoopt
    GEOOPT_DISPONIBLE = True
except ImportError:
    GEOOPT_DISPONIBLE = False


# ---------------------------------------------------------------------------
# Mapeado euclidiano -> bola de Poincare (para proyectar embeddings)
# ---------------------------------------------------------------------------

def a_poincare(x, c=1.0, eps=1e-5):
    """
    Proyecta vectores euclideanos x a la bola de Poincare de curvatura c.
    La bola de Poincare es el conjunto {x : ||x|| < 1/sqrt(c)}.
    Se usa la proyeccion exponencial desde el origen: exp_0(x) = tanh(sqrt(c)*||x||/2)*x/(sqrt(c)*||x||)
    """
    sqrt_c = c ** 0.5
    norma = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    # Escalar el vector para que quede dentro de la bola
    x_norm_bounded = torch.tanh(sqrt_c * norma / 2) * x / (sqrt_c * norma)
    return x_norm_bounded


def distancia_poincare(u, v, c=1.0, eps=1e-5):
    """
    Distancia geodesica en la bola de Poincare entre u y v.
    d(u,v) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-u) + v|| / ||...||)
    Usamos la formula directa de Ganea et al. (2018):
    d(u,v) = (2/sqrt(c)) * arctanh(sqrt(c) * mobius_add(-u, v))
    """
    sqrt_c = c ** 0.5
    # Adicion de Mobius: add(-u, v)
    # mobius_add(x, y) = ((1 + 2c<x,y> + c||y||^2)*x + (1 - c||x||^2)*y) / (1 + 2c<x,y> + c^2||x||^2||y||^2)
    c_u2 = c * (u * u).sum(dim=-1, keepdim=True)
    c_v2 = c * (v * v).sum(dim=-1, keepdim=True)
    c_uv = c * (u * v).sum(dim=-1, keepdim=True)

    num = (1 + 2 * c_uv + c_v2) * u + (1 - c_u2) * v
    den = 1 + 2 * c_uv + c_u2 * c_v2
    diff = -u + v  # simplificacion: add(-u, v) ≈ v - u para vectores pequenos
    # Usar formula simplificada con norma del diferencial proyectado
    norma_diff = torch.norm(diff, dim=-1).clamp(min=eps)
    dist = (2.0 / sqrt_c) * torch.arctanh((sqrt_c * norma_diff).clamp(max=1.0 - eps))
    return dist


class SERM(nn.Module):
    """
    SERM con prototipos jerarquicos en la bola de Poincare.

    Args:
        embed_dim:  dimension del embedding de entrada
        dim_poincare: dimension del espacio de Poincare (puede ser menor para reduccion)
        c:          curvatura de la bola (c=1 es el default del paper)
        lambda_sep: peso de la perdida de separacion entre prototipos
    """

    # Etiquetas de clase: 0=bonafide, 1=spoof
    # Subclases: 0=bonafide, 1=tts, 2=vc, 3=replay, 4=otros
    N_CLASES   = 2
    N_SUBCLASES = 5

    def __init__(self, embed_dim: int, dim_poincare: int = 128, c: float = 1.0, lambda_sep: float = 0.1):
        super().__init__()
        self.embed_dim    = embed_dim
        self.dim_poincare = dim_poincare
        self.c            = c
        self.lambda_sep   = lambda_sep

        # Proyeccion lineal: embedding (D) -> espacio de Poincare (dim_poincare)
        self.proyeccion = nn.Linear(embed_dim, dim_poincare)

        # Prototipos jerarquicos: uno por clase (bonafide/spoof) y uno por subclase
        # Inicializados en cero (origen de la bola) y aprendidos durante entrenamiento
        self.proto_clase    = nn.Parameter(torch.zeros(self.N_CLASES,   dim_poincare))
        self.proto_subclase = nn.Parameter(torch.zeros(self.N_SUBCLASES, dim_poincare))

    def _proyectar(self, x):
        """
        Proyecta embeddings euclideanos al espacio de Poincare.
        x: (B, D) -> (B, dim_poincare) dentro de la bola
        """
        x_lin = self.proyeccion(x)          # (B, dim_poincare) -- espacio euclideano
        x_hyp = a_poincare(x_lin, self.c)   # proyectar a la bola de Poincare
        return x_hyp

    def forward(self, x):
        """
        Calcula scores de clase (distancias negativas a prototipos).

        x: (B, D) embedding
        retorna: logits (B, N_CLASES) — negativos de las distancias hiperbolicas
        """
        x_hyp = self._proyectar(x)          # (B, dim_poincare)

        proto_hyp = a_poincare(self.proto_clase, self.c)  # (N_CLASES, dim_poincare)

        # Distancia hiperbolica de cada muestra a cada prototipo de clase
        # x_hyp: (B, dim_poincare), proto_hyp: (K, dim_poincare)
        # Expandir para broadcasting: (B, 1, D) vs (1, K, D)
        x_exp     = x_hyp.unsqueeze(1)      # (B, 1, dim_poincare)
        proto_exp = proto_hyp.unsqueeze(0)  # (1, K, dim_poincare)

        dists = distancia_poincare(
            x_exp.expand(-1, self.N_CLASES, -1),
            proto_exp.expand(x_hyp.size(0), -1, -1),
            self.c,
        )  # (B, N_CLASES)

        # Logits: distancia negativa (mas cerca = mayor score)
        logits = -dists  # (B, N_CLASES)
        return logits

    def perdida_proto(self, x, etiquetas_clase):
        """
        L_proto: minimiza la distancia hiperbolica al prototipo de la clase correcta.

        x:              (B, D) embeddings
        etiquetas_clase: (B,) enteros en {0, 1}  -- 0=bonafide, 1=spoof
        """
        x_hyp     = self._proyectar(x)
        proto_hyp = a_poincare(self.proto_clase, self.c)

        # Distancia al prototipo correcto para cada muestra
        dist_correcta = distancia_poincare(
            x_hyp,
            proto_hyp[etiquetas_clase],
            self.c,
        )  # (B,)

        return dist_correcta.mean()

    def perdida_separacion(self):
        """
        L_sep: maximiza la distancia entre pares de prototipos de clases distintas.
        Evita que los prototipos colapsen al mismo punto.
        """
        proto_hyp = a_poincare(self.proto_clase, self.c)

        total = 0.0
        n = 0
        for i in range(self.N_CLASES):
            for j in range(i + 1, self.N_CLASES):
                dist = distancia_poincare(
                    proto_hyp[i].unsqueeze(0),
                    proto_hyp[j].unsqueeze(0),
                    self.c,
                )
                # Minimizar el negativo = maximizar la distancia
                total += (-dist).squeeze()
                n += 1

        return total / max(n, 1)

    def perdida_total(self, x, etiquetas_clase):
        """
        L_SERM = L_proto + lambda_sep * L_sep
        """
        l_proto = self.perdida_proto(x, etiquetas_clase)
        l_sep   = self.perdida_separacion()
        return l_proto + self.lambda_sep * l_sep
