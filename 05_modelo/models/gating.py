"""
models/gating.py
Gating Network — calcula pesos adaptativos por muestra para los 3 streams del modelo.

El Gating Network toma como entrada la concatenacion de los 3 embeddings (uno por FM)
y produce un vector de 3 pesos que suman 1 (softmax). Estos pesos determinan cuanto
contribuye cada stream al score final de clasificacion.

La idea: para un audio en ingles con artefactos de TTS, el stream de Whisper
(entrenado en transcripcion) puede ser mas discriminativo que WavLM. El gating
aprende a reconocer este tipo de patron y asigna mayor peso a Whisper en esos casos.

Arquitectura:
    [emb1 || emb2 || emb3]  <- concatenacion de los 3 embeddings (B, D1+D2+D3)
           |
       Linear(D_total, hidden)
       ReLU
       Linear(hidden, 3)
       Softmax
           |
      [g1, g2, g3]          <- pesos adaptativos para los 3 streams

Score final:
    p_final = g1*p1 + g2*p2 + g3*p3

donde p1, p2, p3 son los scores individuales de cada stream.

Referencia: arquitectura propuesta en apuntes de asesoria (Tesis 2, PUCP 2025).
"""

import torch
import torch.nn as nn


class GatingNetwork(nn.Module):
    def __init__(self, dims_entrada: list, hidden_dim: int = 128):
        """
        Args:
            dims_entrada: lista con la dimension de cada stream, ej. [1024, 1024, 1280]
                          para [wav2vec2bert, wavlm, whisper]
            hidden_dim:   dimension de la capa oculta del MLP (defecto: 128)
        """
        super().__init__()
        self.dims_entrada = dims_entrada
        d_total = sum(dims_entrada)  # dimension total de la concatenacion

        # MLP de 2 capas: g = Softmax(W2 * ReLU(W1*P + b1) + b2)
        # Notacion del paper: P = [p1, p2, p3] concatenados
        self.mlp = nn.Sequential(
            nn.Linear(d_total, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(dims_entrada)),
            # Softmax garantiza que los pesos sumen 1 (interpretables como probabilidades)
            nn.Softmax(dim=-1),
        )

    def forward(self, embeddings: list):
        """
        Calcula los pesos adaptativos g para cada stream.

        Args:
            embeddings: lista de tensores [(B, D1), (B, D2), (B, D3)]
        Returns:
            pesos: (B, 3) — pesos que suman 1 por fila
        """
        # Concatenar los 3 embeddings en un solo vector
        concat = torch.cat(embeddings, dim=-1)  # (B, D1+D2+D3)
        pesos  = self.mlp(concat)               # (B, 3)
        return pesos

    def fusionar(self, scores: list, embeddings: list):
        """
        Combina los scores individuales de cada stream ponderados por los pesos del gating.

        Args:
            scores:     lista de tensores [(B,), (B,), (B,)] — score de cada stream
            embeddings: lista de tensores [(B, D1), (B, D2), (B, D3)]
        Returns:
            score_final: (B,) — score ponderado final
        """
        pesos = self.forward(embeddings)  # (B, 3)

        # Apilar scores: (B, 3)
        scores_stack = torch.stack(scores, dim=1)  # (B, 3)

        # Suma ponderada: p_final = g^T * [p1, p2, p3]
        # (B, 3) * (B, 3) elemento a elemento, suma sobre dim=1
        score_final = (pesos * scores_stack).sum(dim=1)  # (B,)
        return score_final
