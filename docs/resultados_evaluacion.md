# Resultados de evaluación — Detección de audio deepfake multilingüe

Actualizado: 2026-11-22. Resultados de los experimentos Exp-1 a Exp-4 (Wav2Vec2BERT + WavLM Large + Whisper Large v3, RASA, red de compuertas y SERM) sobre los escenarios de prueba descritos en la tesis (en distribución, zero-shot, fuera de dominio y compresión), con las ablaciones de RASA y de M y la comparación con líneas base simplificadas. Las rutas con `~/` corresponden al servidor de entrenamiento.

> **Nota sobre las etiquetas del escenario fuera de dominio.** La regla de etiquetado de SONAR marcaba como `bonafide` 7,763 audios sintéticos (los 6 vocoders de LibriSeVoc, 6,698, y la carpeta `spoof` de In_the_Wild, 1,065). Se corrigió la regla (`etiqueta_sonar` en `05_modelo/generar_manifests_test.py`), el índice de prueba (`05_modelo/corregir_etiquetas_fueradominio.py`) y se reevaluaron todos los checkpoints; todas las cifras de este documento usan las etiquetas corregidas. Una prueba automática (`tests/test_etiquetas_test.py`) verifica la regla.

---

## 1. Modelos evaluados

### Experimentos principales (M=160 fijo)

| Checkpoint | Exp | λ | Notas |
|---|---|---|---|
| exp1_20261005_001612 | 1 | 0.01 | No documentado en la versión anterior |
| exp1_20261006_002842 | 1 | 0.1 | |
| exp1_20261006_002845 | 1 | 1.0 | |
| exp1_20261006_000107 | 1 | 0.01 | |
| exp2_20261005_001622 | 2 | 0.01 | RASA v1 (bank colapsado) |
| exp2_20261006_082449 | 2 | 1.0 | **RUN ANÓMALO** en in-distribución |
| exp2_20261115_223104 | 2 | 1.0 | Reentrenado (run correcto) |
| exp3_20261004_230751 | 3 | 0.01 | RASA v1 (bank colapsado) |
| exp3_20261006_082449 | 3 | 1.0 | Corrida original del Exp-3 |

### Ablación de M (Exp3, λ=1.0)

| Checkpoint | M |
|---|---|
| ablacion_M16_20261111_101359 | 16 |
| ablacion_M32_20261111_101359 | 32 |
| ablacion_M64_20261111_101359 | 64 |
| ablacion_M128_20261111_101359 | 128 |
| ablacion_M160_20261111_101359 | 160 |
| ablacion_M256_20261111_101359 | 256 |

### Experimento 4 (Exp-4): bonafide de Common Voice en el entrenamiento

Mismo modelo (M=160), pero el corpus incorpora 9 idiomas de Common Voice (ja, ko, hi, id, ar, he, bg, cs, fi) y se limita cada idioma y clase a un tope de archivos (1,800 para los 7 idiomas originales y 2,400 para los nuevos; el tope es por número de archivos como aproximación a ~3 h, la duración real no se midió). Manifest generado con `05_modelo/generar_manifest_exp4.py`.

| Variante | Checkpoint | λ_orth / λ_reco / λ_serm |
|---|---|---|
| **A** | exp4_20261118_225104 | 0.01 / 0.01 / 0.05 (valores por defecto de `train.py`) |
| B | exp4_20261119_100541 | 1.0 / 1.0 / 1.0 |
| C | exp4_20261119_113340 | 1.0 / 1.0 / 0.05 (mismos λ que el Exp-3) |

La variante A tiene 4 corridas (la original y 3 repeticiones, ver 2.7); B y C, una cada una.

---

## 2. Resultados por escenario

### 2.1 In-distribución (ASVspoof2019 LA, n=96,070)

| Modelo | Exp | λ | M | Acc | AUROC | EER | minDCF | Notas |
|---|---|---|---|---|---|---|---|---|
| ablacion_M256 | 3 | 1.0 | 256 | 99.27% | 0.9993 | **0.98%** | 0.0633 | |
| exp3_20261006_082449 | 3 | 1.0 | 160 | 99.15% | 0.9990 | 1.06% | 0.0701 | Corrida original del Exp-3 |
| ablacion_M16 | 3 | 1.0 | 16 | 99.20% | 0.9992 | 1.01% | 0.0854 | |
| exp2_20261115_223104 | 2 | 1.0 | 160 | 99.11% | 0.9991 | 1.07% | 0.0606 | |
| exp1_20261006_002842 | 1 | 0.1 | 160 | 99.07% | 0.9989 | 1.23% | 0.0778 | |
| exp1_20261006_002845 | 1 | 1.0 | 160 | 98.72% | 0.9981 | 1.56% | 0.0861 | |
| exp1_20261006_000107 | 1 | 0.01 | 160 | 98.70% | 0.9985 | 1.52% | 0.1065 | |
| exp3_20261004_230751 | 3 | 0.01 | 160 | 98.87% | 0.9984 | 1.51% | 0.1777 | RASA v1 |
| exp2_20261005_001622 | 2 | 0.01 | 160 | 98.87% | 0.9985 | 1.59% | 0.1306 | RASA v1 |
| ablacion_M160 | 3 | 1.0 | 160 | 98.81% | 0.9983 | 1.72% | 0.1866 | |
| exp4 A | 4 | 0.01 | 160 | 95.64% | 0.9836 | 6.17% | 0.3672 | |
| exp4 C | 4 | 1.0 | 160 | 82.58% | 0.7912 | 27.72% | 0.9690 | |
| exp4 B | 4 | 1.0 | 160 | 75.88% | 0.7419 | 31.59% | 0.9867 | |
| exp2_20261006_082449 | 2 | 1.0 | 160 | 84.82% | 0.8176 | 24.25% | 0.9988 | Run anómalo |

### 2.2 Zero-shot, solo spoof (MLAAD, 33 idiomas no vistos, n=58,485)

> ⚠️ MLAAD es 100% audio sintético: no hay bonafide. AUROC=NaN y EER=0 son triviales y la accuracy mide únicamente la detección de spoof. Esta versión del escenario **no es evidencia de generalización** (ver 2.4).

| Modelo | Exp | λ | M | Acc (detección de spoof) |
|---|---|---|---|---|
| exp2_20261115_223104 | 2 | 1.0 | 160 | 99.56% |
| exp3_20261004_230751 | 3 | 0.01 | 160 | 99.66% |
| exp2_20261005_001622 | 2 | 0.01 | 160 | 99.67% |
| exp3_20261006_082449 | 3 | 1.0 | 160 | 99.08% |
| ablacion_M16 | 3 | 1.0 | 16 | 99.41% |
| ablacion_M256 | 3 | 1.0 | 256 | 99.41% |
| exp1_20261006_002842 | 1 | 0.1 | 160 | 99.35% |
| exp1_20261006_002845 | 1 | 1.0 | 160 | 99.27% |
| exp1_20261006_000107 | 1 | 0.01 | 160 | 98.90% |
| exp2_20261006_082449 | 2 | 1.0 | 160 | 91.70% |

### 2.3 Fuera de dominio (SONAR + SpoofCeleb, n=31,417), etiquetas corregidas

Composición real tras la corrección: 4,053 bonafide (LibriSeVoc `gt` 1,143, In_the_Wild `bona-fide` 1,629, LJSpeech 1,105, SONAR_core 176) y 27,364 spoof (SpoofCeleb 19,469, vocoders de LibriSeVoc 6,698, In_the_Wild `spoof` 1,062, SONAR_core 130, generated_audio 5).

| Modelo | Exp | λ | M | Acc | AUROC | EER | minDCF | Notas |
|---|---|---|---|---|---|---|---|---|
| **exp4 A** (exp4_20261118_225104) | 4 | 0.01 | 160 | 92.67% | 0.9556 | **10.71%** | 2.2452 | |
| ablacion_M160 | 3 | 1.0 | 160 | 88.90% | 0.9346 | 13.57% | 3.6609 | |
| exp4 C (exp4_20261119_113340) | 4 | 1.0 | 160 | 83.45% | 0.9195 | 16.18% | 0.9193 | |
| exp1_20261005_001612 | 1 | 0.01 | 160 | 88.69% | 0.9031 | 16.71% | 0.9933 | |
| exp1_20261006_000107 | 1 | 0.01 | 160 | 87.99% | 0.8947 | 16.98% | 12.1986 | |
| exp3_20261006_082449 | 3 | 1.0 | 160 | 87.52% | 0.8774 | 18.60% | 10.1550 | Corrida original del Exp-3 |
| ablacion_M256 | 3 | 1.0 | 256 | 87.14% | 0.8534 | 21.10% | 8.5944 | |
| ablacion_M128 | 3 | 1.0 | 128 | 87.35% | 0.8468 | 22.06% | 9.2722 | |
| exp2_20261006_082449 | 2 | 1.0 | 160 | 85.70% | 0.8408 | 22.43% | 1.0000 | Run anómalo en in-dist. |
| ablacion_M64 | 3 | 1.0 | 64 | 85.89% | 0.8410 | 22.55% | 0.9996 | |
| ablacion_M32 | 3 | 1.0 | 32 | 85.93% | 0.8408 | 22.60% | 0.9999 | |
| exp4 B (exp4_20261119_100541) | 4 | 1.0 | 160 | 74.41% | 0.8533 | 22.97% | 1.0405 | |
| exp2_20261115_223104 | 2 | 1.0 | 160 | 86.95% | 0.8189 | 24.32% | 11.1797 | |
| exp2_20261005_001622 | 2 | 0.01 | 160 | 88.38% | 0.8310 | 25.14% | 0.9795 | RASA v1 |
| exp1_20261006_002845 | 1 | 1.0 | 160 | 86.57% | 0.7971 | 26.17% | 12.2945 | |
| exp1_20261006_002842 | 1 | 0.1 | 160 | 86.53% | 0.7992 | 26.52% | 8.2297 | |
| ablacion_M16 | 3 | 1.0 | 16 | 87.13% | 0.7976 | 27.24% | 8.1958 | |
| exp3_20261004_230751 | 3 | 0.01 | 160 | 87.62% | 0.5924 | 40.83% | 18.5700 | RASA v1 |

**Desglose por subconjunto** (`05_modelo/desglose_subconjuntos.py`; umbral 0.5). `det_spoof` es el porcentaje de spoof detectado y `falsa_alarma` el porcentaje de bonafide marcado como spoof:

| Subconjunto | n bonafide / spoof | det_spoof Exp-3 M160 → A | falsa_alarma Exp-3 M160 → A |
|---|---|---|---|
| SpoofCeleb | 0 / 19,469 | 99.6% → 99.3% | — |
| SONAR/LibriSeVoc | 1,143 / 6,698 | 99.9% → 97.5% | 84.1% → 45.2% |
| SONAR/In_the_Wild | 1,629 / 1,062 | 99.6% → 96.5% | 80.7% → 70.7% |
| SONAR/LJSpeech | 1,105 / 0 | — | 90.9% → 18.8% |
| SONAR/SONAR_core | 176 / 130 | 100.0% → 90.8% | 73.3% → 38.6% |
| SONAR/generated_audio | 0 / 5 | 100% → 100% (n=5, no concluyente) | — |

La detección de spoof es casi perfecta en ambos modelos; el punto débil es el bonafide fuera de dominio, que se marca como spoof. El Exp-4 lo reduce pero no lo elimina (In_the_Wild sigue en 70.7%). El AUROC por subconjunto (0.84 a 0.99) indica que el ranking es razonable y que el umbral 0.5, calibrado en distribución, no se traslada: para este escenario conviene reportar EER y AUROC por encima de la accuracy.

### 2.4 Zero-shot con bonafide de Common Voice

Bonafide de Common Voice (18 idiomas) más spoof de MLAAD. Los idiomas de Common Voice no estaban en el entrenamiento de Exp-1/2/3.

**`zeroshot_v2`** (n=100,850; 18 idiomas):

| Modelo | Acc | Precision | Recall | AUROC | EER |
|---|---|---|---|---|---|
| ablacion_M160 (Exp-3) | 60.08% | 59.38% | 98.64% | 0.6238 | 40.65% |
| ablacion_M256 (Exp-3) | 58.77% | 58.51% | 99.41% | 0.5572 | 45.19% |

El modelo detecta el spoof (recall ~99%) pero clasifica como spoof el bonafide de Common Voice: el bonafide de entrenamiento (M-AILABS, estudio, 7 idiomas europeos) define una frontera atada a ese dominio de grabación.

**`zeroshot_exp4`** (n=60,546; los 9 idiomas de Common Voice que no entran al entrenamiento del Exp-4: ro, nl, lt, hu, et, fa, da, el, pt):

| Modelo | Acc | Precision | Recall | F1 | AUROC | EER | minDCF |
|---|---|---|---|---|---|---|---|
| ablacion_M160 (Exp-3) | 67.00% | 66.41% | 98.93% | 79.47% | 0.6496 | 39.07% | 80.4147 |
| **exp4 A** | 96.42% | 98.03% | 96.39% | 97.20% | 0.9943 | **3.57%** | 0.5065 |
| exp4 C | 93.82% | 97.50% | 92.82% | 95.10% | 0.9863 | 5.77% | 0.5630 |
| exp4 B | 92.35% | 98.01% | 89.99% | 93.83% | 0.9832 | 6.21% | 0.5765 |

### 2.5 Compresión (ASVspoof 2019 LA eval recomprimido, MP3 32 kbps, n=96,081)

Solo se generó la variante MP3 32 kbps (no se generaron MP3 64/128 ni AAC).

| Modelo | Acc | Precision | Recall | F1 | AUROC | EER | minDCF |
|---|---|---|---|---|---|---|---|
| ablacion_M160 (Exp-3) | 93.92% | 93.74% | 99.89% | 96.72% | 0.9856 | **5.97%** | 0.5534 |
| exp4 A | 89.39% | 96.83% | 91.16% | 93.91% | 0.9333 | 13.78% | 0.5825 |
| exp4 C | 76.51% | 92.89% | 79.93% | 85.92% | 0.7223 | 32.99% | 0.9994 |
| exp4 B | 56.95% | 95.53% | 54.56% | 69.45% | 0.6951 | 35.63% | 0.9810 |

### 2.6 Resumen Exp-3 M=160 frente a Exp-4 A

| Escenario | Exp-3 M160: EER / AUROC | Exp-4 A: EER / AUROC |
|---|---|---|
| En distribución | **1.72%** / 0.9983 | 6.17% / 0.9836 |
| Fuera de dominio | 13.57% / 0.9346 | **10.71%** / 0.9556 |
| Compresión MP3 32 kbps | **5.97%** / 0.9856 | 13.78% / 0.9333 |
| Zero-shot (9 idiomas) | 39.07% / 0.6496 | **3.57%** / 0.9943 |

### 2.7 Repeticiones de la variante A (λ 0.01 / 0.01 / 0.05)

EER (%) de cuatro corridas con la misma configuración:

| Corrida | En distribución | Fuera de dominio | Compresión MP3 | Zero-shot | val_acc en la época elegida |
|---|---|---|---|---|---|
| exp4_20261118_225104 (original) | 6.17 | 10.71 | 13.78 | 3.57 | 98.74% |
| exp4_20261119_125609 | 7.33 | 15.32 | 13.38 | 2.18 | 99.19% |
| exp4_20261119_125614 | **23.09** | **32.59** | 14.16 | **10.82** | **92.67%** |
| exp4_20261119_125620 | 7.26 | 12.90 | 13.53 | 3.91 | 98.64% |
| Media ± sd (4 corridas) | 10.96 ± 8.10 | 17.88 ± 9.99 | 13.71 ± 0.34 | 5.12 ± 3.88 | |
| Media ± sd (3 convergidas) | 6.92 ± 0.65 | 12.98 ± 2.31 | 13.56 ± 0.20 | 3.22 ± 0.92 | |

**Corrida `exp4_20261119_125614`: entrenamiento que no convergió bien.**
- Su validación en la época elegida (val_bce 0.2050, val_acc 92.67%) es muy peor que la de la corrida original (0.0345, 98.74%).
- No es un problema de selección del checkpoint: el mejor por `val_total` y el mejor por `val_bce` caen en las épocas 99 y 100.
- El fallo está en las dos particiones de ASVspoof (EER 18.6% en `dev` y 24.6% en `eval`), con 63 a 67% de falsa alarma en bonafide frente a 16 a 21% de la corrida original (umbral 0.5).
- La validación la distingue sin mirar el test, por lo que se reporta como corrida fallida según un criterio de validación y se muestran las cuatro corridas.
- Sin explicar: en esta corrida el EER bajo compresión MP3 (14.16%) es menor que en distribución (23.09%); en las otras tres es el doble.

### 2.8 Repeticiones del Exp-3 (M=160, λ 1.0 / 1.0 / 0.05, `train.py`)

EER (%):

| Corrida | En distribución | Fuera de dominio | Compresión MP3 | Zero-shot | val_acc en la época elegida |
|---|---|---|---|---|---|
| exp3_20261119_143040 (rep1) | 0.97 | 20.56 | 3.82 | 42.95 | 98.96% |
| exp3_20261119_144411 (rep2) | 1.04 | 17.69 | 4.08 | 43.31 | 98.91% |
| exp3_20261119_144355 (rep3) | **23.69** | 22.50 | **53.36** | 50.59 | **91.80%** |
| ablacion_M160 (referencia, `train_ablacion_M.py`) | 1.72 | 13.57 | 5.97 | 39.07 | 98.87% |

**El fallo de convergencia no es propio del Exp-4.** Una de las tres repeticiones del Exp-3 también quedó con una validación mucho peor (val_bce 0.2421, val_acc 91.80%), igual que una de las cuatro corridas de la variante A (val_bce 0.2050, val_acc 92.67%). Todas las corridas convergidas tienen val_acc ≥ 98.64% y val_bce ≤ 0.037, así que un criterio de validación (val_acc < 95% en la época elegida) separa sin ambigüedad las corridas fallidas. No hay patrón de dispositivo: la repetición fallida del Exp-3 corrió en la GPU 1, igual que la repetición 2, que sí convergió. El criterio se fijó después de ver los resultados; la separación entre grupos es amplia (92.7% frente a ≥ 98.6%).

Media ± desviación estándar (EER %, solo corridas convergidas; n=2 o 3, por lo que las desviaciones son indicativas):

| Escenario | Exp-3 (rep1 y rep2) | Exp-3 (rep1, rep2 y ablación M160) | Exp-4 A (3 convergidas) |
|---|---|---|---|
| En distribución | 1.01 ± 0.05 | 1.24 ± 0.40 | 6.92 ± 0.65 |
| Fuera de dominio | 19.13 ± 2.03 | 17.27 ± 3.51 | 12.98 ± 2.31 |
| Compresión MP3 32 kbps | 3.95 ± 0.18 | 4.62 ± 1.17 | 13.56 ± 0.20 |
| Zero-shot (9 idiomas) | 43.13 ± 0.25 | 41.78 ± 2.35 | 3.22 ± 0.91 |

### 2.9 Comparación con baselines (R8)

Baselines entrenados por nosotros con el mismo corpus del Exp-4 (`05_modelo/baselines/`: audio truncado a 6 s, semilla 42, una corrida por modelo, checkpoint de menor val_loss). Son **versiones simplificadas**, no las implementaciones oficiales de los modelos publicados, y no se les puede atribuir sus EER. Evaluados sobre los mismos archivos y etiquetas corregidas que nuestros modelos.

| Baseline | Épocas | val_acc |
|---|---|---|
| LFCC-LCNN | 50 | 99.14% |
| RawNet2 simplificado | 30 | 98.34% |
| AASIST simplificado | 30 | 89.93% (no convergió bien: train_acc 89.8%) |

EER (%) con AUROC entre paréntesis:

| Escenario | Exp-4 A (3 convergidas) | Exp-3 (convergidas) | LFCC-LCNN | RawNet2 simpl. | AASIST simpl. |
|---|---|---|---|---|---|
| En distribución (dev+eval) | 6.92 | 1.24 | 19.65 (0.883) | 28.94 (0.788) | 38.91 (0.662) |
| En distribución, solo `eval` | 6.60 ± 0.70 | 1.33 ± 0.40 | 23.15 | 30.09 | 40.12 |
| Fuera de dominio | 12.98 | 17.27 | 70.00 (0.283) | 74.94 (0.227) | 69.68 (0.251) |
| Compresión MP3 32 kbps | 13.56 | 4.62 | 39.22 (0.654) | 29.59 (0.773) | 45.19 (0.570) |
| Zero-shot (9 idiomas) | 3.22 | 41.78 | 3.15 (0.994) | 7.26 (0.981) | 14.86 (0.923) |

EER solo sobre `eval` de nuestros modelos (`desglose_subconjuntos.py --nivel 2`, columna `eer_int`): Exp-3 rep1 1.1%, rep2 1.1%, ablación M160 1.8%; Exp-4 A original 5.8%, rep1 7.1%, rep3 6.9%.

Referencia de la literatura sobre ASVspoof 2019 LA `eval` (modelos entrenados solo con LA train, distinto protocolo que el nuestro): AASIST 0.83% y AASIST-L 0.99% (README de `clovaai/aasist`).

Lectura:
- Nuestro modelo supera por amplio margen a los tres baselines en distribución, fuera de dominio y compresión.
- **En zero-shot, LFCC-LCNN (3.15%) iguala al Exp-4 A (3.22 ± 0.91%).** La mejora de zero-shot proviene sobre todo de incluir bonafide de Common Voice en el entrenamiento y no de la arquitectura; la ventaja de la arquitectura aparece en los otros escenarios.
- Fuera de dominio, los tres baselines quedan con AUROC por debajo de 0.5 (0.23 a 0.28): ordenan peor que el azar, con 25 a 33% de detección de spoof y 63 a 75% de falsas alarmas.
- Bajo compresión marcan casi todo como bonafide (detección de spoof de 4 a 28%).
- Salvedades: cada baseline es una sola corrida; son modelos simplificados; y ASVspoof no es su dominio de entrenamiento (el corpus del Exp-4 casi no incluye ASVspoof), por lo que su EER en distribución no es comparable con los publicados.

### 2.10 Análisis estadístico (R8) y t-DCF (R7)

Modelos analizados: Exp-4 = `exp4_20261119_125609` y Exp-3 = `exp3_20261119_143040`, elegidos por menor `val_bce` entre las corridas convergidas; baselines de 2.9. Scripts: `exportar_scores.py` y `analisis_r8.py` (resultado completo en `~/evaluaciones/analisis_r8/` del servidor).

**EER (%) con IC 95 % por bootstrap de archivos** (300 repeticiones). Los IC reflejan solo la variabilidad del conjunto de prueba (los archivos se remuestrean como independientes; en ASVspoof hay correlación por locutor y ataque) y **no** la variabilidad entre corridas de entrenamiento, que es mayor (ver 2.7 y 2.8).

| Escenario | Exp-4 | Exp-3 | LFCC-LCNN | RawNet2 simpl. | AASIST simpl. |
|---|---|---|---|---|---|
| En distribución | 7.33 [7.07, 7.53] | 0.97 [0.90, 1.06] | 19.65 | 28.94 | 38.91 |
| Solo LA `eval` | 7.06 [6.83, 7.34] | 1.09 [0.98, 1.20] | 23.15 | 30.09 | 40.12 |
| Fuera de dominio | 15.32 [14.69, 15.86] | 20.56 [19.98, 21.08] | 70.00 | 74.94 | 69.68 |
| Compresión MP3 32 kbps | 13.38 [13.11, 13.63] | 3.82 [3.65, 4.00] | 39.22 | 29.64 | 45.19 |
| Zero-shot (9 idiomas) | 2.18 [2.04, 2.32] | 45.31 [44.84, 45.80] | 3.15 | 7.26 | 14.86 |

Las diferencias pareadas de EER (mismos archivos remuestreados) son estadísticamente significativas en todas las comparaciones, porque el conjunto de prueba es grande. En la práctica, la diferencia del Exp-4 frente a LFCC-LCNN en zero-shot (−0.97 pp) es del orden de la variabilidad entre corridas (sd 0.9 pp), por lo que se considera un empate.

**min t-DCF normalizado en ASVspoof 2019 LA `eval`** (modelo de costos del reto; ASV de referencia del paquete LA con EER 2.46 %; 71,228 de los 71,237 archivos, por los 9 sin embeddings):

| Modelo | min t-DCF [IC 95 %] |
|---|---|
| Exp-3 | 0.0293 [0.0251, 0.0325] |
| Exp-4 | 0.1780 [0.1694, 0.1845] |

Referencia publicada sobre el mismo conjunto: AASIST 0.0275 y AASIST-L 0.0309 (README de `clovaai/aasist`), con modelos entrenados solo con LA 2019 train y otro protocolo. El cálculo sigue la fórmula del código de AASIST y no se contrastó con el paquete oficial de ASVspoof, por lo que no se presenta como equivalente. En los demás escenarios el t-DCF no está definido (no hay sistema ASV).

**EER por idioma en el zero-shot** (9 idiomas con bonafide y spoof):

| Idioma | Exp-4 | Exp-3 | LFCC-LCNN | RawNet2 simpl. | AASIST simpl. |
|---|---|---|---|---|---|
| da | 0.92 | 29.27 | 0.84 | 18.07 | 20.71 |
| el | 0.63 | 60.26 | 1.04 | 2.25 | 9.92 |
| et | 3.00 | 23.84 | 1.25 | 3.66 | 10.53 |
| fa | 0.39 | 38.26 | 0.41 | 4.16 | 28.12 |
| hu | 0.75 | 57.62 | 0.83 | 2.00 | 8.75 |
| lt | 1.32 | 18.52 | 0.61 | 9.20 | 20.80 |
| nl | 1.46 | 40.88 | 3.21 | 3.96 | 16.21 |
| pt | 3.83 | 59.36 | 1.29 | 6.92 | 12.01 |
| ro | 0.71 | 55.46 | 2.42 | 3.58 | 10.59 |
| **Media ± desv.** | **1.45 ± 1.19** | 42.61 ± 16.26 | 1.32 ± 0.91 | 5.98 ± 5.07 | 15.29 ± 6.63 |
| Peor idioma | 3.83 | 60.26 | 3.21 | 18.07 | 28.12 |

La dispersión entre idiomas pasa de 16.3 pp en el Exp-3 a 1.2 pp en el Exp-4. **Salvedad de validez:** en este test el bonafide es siempre de Common Voice y el spoof siempre de MLAAD, y la misma pareja de fuentes aparece en el entrenamiento del Exp-4. Que un LFCC-LCNN, débil en ASVspoof (EER 19.65 %), llegue a 1.3 % apunta a que parte de la separación puede deberse a diferencias de fuente y canal y no solo a "real frente a sintético". Por eso este resultado se interpreta como generalización entre idiomas dentro de la misma pareja de fuentes, no como prueba de detección independiente del origen del audio; el escenario fuera de dominio (fuentes distintas) es la medida complementaria.

### 2.11 Alineación de estilo con RASA en el modelo final (R6)

`analisis_rasa_tsne.py --exp 4` sobre `exp4_20261119_125609` (λ 0.01 / 0.01 / 0.05): muestra de 150 audios por idioma y clase del conjunto de validación (16 idiomas, 4,800 audios), antes y después del RASA entrenado. Resultados en `~/entrenamientos/exp4_20261119_125609/analisis_rasa/` (figuras `tsne_<flujo>.png`).

| Flujo | Discrepancia de estilo entre idiomas (antes → después) | Dispersión de estilo | Exactitud lineal de idioma (antes → después) | Exactitud lineal de clase (antes → después) |
|---|---|---|---|---|
| Wav2Vec2BERT | 0.512 → 0.137 (−73 %) | 0.99 → 0.32 | 0.585 → 0.587 | 0.865 → 0.866 |
| WavLM Large | 0.630 → 0.114 (−82 %) | 1.01 → 0.31 | 0.882 → 0.887 | 0.819 → 0.819 |
| Whisper Large v3 | 0.435 → 0.154 (−65 %) | 1.00 → 0.40 | 0.980 → 0.978 | 0.883 → 0.880 |

Lectura: RASA reduce la discrepancia de estilo entre idiomas en un 65 a 82 %, pero **no elimina la información de idioma** de los embeddings (el idioma sigue siendo decodificable linealmente, con azar de 6.25 % para 16 idiomas) ni cambia la separación por clase. Con λ = 0.01, la pérdida de separación entre pares del banco al final del entrenamiento es de 0.86 a 0.98 (el banco conserva poca diversidad), frente a 0.09 a 0.10 con λ = 1.0. La invarianza al idioma del sistema se apoya en el EER uniforme por idioma (2.10) y no en la invarianza de los embeddings.

### 2.12 Degradaciones en varios idiomas (R7)

Muestra zero-shot de 5,388 audios (`~/particiones/muestra_zs.csv`), generada en el servidor con `generar_degradaciones.py`; la degradación se aplica a bonafide y spoof. Condiciones: `zs_limpio` (referencia), `zs_mp3_32kbps` (MP3 calibrado a ~33 kbps), `zs_plr10` (10 % de tramas de 20 ms anuladas al azar). Modelos: Exp-4 `exp4_20261119_125609`, Exp-3 `exp3_20261119_143040`, baselines `*_exp4_0059*`. Una corrida por modelo.

| Modelo | Limpio EER / AUROC | MP3 33 kbps EER / AUROC | PLR 10 % EER / AUROC |
|---|---|---|---|
| Exp-4 | 2.49 / 0.9977 | 3.71 / 0.9944 | 5.10 / 0.9891 |
| Exp-3 | 43.47 / 0.5848 | 39.43 / 0.6251 | 48.37 / 0.5367 |
| LFCC-LCNN simpl. | 3.86 / 0.9914 | 33.41 / 0.7255 | 6.89 / 0.9785 |
| RawNet2 simpl. | 8.67 / 0.9749 | 22.59 / 0.8510 | 13.47 / 0.9496 |
| AASIST simpl. | 15.92 / 0.9177 | 21.96 / 0.8435 | 16.18 / 0.9164 |

Exactitud del Exp-4: 97.62 % (limpio), 92.67 % (MP3; recall 85.60 %, precisión 99.65 %), 94.12 % (PLR). Lectura: el EER del Exp-4 sube poco y el AUROC se mantiene sobre 0.98, pero con MP3 el umbral se desplaza (cae el recall); sería necesario recalibrarlo. El Exp-3 no vio estos idiomas, así que su referencia sin degradar ya es de azar y no sirve para medir el efecto de la degradación. LFCC-LCNN es el más frágil frente a MP3. Salvedades: una corrida, un bitrate, un porcentaje de pérdida, sin AAC, misma pareja de fuentes Common Voice/MLAAD que el entrenamiento del Exp-4. Las EER por idioma de cada condición están en `~/evaluaciones/<escenario>/<run>.json` y no se analizan en este documento.

### 2.13 Ablación de RASA en el Exp-4 (R6): entrenamiento sin RASA

`train.py --exp 4 --sin-rasa --seed {1,2,3}` (`run_sin_rasa.sh`): RASA reemplazado por la identidad, pérdidas orth/reco en 0; el resto (gating, SERM con λ 0.05, corpus, 100 épocas) idéntico a la variante A. Corridas `exp4sinrasa_20261121_205142`, `_211007` y `_212843`; las tres convergieron.

EER (%), media ± sd (n=3 en cada grupo; "con RASA" = variante A convergida):

| Escenario | Sin RASA | Con RASA (A) | Diferencia (con − sin) |
|---|---|---|---|
| En distribución | 5.68 ± 0.86 | 6.92 ± 0.65 | +1.24 |
| Fuera de dominio | 15.49 ± 0.61 | 12.98 ± 2.31 | −2.51 |
| Compresión MP3 (ASVspoof) | 10.51 ± 1.83 | 13.56 ± 0.20 | +3.05 |
| Zero-shot (9 idiomas) | 2.47 ± 0.51 | 3.22 ± 0.92 | +0.75 |

Condiciones multilingües (muestra de 5,388 audios; con RASA solo hay una corrida, `exp4_20261119_125609`, con 2.49 / 3.71 / 5.10): sin RASA, limpio 2.65 ± 0.57, MP3 33 kbps 5.13 ± 1.44, PLR 10 % 4.70 ± 0.64.

Lectura: ninguna diferencia es estadísticamente clara con n=3 (contrastes de Welch con p entre 0.09 y 0.2 en los tres escenarios donde más difieren). Sin RASA el modelo no es peor en distribución, en MP3 ni en zero-shot, y es peor fuera de dominio por unos 2.5 puntos, dentro del ruido entre corridas de la variante A. Las tres corridas sin RASA convergieron, frente a 1 fallida de 4 con RASA en el Exp-4 y 1 de 3 en el Exp-3 (n pequeño: no es concluyente). Conclusión: en la configuración del modelo final **no hay evidencia de que RASA mejore la detección**; recordar que en el Exp-4 el banco de estilos está casi colapsado (train_orth 0.86–0.96; ver 2.11). La misma ablación sobre el Exp-3 se presenta en 2.13.2.

#### 2.13.1 Exp-4 con λ alto (variante C repetida, 3 semillas; 2026-11-22)

`run_lambda1.sh`: Exp-4 con RASA, λ orth/reco/serm = 1.0/1.0/0.05. Corridas `exp4_20261122_020351`, `_022204`, `_024013`. Las tres convergieron (val_acc 0.991) y el banco quedó diverso (L_orth ≈ 0.09–0.10, L_reco ≈ 0.033–0.039, frente a 0.86–0.96 / 1.2–1.4 de la variante A).

EER (%), media ± sd (n=3 por grupo):

| Escenario | A (λ 0.01) | C (λ 1) | Sin RASA |
|---|---|---|---|
| En distribución | 6.92 ± 0.65 | 6.40 ± 1.52 | 5.68 ± 0.86 |
| Fuera de dominio | 12.98 ± 2.31 | 16.40 ± 0.23 | 15.49 ± 0.61 |
| MP3 (ASVspoof) | 13.56 ± 0.20 | 15.11 ± 0.58 | 10.51 ± 1.83 |
| Zero-shot | 3.22 ± 0.92 | 2.28 ± 0.07 | 2.47 ± 0.51 |
| zs limpio / MP3 33 kbps / PLR 10 % (C, individuales) | | 2.23, 2.23, 2.15 / 3.60, 3.93, 3.64 / 4.32, 4.81, 4.99 | |

Lectura: la comparación previa con n=1 (B y C "peores que A") **no se sostiene** con 3 semillas. C iguala a A en distribución y es mejor en zero-shot (2.28 frente a 3.22; Welch p ≈ 0.2, tendencia, no diferencia demostrada), con variabilidad muy baja (sd 0.07). Frente a sin RASA, el zero-shot es un empate técnico. En fuera de dominio C (16.40) es peor que A y que sin RASA, así que el indicio a favor de RASA que daba A en ese escenario depende de la configuración y no es robusto. En MP3 sin RASA es claramente mejor. Conclusión: con λ bajo, λ alto y sin RASA (3 semillas cada uno) **no hay ventaja consistente de RASA en ningún escenario**; lo único sistemático es la menor variabilidad entre semillas de C en zero-shot (observación, n=3).

#### 2.13.2 Ablación sin RASA en el Exp-3 (banco diverso, λ 1.0; 2026-11-22)

`run_sin_rasa.sh cuda:1 3`: corridas `exp3sinrasa_20261121_215726` (semilla 1), `_231213` (2), `_002258` (3).

| Semilla | val_acc final | val_bce | Estado | Indist | Fuera dom. | MP3 | Zero-shot |
|---|---|---|---|---|---|---|---|
| 1 | 0.918 | 0.241 | No convergió | 23.72 | 22.55 | 53.86 | 50.71 |
| 2 | 0.988 | 0.022 | Convergió | 1.12 | 25.04 | 4.18 | 43.82 |
| 3 | 0.918 | 0.241 | No convergió | 23.68 | 22.45 | 53.90 | 49.33 |

Las semillas 1 y 3 cayeron en el mismo estado degenerado (curvas casi idénticas). La única convergida iguala al Exp-3 con RASA (indist 1.24 ± 0.40, MP3 4.62 ± 1.17, zero-shot 41.78 ± 2.35) y es peor fuera de dominio (25.04 frente a 17.27 ± 3.51; n=1). Falló 2 de 3 sin RASA frente a 1 de 3 con RASA en el Exp-3, pero en el Exp-4 ocurrió lo contrario (0 de 3 sin RASA frente a 1 de 4 con RASA), así que no hay señal consistente de que RASA estabilice el entrenamiento.

### 2.14 Aporte del Gating y línea base MLP sobre embeddings concatenados (R6; 2026-11-22)

**Streams individuales y fusión** (`analisis_gating.py`, checkpoints elegidos por val; EER %):

| Exp-4 | Indist | Fuera dom. | Zero-shot |
|---|---|---|---|
| Wav2Vec2BERT solo | 58.14 | 54.67 | 46.71 |
| WavLM solo | 48.91 | 28.47 | 45.58 |
| Whisper solo | 43.78 | 66.25 | 25.49 |
| Promedio fijo (1/3) | 45.05 | 46.12 | 27.48 |
| Gating aprendido | 7.33 | 15.32 | 2.18 |

Exp-3: mejor stream solo 22.65 / 36.32 / 34.49; promedio fijo 44.58 / 35.11 / 79.22; Gating 0.97 / 20.56 / 45.31. Pesos g medios: Wav2Vec2BERT ≈ 0.000–0.003 en ambos (anulado); Exp-4 WavLM 0.78–0.95, Whisper 0.05–0.22; Exp-3 WavLM 0.39–0.53, Whisper 0.47–0.60. Los clasificadores lineales por stream no son detectores independientes (varios con AUROC < 0.5); la detección aparece con la fusión no lineal. Los g medios no se deben leer como "importancia" de cada modelo (`analisis_gating.py` calcula además g por clase y su desviación).

**Línea base MLP** (`train_mlp_concat.py`, `run_mlp.sh`): MLP (3328→256→1) sobre los 3 embeddings concatenados, sin RASA, Gating ni SERM; misma partición, optimizador, 100 épocas y criterio (min val_bce). Exp-4, 3 semillas (`exp4mlp_20261122_150649`, `_152220`, `_153747`), todas convergen (val_acc 0.980–0.983; val_bce 0.045–0.053, frente a 0.026–0.038 de la variante C).

EER (%), media ± sd, n=3:

| Escenario | MLP concatenado | A (RASA λ 0.01) | C (RASA λ 1) | Sin RASA |
|---|---|---|---|---|
| Indist | 6.41 ± 1.39 | 6.92 ± 0.65 | 6.40 ± 1.52 | 5.68 ± 0.86 |
| Fuera de dominio | 14.10 ± 0.83 | 12.98 ± 2.31 | 16.40 ± 0.23 | 15.49 ± 0.61 |
| Zero-shot | 3.27 ± 0.59 | 3.22 ± 0.92 | 2.28 ± 0.07 | 2.47 ± 0.51 |
| zs limpio | 3.43 ± 0.58 | 2.49 (n=1) | 2.20 | 2.65 ± 0.57 |
| zs MP3 33 kbps | 4.85 ± 0.82 | 3.71 (n=1) | 3.72 | 5.13 ± 1.44 |
| zs PLR 10 % | 5.08 ± 0.92 | 5.10 (n=1) | 4.71 | 4.70 ± 0.64 |

(MP3 de ASVspoof no se evaluó para el MLP.)

Lectura: el MLP simple iguala al modelo completo dentro del ruido entre corridas en distribución, fuera de dominio y en las tres condiciones multilingües degradadas; en zero-shot es algo peor que C y sin RASA (3.27 frente a 2.28 y 2.47; Welch C vs MLP p ≈ 0.1, tendencia, no diferencia demostrada) y equivalente a A. Con n=3, **no hay evidencia de que la arquitectura RASA + Gating + SERM supere a un MLP sobre los embeddings concatenados** en EER; sí hay una pequeña ventaja de val_bce en distribución del modelo completo. La capacidad de detección proviene de los embeddings (efectivamente WavLM y Whisper) y del corpus; el Gating aporta la no linealidad que los streams lineales por separado no tienen, no se demostró un aporte específico de su diseño.

---

## 3. Ablación de M — resumen

Todos Exp3, λ=1.0. Solo varía el tamaño del style bank. In-distribución de M=32/64/128 no se evaluó en test (n/d).

| M | In-dist EER | In-dist AUROC | Fuera dom. AUROC | Fuera dom. EER |
|---|---|---|---|---|
| 16 | 1.01% | 0.9992 | 0.7976 | 27.24% |
| 32 | n/d | n/d | 0.8408 | 22.60% |
| 64 | n/d | n/d | 0.8410 | 22.55% |
| 128 | n/d | n/d | 0.8468 | 22.06% |
| 160 (ablacion_M160) | 1.72% | 0.9983 | **0.9346** | **13.57%** |
| 160 (exp3_20261006) | 1.06% | 0.9990 | 0.8774 | 18.60% |
| 256 | **0.98%** | **0.9993** | 0.8534 | 21.10% |

---

## 4. Conclusiones

### Modelo de referencia del Exp-3
- `ablacion_M160_20261111_101359` (Exp3, λ=1.0, M=160) fue la mejor corrida de la ablación en fuera de dominio (EER 13.57%, AUROC 0.9346), pero las repeticiones de la misma configuración dieron 20.56% y 17.69% (ver 2.8): esa ventaja no se reproduce.
- La corrida original del Exp-3 (`exp3_20261006_082449`) tiene la misma configuración nominal pero rinde peor en fuera de dominio (EER 18.60%, AUROC 0.8774) y mejor en in-distribución (EER 1.06% frente a 1.72%). Dos corridas de la misma configuración difieren ~5 puntos de EER fuera de dominio: **hay varianza entre corridas**.

### Sobre el parámetro λ
- λ=1.0 supera a λ=0.01 en in-distribución para Exp3 (modelos RASA v1 descartados).
- En Exp-4, con n=1 B y C parecían peores que A, pero C repetida con 3 semillas (2.13.1) iguala a A en distribución y es mejor en zero-shot; la conclusión "λ bajo es mejor" no se sostiene. B (1/1/1) sigue siendo n=1.

### Sobre la ablación de M
- En in-distribución las diferencias entre M son mínimas (EER 0.98% a 1.72%).
- En fuera de dominio, M=160 salió mejor en la ablación (13.57% frente a 21.10 a 27.24% del resto), pero sus repeticiones (20.56% y 17.69%) caen en el mismo rango que los demás M. Con una sola corrida por valor de M y esta varianza, **la ablación no permite concluir que M=160 sea mejor**; la elección de M=160 debe apoyarse en el valor del artículo original y no en este resultado.

### Sobre el tamaño del corpus (Exp1/Exp2/Exp3)
- Con las etiquetas corregidas, Exp-1 (EER 16.7 a 17.0%) rinde igual o mejor que Exp-2 (22.4 a 25.1%) y Exp-3 (18.6%) en fuera de dominio. La conclusión anterior de que "más datos mejora la generalización fuera de dominio" **no se sostiene**. Las corridas mezclan λ distintos, así que no es una comparación controlada.

### Sobre el zero-shot y el Exp-4
- El Exp-3 detecta el spoof en idiomas no vistos pero falla con el bonafide de Common Voice (EER 39 a 41%), por el dominio acústico del bonafide de entrenamiento.
- Incorporar bonafide de Common Voice (Exp-4) lleva el zero-shot a EER 3.57% (variante A) a costa de in-distribución (6.17%) y compresión (13.78%).
- En fuera de dominio, el Exp-4 A también mejora (10.71%).
- Con las corridas convergidas (Exp-4 A: 3; Exp-3: 2 repeticiones más la ablación), el zero-shot es robusto (EER 3.22 ± 0.91% frente a 41.78 ± 2.35%), y el costo en distribución (6.92 ± 0.65% frente a 1.24 ± 0.40%) y compresión (13.56 ± 0.20% frente a 4.62 ± 1.17%) también.
- En fuera de dominio, el Exp-4 A queda mejor (12.98 ± 2.31% frente a 17.27 ± 3.51%), una diferencia de unos 4 puntos con n=3 por grupo: sugerente, no concluyente.
- Una de cada tres o cuatro corridas no convergió, tanto en el Exp-3 como en el Exp-4 (ver 2.7 y 2.8); es una inestabilidad del entrenamiento, no del conjunto de datos.

### Sobre `minDCF`
- Los valores de minDCF de este documento usan la implementación de `evaluate.py` (`P_TARGET=0.05` aplicado a la clase spoof), que no coincide necesariamente con el t-DCF de ASVspoof. No son comparables con los publicados.

---

## 5. Limitaciones

- Compresión: solo se evaluó MP3 a 32 kbps (no otros bitrates ni AAC); pérdida de paquetes: un solo porcentaje (10 %).
- Las líneas base son versiones simplificadas, con una corrida por modelo y entrenadas con el corpus del Exp-4.
- El t-DCF solo está definido en ASVspoof 2019 LA (hay sistema de verificación de referencia) y no se contrastó con el paquete oficial.
- Los valores de minDCF usan la implementación de `evaluate.py` y no son comparables con los publicados.
- Los intervalos de confianza por *bootstrap* no incluyen la variabilidad entre corridas de entrenamiento.
