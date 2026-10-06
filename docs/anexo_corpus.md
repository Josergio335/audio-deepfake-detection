# Anexo Técnico: Estadística Descriptiva del Corpus

**Proyecto:** Desarrollo y evaluación de modelos de aprendizaje profundo para la detección de audios sintéticos con fines de mitigación de desinformación digital  
**Autor:** Sergio Estéfano Huamán Noriega  
**Generado:** 2026-09-08 | Semilla de partición: 42

---

## 1. Datasets utilizados

El corpus de entrenamiento integra 4 bases de datos públicas de vanguardia:

| Dataset | Tipo | Idiomas | Clases disponibles |
|---------|------|---------|-------------------|
| ASVspoof 2019 LA | Voz sintética (TTS/VC) | Inglés | bonafide, spoof |
| M-AILABS | Audiolibros narrados | de, fr, es, en, it, pl, ru | bonafide |
| MLAAD | Voz sintética multilingüe (54 idiomas) | de, fr, es, en, it, pl, ru (core) | spoof |
| CodecFake | Voz sintética con codecs neuronales | Inglés | bonafide, spoof |

### Datasets excluidos del entrenamiento

| Dataset | Motivo |
|---------|--------|
| SpoofCeleb | Solo inglés spoof; el volumen en inglés ya supera el cap experimental |
| SONAR | Audio "in the wild" introduce dominio acústico distinto (no controlado) |
| ASVspoof 2019 LA (dev/eval) | Reservados para validación y prueba final |
| ASVspoof PA | Protocolo de replay attacks — problema distinto a la voz sintética |

---

## 2. Idiomas de entrenamiento

7 idiomas tipológicamente diversos seleccionados por disponibilidad de ambas clases (bonafide y spoof):

| Idioma | Código | Fuente bonafide | Fuente spoof |
|--------|--------|----------------|--------------|
| Inglés | en | M-AILABS, CodecFake | ASVspoof, MLAAD, CodecFake |
| Alemán | de | M-AILABS | MLAAD |
| Francés | fr | M-AILABS | MLAAD |
| Español | es | M-AILABS | MLAAD |
| Italiano | it | M-AILABS | MLAAD |
| Polaco | pl | M-AILABS | MLAAD |
| Ruso | ru | M-AILABS | MLAAD |

---

## 3. Diseño experimental — particiones anidadas

Se definen 3 experimentos con cap progresivo de horas por (idioma, clase), formando subconjuntos anidados: **Exp-1 ⊂ Exp-2 ⊂ Exp-3**. El objetivo es evaluar el impacto del volumen de datos en el rendimiento del modelo.

| Experimento | Cap por (idioma, clase) | Cuello de botella | Total archivos | Total horas |
|-------------|------------------------|-------------------|---------------|-------------|
| Exp-1 | 18h | Ruso spoof (~18h) | 136,356 | 252.0h |
| Exp-2 | 32h | Polaco bonafide (~32h) | 236,240 | 434.2h |
| Exp-3 | 46h | Italiano spoof (~46h) | 329,262 | 602.4h |

Los subconjuntos se derivan con semilla fija (seed=42) para garantizar reproducibilidad.

---

## 4. Distribución por idioma y clase

### Exp-1 (252h — cap 18h/clase)

| Idioma | Clase | Archivos | Horas |
|--------|-------|----------|-------|
| de | bonafide | 9,123 | 18.0h |
| de | spoof | 9,236 | 18.0h |
| en | bonafide | 11,018 | 18.0h |
| en | spoof | 15,652 | 18.0h |
| es | bonafide | 9,730 | 18.0h |
| es | spoof | 10,213 | 18.0h |
| fr | bonafide | 8,444 | 18.0h |
| fr | spoof | 8,911 | 18.0h |
| it | bonafide | 9,712 | 18.0h |
| it | spoof | 10,575 | 18.0h |
| pl | bonafide | 8,839 | 18.0h |
| pl | spoof | 8,410 | 18.0h |
| ru | bonafide | 7,876 | 18.0h |
| ru | spoof | 8,617 | 18.0h |
| **TOTAL** | | **136,356** | **252.0h** |

### Exp-2 (434h — cap 32h/clase)

| Idioma | Clase | Archivos | Horas |
|--------|-------|----------|-------|
| de | bonafide | 16,221 | 32.0h |
| de | spoof | 16,435 | 32.0h |
| en | bonafide | 19,641 | 32.0h |
| en | spoof | 27,984 | 32.0h |
| es | bonafide | 17,386 | 32.0h |
| es | spoof | 18,207 | 32.0h |
| fr | bonafide | 15,018 | 32.0h |
| fr | spoof | 15,897 | 32.0h |
| it | bonafide | 17,285 | 32.0h |
| it | spoof | 18,852 | 32.0h |
| pl | bonafide | 15,690 | 32.0h |
| pl | spoof | 14,840 | 32.0h |
| ru | bonafide | 14,059 | 32.0h |
| ru | spoof | 8,725 | 18.2h |
| **TOTAL** | | **236,240** | **434.2h** |

### Exp-3 (602h — cap 46h/clase)

| Idioma | Clase | Archivos | Horas |
|--------|-------|----------|-------|
| de | bonafide | 23,353 | 46.0h |
| de | spoof | 23,627 | 46.0h |
| en | bonafide | 28,273 | 46.0h |
| en | spoof | 40,170 | 46.0h |
| es | bonafide | 24,974 | 46.0h |
| es | spoof | 26,184 | 46.0h |
| fr | bonafide | 21,685 | 46.0h |
| fr | spoof | 22,816 | 46.0h |
| it | bonafide | 24,805 | 46.0h |
| it | spoof | 27,000 | 45.9h |
| pl | bonafide | 22,496 | 46.0h |
| pl | spoof | 15,000 | 32.3h |
| ru | bonafide | 20,154 | 46.0h |
| ru | spoof | 8,725 | 18.2h |
| **TOTAL** | | **329,262** | **602.4h** |

---

## 5. Distribución por dataset

| Dataset | Exp-1 archivos | Exp-1 horas | Exp-2 archivos | Exp-2 horas | Exp-3 archivos | Exp-3 horas |
|---------|---------------|-------------|---------------|-------------|---------------|-------------|
| ASVspoof | 654 | 0.6h | 1,185 | 1.1h | 1,708 | 1.6h |
| CodecFake | 17,089 | 17.1h | 30,641 | 30.6h | 44,023 | 43.9h |
| M-AILABS | 60,347 | 121.6h | 107,420 | 216.2h | 154,389 | 310.7h |
| MLAAD | 58,266 | 112.7h | 96,994 | 186.4h | 129,142 | 246.2h |
| **TOTAL** | **136,356** | **252.0h** | **236,240** | **434.2h** | **329,262** | **602.4h** |

---

## 6. Preprocesamiento aplicado

Todos los archivos del corpus fueron preprocesados con el siguiente pipeline antes del entrenamiento:

| Paso | Descripción | Herramienta |
|------|-------------|-------------|
| Remuestreo | 16,000 Hz, canal mono | librosa |
| VAD | Modo agresividad 2, frames de 30ms, mínimo 0.5s de habla | webrtcvad |
| Formato de salida | WAV PCM 16-bit | soundfile |

**Resultado del preprocesamiento (Exp-3):**
- Archivos procesados exitosamente: 327,739
- Archivos vacíos tras VAD (descartados): 491
- Errores de lectura: 0

Los archivos descartados por VAD fueron removidos de los manifests mediante `04_preprocesamiento/filtrar_manifests.py`.
