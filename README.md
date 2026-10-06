# Detección de audios sintéticos con modelos fundacionales y datos multilingües

[![CI](https://github.com/Josergio335/audio-deepfake-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Josergio335/audio-deepfake-detection/actions/workflows/ci.yml)

Código de la tesis *Desarrollo y evaluación de modelos de aprendizaje profundo para la detección de audios sintéticos en un entorno multilingüe* (Pontificia Universidad Católica del Perú, Sergio Huamán; asesor: Pablo Fonseca).

El sistema construye un corpus multilingüe balanceado, extrae representaciones de audio con tres modelos fundacionales congelados (Wav2Vec2BERT, WavLM Large y Whisper Large v3), las alinea en estilo con RASA y las fusiona con una red de compuertas para clasificar cada audio como real (*bonafide*) o sintético (*spoof*).

## Estructura del repositorio

| Carpeta | Contenido | Resultado de la tesis |
|---|---|---|
| `01_extraccion/` | Extracción de las bases de datos (`extraer_datasets.py`, `extraer_codecfake.py`) | R3 |
| `02_organizacion/` | Organización y medición del corpus (`organizar_datasets.py`) | R3 |
| `03_particion/` | Partición en tres experimentos anidados Exp-1 ⊂ Exp-2 ⊂ Exp-3 (`particion_dataset.py`) | R3 |
| `04_preprocesamiento/` | Remuestreo a 16 kHz mono, VAD y descarte (`preprocesamiento_dataset.py`, `filtrar_manifests.py`; versión para prueba: `preprocesamiento_test.py`) | R4 |
| `05_modelo/extraer_embeddings.py` | Extracción de embeddings con los tres modelos fundacionales | R5 |
| `05_modelo/models/rasa.py` | Alineación de estilo RASA (versión final); `rasa_v1_colapsado.py` conserva la primera implementación | R6 |
| `05_modelo/models/serm.py`, `gating.py`, `train.py` | SERM, red de compuertas y entrenamiento | R1 |
| `05_modelo/analisis_rasa_tsne.py` | Análisis t-SNE y sondas lineales del efecto de RASA | R6 |
| `05_modelo/generar_manifests_test.py`, `evaluate.py` | Manifests de prueba y evaluación | R7 (en curso) |
| `tests/` | Pruebas unitarias y prueba de humo del entrenamiento | R2, R5, R6 |
| `modelos_entrenados/` | Mejor checkpoint, curvas (`metricas.csv`) y configuración de Exp-1, Exp-2 y Exp-3 | R1 |
| `resultados_rasa_exp1/` | Figuras y tablas del análisis de RASA | R6 |
| `docs/` | Anexo técnico del corpus | R3 |

## Instalación

Entorno completo (servidor con GPU NVIDIA):

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Entorno mínimo para las pruebas (CPU, sin modelos fundacionales):

```bash
pip install -r requirements-ci.txt
```

## Pruebas

```bash
python -m pytest tests -v
```

Las pruebas no necesitan GPU ni datos reales:

- `test_extractor.py` (R5): embedding de dimensión fija sin importar la duración del audio, nombres por hash, extracción reanudable, manejo de archivos faltantes o vacíos e índice de embeddings.
- `test_rasa.py` (R6): el banco de estilos se inicializa con estilos reales, no colapsa, las pérdidas auxiliares se comportan como se espera y los checkpoints antiguos se cargan.
- `test_train_smoke.py`: ejecuta `train.py` de punta a punta con datos sintéticos (RASA, SERM y red de compuertas), y comprueba las métricas y los checkpoints; también el reparto determinista train/validación.

Estas pruebas se ejecutan en cada *push* con GitHub Actions (`.github/workflows/ci.yml`).

## Flujo de ejecución

Los datos no se incluyen en el repositorio. Las rutas de datos están definidas al inicio de cada script y deben ajustarse al entorno.

1. `01_extraccion/` → `02_organizacion/` → `03_particion/`: producen los manifests de Exp-1, Exp-2 y Exp-3.
2. `04_preprocesamiento/preprocesamiento_dataset.py`: audio estandarizado (16 kHz, mono, sin silencios, WAV de 16 bits).
3. `05_modelo/extraer_embeddings.py`: un tensor `.pt` por audio y modelo, más `indice.csv`. Es reanudable.
4. `05_modelo/train.py --exp {1,2,3} --lambda-orth 1.0 --lambda-reco 1.0`: entrena RASA, SERM y la red de compuertas sobre los embeddings (validación del 20 % por hash de la ruta).
5. `05_modelo/analisis_rasa_tsne.py`: análisis del efecto de RASA sobre los embeddings.

## Modelos entrenados

`modelos_entrenados/exp{1,2,3}_*/` contiene el mejor checkpoint (`checkpoint_mejor.pt`), las curvas de pérdida y precisión por época (`metricas.csv`) y los hiperparámetros (`config.json`) de las corridas con la versión final de RASA (peso 1.0 de las pérdidas auxiliares). Los modelos fundacionales se descargan de Hugging Face (`facebook/w2v-bert-2.0`, `microsoft/wavlm-large`, `openai/whisper-large-v3`).

## Estado

Entregable 2: resultados R2 a R6. La evaluación sobre los conjuntos de prueba (idiomas no vistos y fuera de dominio) y la comparación con otros sistemas corresponden al siguiente entregable.
