# Detección de audios sintéticos con modelos fundacionales y datos multilingües

[![CI](https://github.com/Josergio335/audio-deepfake-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Josergio335/audio-deepfake-detection/actions/workflows/ci.yml)

Código de la tesis *Desarrollo y evaluación de modelos de aprendizaje profundo para la detección de audios sintéticos en un entorno multilingüe* (Pontificia Universidad Católica del Perú, Sergio Huamán; asesor: Pablo Fonseca).

El sistema construye un corpus multilingüe balanceado, extrae representaciones de audio con tres modelos fundacionales congelados (Wav2Vec2BERT, WavLM Large y Whisper Large v3), las alinea en estilo con RASA y las fusiona con una red de compuertas para clasificar cada audio como real (*bonafide*) o sintético (*spoof*). Se evalúa en cuatro escenarios (en distribución, zero-shot en idiomas no vistos, fuera de dominio y bajo compresión) y se compara con líneas base simplificadas.

## Estructura del repositorio

Los scripts se organizan por etapas del flujo. Cada tabla indica a qué resultado de la tesis (R1 a R8) corresponden.

### Código del flujo principal (reproduce los resultados)

| Carpeta o archivo | Contenido | Resultado |
|---|---|---|
| `01_extraccion/` | Extracción de las bases de datos (`extraer_datasets.py`, `extraer_codecfake.py`, `extraer_commonvoice.py`) | R3 |
| `02_organizacion/` | Organización y medición del corpus (`organizar_datasets.py`) | R3 |
| `03_particion/particion_dataset.py` | Partición en los experimentos anidados Exp-1 ⊂ Exp-2 ⊂ Exp-3 | R3 |
| `05_modelo/generar_manifest_exp4.py` | Manifest del corpus ampliado Exp-4 (16 idiomas) | R3 |
| `04_preprocesamiento/` | Remuestreo a 16 kHz mono, VAD y descarte (`preprocesamiento_dataset.py`, `preprocesamiento_commonvoice.py`, `filtrar_manifests.py`; para prueba: `preprocesamiento_test.py`) | R4 |
| `05_modelo/extraer_embeddings.py` | Extracción de embeddings con los tres modelos fundacionales | R5 |
| `05_modelo/models/` | RASA (`rasa.py`), SERM (`serm.py`) y red de compuertas (`gating.py`) | R1, R6 |
| `05_modelo/train.py` | Entrenamiento (`--exp {1,2,3,4}`) | R1 |
| `05_modelo/generar_manifests_test.py`, `extraer_embeddings_test.py` | Manifests y embeddings de los conjuntos de prueba | R7 |
| `05_modelo/generar_degradaciones.py`, `04_preprocesamiento/comprimir_asv_eval.py` | Degradaciones de prueba (compresión MP3 y pérdida de paquetes) | R7 |
| `05_modelo/evaluate.py`, `desglose_subconjuntos.py` | Evaluación (EER, AUROC) y desglose por subconjunto | R7 |
| `05_modelo/exportar_scores.py`, `analisis_r8.py` | Puntajes por archivo, intervalos de confianza por *bootstrap* y t-DCF | R7, R8 |
| `05_modelo/baselines/` | Líneas base simplificadas (LFCC-LCNN, RawNet2, AASIST): `train_baseline.py`, `eval_baseline.py`, `run_baseline.sh` | R8 |
| `tests/`, `.github/workflows/ci.yml` | Pruebas unitarias, prueba de humo del entrenamiento e integración continua | R2 |

### Ablaciones y análisis (respaldan las tablas del Capítulo 6)

| Archivo | Contenido |
|---|---|
| `05_modelo/analisis_rasa_tsne.py` | Análisis t-SNE y sondas lineales del efecto de RASA |
| `05_modelo/train_ablacion_M.py` | Ablación del tamaño del banco de estilos (M) |
| `05_modelo/run_sin_rasa.sh`, `run_lambda1.sh` | Entrenamiento sin RASA (`--sin-rasa`) y con pesos de pérdida λ = 1.0 en el Exp-4 |
| `05_modelo/analisis_gating.py` | Puntajes por flujo, promedio fijo y red de compuertas |
| `05_modelo/train_mlp_concat.py`, `run_mlp.sh` | Clasificador de referencia sobre los embeddings concatenados |

### Utilidades puntuales

`03_particion/analizar_manifests_test.py`, `03_particion/añadir_bonafide_zeroshot.py`, `05_modelo/corregir_etiquetas_fueradominio.py` (corrección de etiquetas de SONAR), `05_modelo/baselines/prueba_rapida.sh`, `05_modelo/models/rasa_v1_colapsado.py` (primera implementación de RASA, conservada como referencia) y los scripts de figuras de `resultados_e3/`.

### Resultados y datos de referencia

| Carpeta | Contenido |
|---|---|
| `modelos_entrenados/` | Mejor checkpoint, curvas (`metricas.csv`), configuración y registro de entrenamiento de los modelos finales (ver más abajo) |
| `manifests/` | Manifests de Exp-1 a Exp-4, comprimidos (`.csv.gz`) |
| `resultados_e3/` | Curvas de entrenamiento (`curvas_r1.png`), métricas de las corridas, y coordenadas y figuras t-SNE del modelo final |
| `resultados_rasa_exp1/` | Figuras y tablas del análisis de RASA sobre el Exp-1 |
| `docs/` | Anexo técnico del corpus (`anexo_corpus.md`) y resultados de la evaluación (`resultados_evaluacion.md`) |

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
- `test_sin_rasa.py` (R6): con `--sin-rasa` los embeddings pasan sin cambios y las pérdidas de RASA son nulas.
- `test_train_smoke.py`: ejecuta `train.py` de punta a punta con datos sintéticos (RASA, SERM y red de compuertas), y comprueba las métricas y los checkpoints; también el reparto determinista train/validación.
- `test_etiquetas_test.py` (R7): regla de etiquetado de SONAR en el escenario fuera de dominio.

Estas pruebas se ejecutan en cada *push* con GitHub Actions (`.github/workflows/ci.yml`).

## Flujo de ejecución

Los datos no se incluyen en el repositorio. Las rutas de datos están definidas al inicio de cada script y deben ajustarse al entorno. Las opciones de cada script se describen en su encabezado.

1. `01_extraccion/` → `02_organizacion/` → `03_particion/`: producen los manifests de Exp-1, Exp-2 y Exp-3; `05_modelo/generar_manifest_exp4.py` produce el del Exp-4.
2. `04_preprocesamiento/`: audio estandarizado (16 kHz, mono, sin silencios, WAV de 16 bits).
3. `05_modelo/extraer_embeddings.py`: un tensor `.pt` por audio y modelo, más `indice.csv`. Es reanudable.
4. `05_modelo/train.py --exp {1,2,3,4}`: entrena RASA, SERM y la red de compuertas sobre los embeddings (validación del 20 % por hash de la ruta). Los pesos de las pérdidas auxiliares se fijan con `--lambda-orth`, `--lambda-reco` y `--lambda-serm`: el Exp-3 se entrenó con 1.0 / 1.0 / 0.05 y el Exp-4, con los valores por defecto (0.01 / 0.01 / 0.05).
5. Evaluación: `generar_manifests_test.py` → `preprocesamiento_test.py` → `extraer_embeddings_test.py` → `evaluate.py --checkpoint <ruta> --escenario <nombre>`. Las degradaciones se generan con `generar_degradaciones.py` y `comprimir_asv_eval.py`.
6. Comparación: `baselines/run_baseline.sh` entrena y evalúa las líneas base; `exportar_scores.py` y `analisis_r8.py` calculan intervalos de confianza y t-DCF.
7. Ablaciones y análisis de RASA y de las compuertas: los scripts de la sección correspondiente.

## Manifests

`manifests/exp{1,2,3,4}_manifest.csv.gz` listan los audios exactos de cada experimento, con su base de datos, idioma, clase y duración. Contienen rutas absolutas del equipo de trabajo y del servidor. Los manifests se generan con una semilla fija, pero reproducirlos exige disponer del mismo catálogo de duraciones y de los mismos archivos descargados (en el Exp-4, los clips de Common Voice).

## Modelos entrenados

`modelos_entrenados/` contiene, por corrida, el mejor checkpoint (`checkpoint_mejor.pt`), las curvas por época (`metricas.csv`) y los hiperparámetros (`config.json`):

| Carpeta | Descripción |
|---|---|
| `exp4_20261119_125609/` | **Modelo final**: corpus ampliado de 16 idiomas, pesos de las pérdidas auxiliares 0.01 / 0.01 / 0.05 |
| `exp3_20261119_143040/` | Referencia con el corpus de 7 idiomas, pesos 1.0 / 1.0 / 0.05 |
| `exp1_20261006_002845/`, `exp2_20261006_082449/`, `exp3_20261006_082449/` | Corridas con el corpus de 7 idiomas y volúmenes crecientes |

Los modelos fundacionales se descargan de Hugging Face (`facebook/w2v-bert-2.0`, `microsoft/wavlm-large`, `openai/whisper-large-v3`).

## Estado

Incluye el flujo completo de la tesis: corpus (R3), preprocesamiento (R4), extracción de embeddings (R5), RASA (R6), modelo (R1), pruebas e integración continua (R2), evaluación en los cuatro escenarios (R7) y comparación con líneas base (R8). Los resultados están en `docs/resultados_evaluacion.md`.
