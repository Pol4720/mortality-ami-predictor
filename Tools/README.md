# Predictor de Mortalidad Intrahospitalaria / Arritmia Ventricular (Tools)

Proyecto de ML end-to-end y reproducible para predecir mortalidad intrahospitalaria y arritmias ventriculares. Incluye código modular en Python, notebooks, pruebas automáticas y un dashboard en Streamlit.

Importante: La única entrada obligatoria es `DATASET_PATH` (ruta a tu dataset CSV/Parquet). No se deben codificar otras rutas en el código.

## Instalación rápida (Windows PowerShell)

1) Crear entorno

```powershell
# Opción Conda
conda env create -f environment.yml; conda activate mortality-ami-env

# Opción venv (Python)
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Nota sobre PyTorch/torchvision (opcional):
- La red neuronal en `src/models.py` es opcional y está desactivada por defecto. Si no la vas a usar, no necesitas instalar `torch`/`torchvision`.
- En Windows, instala PyTorch desde el índice oficial para evitar errores como: `ERROR: No matching distribution found for torchvision`.

Instalación PyTorch en Windows (elige una):

```powershell
# CPU-only
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision

# CUDA 12.1 (si cuentas con drivers/CUDA compatibles)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# Extras opcionales
pip install pytorch-lightning captum
```

Para activar el modelo de red neuronal en los experimentos, define:

```powershell
$env:ENABLE_TORCH_MODEL = "1"
```

2) Definir DATASET_PATH (ejemplo)

```powershell
$env:DATASET_PATH = "C:\\data\\ami_dataset.csv"
```

3) Ejecutar experimentos (entrenamiento + evaluación)

```powershell
# Script PowerShell
.\u200brun_experiments.ps1 -Data $env:DATASET_PATH

# (Opcional en Linux/macOS)
# export DATASET_PATH="/ruta/ami_dataset.csv"
# bash run_experiments.sh
```

4) Lanzar el dashboard en Streamlit

```powershell
streamlit run streamlit_app.py -- --data $env:DATASET_PATH
```

5) Ejecutar pruebas unitarias

```powershell
pytest -q
```

## Estructura del proyecto

- `src/` código modular:
	- `config.py`: configuración global (incluye `DATASET_PATH`), semillas y nombres de columnas objetivo.
	- `data.py`: carga de datos (CSV/Parquet/Feather), resumen EDA, gráficos y split (estratificado o temporal).
	- `preprocess.py`: imputación (IterativeImputer/KNNImputer), codificación One-Hot, escalado, selección opcional de variables.
	- `features.py`: utilidades para seleccionar columnas seguras (excluye identificadores/objetivos).
	- `models.py`: modelos KNN, Regresión Logística (L1/L2 + calibración), Árbol de Decisión, XGBoost, LightGBM, Red Neuronal (PyTorch con focal loss opcional), KMeans y Regresión Lineal.
	- `train.py`: validación cruzada anidada, RandomizedSearchCV, SMOTE (si disponible), logging en MLflow (opcional), y guardado de artefactos.
	- `evaluate.py`: métricas (AUROC, AUPRC, Accuracy, Precision, Recall, F1, Brier), curvas de calibración, Decision Curve Analysis, matriz de confusión, comparación con GRACE/TIMI (si existen columnas), perfiles KMeans y `final_evaluation.pdf`.
	- `explain.py`: explicabilidad global/local con SHAP (importación perezosa), importancia por permutación y PDP.
	- `predict.py`: recarga un modelo guardado y genera predicciones por lotes.
- `notebooks/`:
	- `eda.ipynb`: EDA (resúmenes, faltantes, correlaciones, distribución del objetivo) y guarda gráficos en `reports/figures/`.
	- `modeling.ipynb`: entrenamiento rápido de un modelo base.
	- `explainability.ipynb`: resumen SHAP global del mejor modelo.
- `reports/`: figuras y `final_evaluation.pdf` tras evaluar.
- `models/`: artefactos guardados (`.joblib`) y el conjunto de test hold-out (`.parquet`).
- `tests/`: pruebas con pytest (carga/preproceso, entrenamiento rápido, guardado/carga de modelo).
- `run_experiments.ps1` y `run_experiments.sh`: scripts para ejecutar experimentos y registrar métricas (MLflow opcional).
- `.github/workflows/ci.yml`: skeleton de CI (instala dependencias y corre pruebas).

## Cómo entrenar y evaluar

Entrenamiento (validación cruzada anidada + búsqueda aleatoria de hiperparámetros):

```powershell
python -m src.train --data $env:DATASET_PATH --task mortality
python -m src.train --data $env:DATASET_PATH --task arrhythmia
# Si tienes una columna continua (p.ej. estancia):
python -m src.train --data $env:DATASET_PATH --task regression
```

- Usa `--quick` para una ejecución rápida de depuración.
- Los mejores modelos se guardan en `models/` y el conjunto de test en `models/testset_*.parquet`.

Evaluación (hold-out):

```powershell
python -m src.evaluate --data $env:DATASET_PATH --task mortality
```

Salidas: `reports/final_metrics_mortality.csv`, `reports/final_evaluation.pdf` y figuras en `reports/figures/`.

## Dashboard (Streamlit)

- Cargar o seleccionar un paciente (fila), mostrar probabilidad predicha, contribuciones globales (SHAP) y análisis "what-if" para variables numéricas.
- Ejecuta:

```powershell
streamlit run streamlit_app.py -- --data $env:DATASET_PATH
```

## Predicción por lotes

```powershell
python -m src.predict --model .\models\best_classifier_mortality.joblib --input .\mis_pacientes.csv --output .\predicciones.csv
```

## Seguimiento de experimentos (opcional, MLflow)

Define las variables para activar el logging:

```powershell
$env:EXPERIMENT_TRACKER = "mlflow"
$env:TRACKING_URI = "http://localhost:5000"   # o ruta de tracking
```

## DATASET_PATH (única fuente de datos)

- Debe apuntar al fichero CSV/Parquet del dataset.
- Configúralo vía variable de entorno o pásalo por `--data` en CLI/Streamlit.
- Ejemplo: `C:\\data\\ami_dataset.csv`.

## Consideraciones de privacidad

- No se registran ni se muestran identificadores directos de pacientes.
- Si tu dataset incluye IDs (patient_id, MRN), evita usarlos como features y no los vuelques en logs/figuras.

## Reproducibilidad

- Semilla fija
- Validación cruzada estratificada y hold-out final
- Pipelines y modelos guardados en `models/` y reutilizables

## Resolución de problemas

- Dependencias opcionales: si ves errores de importación de `imblearn`, `mlflow` o `shap`, asegúrate de instalarlas (están en `requirements.txt` y `environment.yml`).
- Columnas faltantes: el código tolera columnas opcionales; si falta la columna objetivo configurada, se devolverá un error claro.
- GPU/NN: si no hay CUDA, la red neuronal usa CPU automáticamente.
