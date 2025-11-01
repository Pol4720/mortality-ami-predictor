# Custom Models System Architecture

Visual diagrams and architecture overview for the Custom Models system.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Custom Models Ecosystem                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  Base Classes │          │  Integration  │          │  Persistence  │
│               │          │    Modules    │          │    System     │
├───────────────┤          ├───────────────┤          ├───────────────┤
│ • BaseCustom  │◄─────────┤ • Training    │◄─────────┤ • Save/Load   │
│   Model       │          │ • Evaluation  │          │ • Validation  │
│ • Classifier  │          │ • Explain-    │          │ • Versioning  │
│ • Regressor   │          │   ability     │          │ • Bundles     │
│ • Wrapper     │          │               │          │               │
└───────────────┘          └───────────────┘          └───────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                          ┌──────────────────┐
                          │ Dashboard UI     │
                          │ (Streamlit)      │
                          ├──────────────────┤
                          │ • Upload Models  │
                          │ • Train          │
                          │ • Evaluate       │
                          │ • Manage         │
                          └──────────────────┘
```

---

## Core Components

### 1. Base Classes Hierarchy

```
                    BaseCustomModel (ABC)
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
BaseCustomClassifier               BaseCustomRegressor
        │                                       │
        │ Required:                             │ Required:
        │ • fit(X, y)                          │ • fit(X, y)
        │ • predict(X)                         │ • predict(X)
        │ • predict_proba(X)                   │ • get_params()
        │ • classes_ (property)                │ • set_params(**params)
        │ • get_params()                       │
        │ • set_params(**params)               │
        │                                       │
        └───────────────┬───────────────────────┘
                        │
                        │ Used by:
                        ▼
            CustomModelWrapper
                        │
                        │ Adds:
                        │ • Preprocessing pipeline
                        │ • Metadata storage
                        │ • fit/predict with preprocessing
                        │
```

### 2. Training Integration Flow

```
┌─────────────────────┐
│   User Creates      │
│   Custom Model      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Validate Model      │◄───── is_custom_model()
│ (sklearn API?)      │       validate_custom_model()
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Prepare CV Strategy │◄───── prepare_cv_strategy_for_custom()
│ (Stratified, Time)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Train with CV       │◄───── train_custom_model()
│ (cross_val_score)   │       cross_validate_custom_model()
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Hyperparameter      │◄───── GridSearchCV
│ Tuning (optional)   │       RandomizedSearchCV
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Return Trained      │
│ Model + Results     │
└─────────────────────┘
```

### 3. Evaluation Integration Flow

```
┌─────────────────────┐
│ Load Custom Model   │◄───── load_custom_model()
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Make Predictions    │◄───── model.predict()
│ (y_pred, y_proba)   │       model.predict_proba()
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Compute Metrics     │◄───── evaluate_custom_classifier()
│ (AUC, AUPRC, F1...) │       compute_classification_metrics()
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Generate Plots      │◄───── plot_roc_curve()
│ (ROC, PR, CM)       │       plot_confusion_matrix()
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Return Results      │
│ (dict with metrics) │
└─────────────────────┘
```

### 4. Explainability Integration Flow

```
┌─────────────────────┐
│ Custom Model        │
│ (trained)           │
└──────────┬──────────┘
           │
    ┌──────┴──────┬────────────────┬──────────────────┐
    │             │                │                  │
    ▼             ▼                ▼                  ▼
┌───────┐    ┌─────────┐    ┌──────────┐    ┌──────────────┐
│ SHAP  │    │ Permut. │    │ Feature  │    │ Individual   │
│ Values│    │ Import. │    │ Import.  │    │ Explanation  │
└───┬───┘    └────┬────┘    └────┬─────┘    └──────┬───────┘
    │             │              │                  │
    │ Uses:       │ Uses:        │ Uses:            │ Uses:
    │ • Kernel   │ • Sklearn    │ • Universal      │ • SHAP
    │   SHAP     │   permut.    │   extractor      │   waterfall
    │ • Tree     │ • Custom     │ • Model attrs    │ • Force plot
    │   SHAP     │   scoring    │   (coef_, etc.)  │
    │             │              │                  │
    └──────┬──────┴──────┬───────┴──────────┬───────┘
           │             │                  │
           ▼             ▼                  ▼
    ┌─────────────────────────────────────────┐
    │      Visualization & Interpretation     │
    └─────────────────────────────────────────┘
```

### 5. Persistence System

```
┌─────────────────────────────────────────────────────────┐
│                  save_custom_model()                    │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    ┌────────┐   ┌──────────┐   ┌───────────┐
    │ Model  │   │ Metadata │   │ Preproc.  │
    │ .pkl   │   │ .json    │   │ .pkl      │
    └────────┘   └──────────┘   └───────────┘
         │             │             │
         └─────────────┼─────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Manifest.json  │
              ├─────────────────┤
              │ • version       │
              │ • timestamp     │
              │ • checksums     │
              │ • file_paths    │
              └─────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  load_custom_model()                    │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    ┌────────┐   ┌──────────┐   ┌───────────┐
    │ Read   │   │ Read     │   │ Read      │
    │ Model  │   │ Metadata │   │ Preproc.  │
    └────┬───┘   └────┬─────┘   └─────┬─────┘
         │            │               │
         └────────────┼───────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │ Validate Model   │
            ├──────────────────┤
            │ • Check methods  │
            │ • Verify classes │
            │ • Test predict   │
            │ • Hash check     │
            └──────────────────┘
```

---

## Data Flow Diagram

### Complete Workflow: From Creation to Deployment

```
┌──────────────────────────────────────────────────────────────────┐
│                    Phase 1: Model Creation                       │
└──────────────────────────────────────────────────────────────────┘
                                │
                    User defines custom class
                    (inherits from BaseCustom*)
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Phase 2: Training                             │
└──────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Load Data    │      │ Preprocess   │      │ Validation   │
│              │─────►│ Features     │─────►│ Strategy     │
└──────────────┘      └──────────────┘      └──────────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────┐
                                          │ Cross Validation │
                                          │ (5-fold, etc.)   │
                                          └─────────┬────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────┐
                                          │ Trained Model    │
                                          └─────────┬────────┘
┌──────────────────────────────────────────────────┼────────────┐
│                    Phase 3: Persistence                        │
└────────────────────────────────────────────────────────────────┘
                                                     │
                                          save_custom_model()
                                                     │
                    ┌────────────────────────────────┼────────────────┐
                    │                                │                │
                    ▼                                ▼                ▼
        ┌──────────────────┐          ┌──────────────────┐  ┌───────────┐
        │ model.pkl        │          │ metadata.json    │  │ manifest  │
        │ (serialized)     │          │ (description,    │  │ (version, │
        │                  │          │  author, etc.)   │  │  hashes)  │
        └──────────────────┘          └──────────────────┘  └───────────┘
                                                     │
┌──────────────────────────────────────────────────┼────────────────┐
│                    Phase 4: Dashboard Integration                 │
└────────────────────────────────────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┼────────────────┐
                    │                                │                │
                    ▼                                ▼                ▼
        ┌──────────────────┐          ┌──────────────────┐  ┌───────────────┐
        │ Upload UI        │          │ Training UI      │  │ Evaluation UI │
        │ • Manage models  │          │ • Select custom  │  │ • Compare     │
        │ • View details   │          │ • Include in CV  │  │ • Metrics     │
        │ • Test/Delete    │          │ • Train mixed    │  │ • Plots       │
        └──────────────────┘          └──────────────────┘  └───────────────┘
                                                     │
┌──────────────────────────────────────────────────┼────────────────┐
│                    Phase 5: Evaluation & Explainability           │
└────────────────────────────────────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┼────────────────┐
                    │                                │                │
                    ▼                                ▼                ▼
        ┌──────────────────┐          ┌──────────────────┐  ┌───────────────┐
        │ Test Set Metrics │          │ SHAP Analysis    │  │ Feature Imp.  │
        │ • ROC-AUC        │          │ • Summary plot   │  │ • Permutation │
        │ • AUPRC          │          │ • Waterfall      │  │ • Tree-based  │
        │ • Confusion      │          │ • Force plot     │  │ • Coef-based  │
        └──────────────────┘          └──────────────────┘  └───────────────┘
```

---

## Module Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                      External Dependencies                  │
│  sklearn • numpy • pandas • shap • joblib • pathlib         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    src/models/custom_base.py                │
│  BaseCustomModel • BaseCustomClassifier • BaseCustomRegressor│
│  CustomModelWrapper                                         │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
    ┌──────┴─────┐                     ┌──────┴─────┐
    │            │                     │            │
    ▼            ▼                     ▼            ▼
┌────────┐  ┌─────────┐         ┌──────────┐  ┌──────────┐
│Training│  │Evaluation│        │Explain.  │  │Persist.  │
│Integ.  │  │Integ.    │        │Integ.    │  │System    │
└────┬───┘  └────┬────┘         └────┬─────┘  └────┬─────┘
     │           │                   │             │
     └───────────┼───────────────────┼─────────────┘
                 │                   │
                 ▼                   ▼
        ┌────────────────┐  ┌────────────────┐
        │  Dashboard UI  │  │  Test Suite    │
        │  (Streamlit)   │  │  (pytest)      │
        └────────────────┘  └────────────────┘
```

---

## File Organization

```
Tools/
├── src/
│   └── models/
│       ├── custom_base.py          ← Base classes
│       ├── persistence.py          ← Save/Load system
│       └── __init__.py             ← Exports
│
├── src/training/
│   └── custom_integration.py       ← Training functions
│
├── src/evaluation/
│   └── custom_integration.py       ← Evaluation functions
│
├── src/explainability/
│   └── custom_integration.py       ← Explainability functions
│
├── dashboard/pages/
│   ├── 07_🔧_Custom_Models.py      ← Model management UI
│   ├── 02_🤖_Model_Training.py     ← Training integration
│   └── 04_📈_Model_Evaluation.py   ← Evaluation integration
│
├── tests/
│   └── test_custom_models.py       ← 31 comprehensive tests
│
├── docs/
│   ├── CUSTOM_MODELS_GUIDE.md      ← Complete guide
│   └── CUSTOM_MODELS_ARCHITECTURE.md ← This file
│
└── models/custom/                   ← Saved models directory
    ├── model_name_1/
    │   ├── model.pkl
    │   ├── metadata.json
    │   ├── preprocessing.pkl
    │   └── manifest.json
    └── model_name_2/
        └── ...
```

---

## Interaction Patterns

### Pattern 1: Creating a New Custom Model

```
Developer
    │
    ├─► Define class (inherit BaseCustomClassifier)
    │
    ├─► Implement required methods
    │   ├── fit(X, y)
    │   ├── predict(X)
    │   ├── predict_proba(X)
    │   └── set classes_
    │
    ├─► Train on data
    │   └── model.fit(X_train, y_train)
    │
    ├─► Validate
    │   └── validate_custom_model(model)
    │
    └─► Save
        └── save_custom_model(model, path, metadata)
```

### Pattern 2: Using Saved Model in Dashboard

```
User (Dashboard)
    │
    ├─► Navigate to Custom Models page
    │
    ├─► Upload model
    │   ├── Select .pkl file
    │   ├── Fill metadata form
    │   └── Click "Upload"
    │
    ├─► Train with custom model
    │   ├── Go to Training page
    │   ├── Check "Include Custom Models"
    │   ├── Select model from dropdown
    │   └── Click "Start Training"
    │
    ├─► Evaluate
    │   ├── Go to Evaluation page
    │   ├── Select "Custom Models"
    │   ├── Choose model
    │   └── View metrics & plots
    │
    └─► Explain
        ├── Go to Explainability page
        ├── Select custom model
        └── View SHAP/Feature importance
```

### Pattern 3: Extending with Preprocessing

```
Developer
    │
    ├─► Create preprocessing pipeline
    │   └── StandardScaler() + PCA()
    │
    ├─► Wrap model
    │   └── CustomModelWrapper(model, preprocessing)
    │
    ├─► Train (auto-applies preprocessing)
    │   └── wrapper.fit(X_train, y_train)
    │
    ├─► Predict (auto-applies preprocessing)
    │   └── wrapper.predict(X_test)
    │
    └─► Save (includes preprocessing)
        └── save_custom_model(wrapper, path)
```

---

## Security & Validation

### Model Validation Checklist

```
load_custom_model(path, validate=True)
    │
    ├─► Check file existence
    │   └── model.pkl, metadata.json, manifest.json exist?
    │
    ├─► Verify checksums
    │   └── SHA256 hashes match manifest?
    │
    ├─► Validate sklearn API
    │   ├── Has fit() method?
    │   ├── Has predict() method?
    │   ├── Has get_params() method?
    │   └── Has set_params() method?
    │
    ├─► Check classifier requirements
    │   ├── Has predict_proba() method?
    │   └── Has classes_ attribute?
    │
    ├─► Test predictions
    │   ├── Can make predictions?
    │   ├── Output shape correct?
    │   └── Output type correct?
    │
    └─► Version compatibility
        └── Persistence version supported?
```

### Metadata Structure

```json
{
  "model_name": "MyCustomClassifier",
  "model_type": "classifier",
  "description": "Custom ensemble model",
  "algorithm": "Ensemble (RF + GB)",
  "author": "Research Team",
  "created_at": "2025-11-01T12:00:00",
  "training_date": "2025-11-01",
  "feature_names": ["age", "bp", "cholesterol", ...],
  "n_features": 25,
  "hyperparameters": {
    "n_layers": 5,
    "learning_rate": 0.01
  },
  "training_info": {
    "cv_scores": [0.85, 0.87, 0.86, 0.84, 0.88],
    "mean_cv_score": 0.86,
    "training_samples": 1000
  },
  "sklearn_compatible": true
}
```

---

## Performance Considerations

### Memory Management

```
┌────────────────────────────────────┐
│  Large Model (>500MB)              │
├────────────────────────────────────┤
│  Solution:                         │
│  • Use joblib compress parameter   │
│  • Stream data in batches          │
│  • Consider model quantization     │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│  Many Models Loaded                │
├────────────────────────────────────┤
│  Solution:                         │
│  • Load on-demand (lazy loading)   │
│  • Cache frequently used models    │
│  • Unload inactive models          │
└────────────────────────────────────┘
```

### SHAP Computation

```
┌────────────────────────────────────┐
│  Large Dataset (>10K samples)      │
├────────────────────────────────────┤
│  Solution:                         │
│  • Sample background data (100)    │
│  • Use TreeExplainer if possible   │
│  • Compute in batches              │
│  • Set max_samples parameter       │
└────────────────────────────────────┘
```

---

## Extension Points

### Adding New Functionality

1. **New Model Type (e.g., Survival Analysis)**
   ```python
   class BaseCustomSurvival(BaseCustomModel):
       def predict_survival(self, X, times):
           """Predict survival probabilities."""
           pass
   ```

2. **New Evaluation Metric**
   ```python
   def evaluate_with_custom_metric(model, X, y, metric_fn):
       y_pred = model.predict(X)
       return metric_fn(y, y_pred)
   ```

3. **New Explainability Method**
   ```python
   def compute_lime_for_custom_model(model, X, instance):
       # LIME implementation
       pass
   ```

4. **New Persistence Format**
   ```python
   def save_custom_model_onnx(model, path):
       # Convert to ONNX and save
       pass
   ```

---

## Version History

| Version | Date       | Changes                                      |
|---------|------------|----------------------------------------------|
| 1.0.0   | 2025-11-01 | Initial release of custom models system      |
|         |            | • Base classes                               |
|         |            | • Training/Evaluation/Explainability modules |
|         |            | • Persistence with versioning                |
|         |            | • Dashboard integration                      |
|         |            | • 31 comprehensive tests                     |

---

**Maintained by:** Research Team  
**Last Updated:** November 2025  
**License:** MIT
