# Mortality-AMI-Predictor

[![CI](https://github.com/Pol4720/mortality-ami-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Pol4720/mortality-ami-predictor/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-41%20passed-brightgreen)](https://github.com/Pol4720/mortality-ami-predictor/tree/main/Tools/tests)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Final project for the **Machine Learning** course in the last year of the **Bachelor's Degree in Computer Science**.

## 🧠 Project Overview
This project focuses on developing predictive models for **in-hospital mortality in patients with acute myocardial infarction (AMI)** using modern **machine learning techniques**.  
It also explores the prediction of **ventricular arrhythmias** in the hospital setting as a secondary task.

The study is based on a real clinical dataset of approximately **4,500 patients**, including demographic, clinical, laboratory, and electrocardiographic variables.

## 🎯 Objectives
- Analyze and summarize the state of the art on AMI mortality prediction.
- Preprocess and model clinical data using ML algorithms (Logistic Regression, Random Forest, XGBoost, Neural Networks).
- Evaluate models through discrimination, calibration, and clinical utility metrics.
- Compare results against classical clinical scores (e.g., GRACE, TIMI).
- Ensure model interpretability using SHAP and LIME.

## 🛠️ Development & Testing

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12
- pip package manager

### Installation
```bash
cd Tools
pip install -r requirements.txt
```

### Running Tests
The project includes comprehensive unit tests to ensure code quality and reliability:

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test module
pytest tests/test_data_cleaning.py -v
```

### Continuous Integration
All code changes are automatically tested using GitHub Actions CI/CD pipeline:
- ✅ Tests run on Python 3.9, 3.10, 3.11, and 3.12
- ✅ Code quality checks (flake8, black, isort)
- ✅ Coverage reporting
- ✅ Automated on every push and pull request

Check the [CI status](https://github.com/Pol4720/mortality-ami-predictor/actions) for the latest build results.

## 📊 Project Structure
```
mortality-ami-predictor/
├── Tools/
│   ├── src/              # Source code modules
│   │   ├── cleaning/     # Data cleaning & preprocessing
│   │   ├── eda/          # Exploratory data analysis
│   │   ├── models/       # ML model implementations
│   │   ├── evaluation/   # Model evaluation metrics
│   │   └── explainability/ # Model interpretation (SHAP, LIME)
│   ├── tests/            # Unit tests (41 tests, 100% passing)
│   ├── notebooks/        # Jupyter notebooks for analysis
│   └── dashboard/        # Streamlit dashboard application
├── DATA/                 # Clinical datasets
└── Report/               # Project documentation and reports
```

 
