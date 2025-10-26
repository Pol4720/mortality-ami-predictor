# GitHub Actions CI/CD Configuration

This directory contains the GitHub Actions workflows for continuous integration and deployment.

## Workflows

### CI Workflow (`ci.yml`)
Runs on every push and pull request to `main` and `develop` branches.

**Jobs:**
1. **Test** - Runs test suite on multiple Python versions
   - Python 3.9, 3.10, 3.11, 3.12
   - Includes code coverage reporting
   - Uploads coverage to Codecov

2. **Lint** - Checks code quality
   - Black formatting
   - isort import sorting
   - flake8 linting

## Test Results

Current status: ✅ **41 tests passing**

### Test Coverage
- `test_data_cleaning.py`: 25 tests - Data cleaning and preprocessing
- `test_data_preprocess.py`: 1 test - Preprocessing pipeline
- `test_model_io.py`: 1 test - Model serialization
- `test_modular_structure.py`: 13 tests - Module integration
- `test_smoke_train.py`: 1 test - End-to-end training

## Running Locally

```bash
cd Tools
pytest tests/ -v
```

For coverage report:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

Then open `htmlcov/index.html` in your browser.
