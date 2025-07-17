# MLOps Artifact Pipeline

This repository contains a complete MLOps pipeline for digit classification using Logistic Regression with automated CI/CD workflows using GitHub Actions.

## Project Overview

This project implements a comprehensive MLOps pipeline that demonstrates:
- Parameterized model training with JSON configuration
- Comprehensive unit and integration testing
- Automated CI/CD workflows using GitHub Actions
- Artifact management and job dependencies
- Model training, testing, and inference automation

## Dataset and Model
- **Dataset**: sklearn.datasets.load_digits (built-in, no download required)
- **Task**: Multiclass classification (digits 0–9)
- **Features**: 64 grayscale pixel values (flattened 8x8 images)
- **Model**: LogisticRegression from sklearn.linear_model

## Project Structure
```
.
├── src/
│   ├── train.py          # Training script with configurable parameters
│   ├── inference.py      # Inference script for model predictions
│   └── utils.py          # Utility functions for data loading and model handling
├── config/
│   └── config.json       # Model hyperparameters configuration
├── tests/
│   └── test_train.py     # Comprehensive test suite
├── .github/
│   └── workflows/
│       ├── train.yml     # Training pipeline workflow
│       ├── test.yml      # Testing pipeline workflow
│       └── inference.yml # Complete MLOps workflow (test → train → inference)
├── requirements.txt      # Python dependencies
├── model_train.pkl       # Trained model (generated)
└── README.md
```

## Branching Strategy
The project follows a linear branching approach as specified:
```
main → classification_branch → test_branch → inference_branch
```

Each branch builds upon the previous one without merging back to main, ensuring clean development history and functional CI workflows across all branches.

## Features

### Phase 1: Training Pipeline (classification_branch)
- **Configurable Training**: All hyperparameters loaded from `config/config.json`
- **Model Persistence**: Trained model saved as `model_train.pkl`
- **Performance Metrics**: Accuracy and F1-score reporting
- **GitHub Actions**: Automated training workflow with artifact upload

### Phase 2: Testing Pipeline (test_branch)
- **Comprehensive Test Suite**: 9 test cases covering:
  - Configuration file loading and validation
  - Model creation and fitting verification
  - Accuracy threshold validation
  - Model persistence testing
- **Automated Testing**: GitHub Actions workflow for continuous testing

### Phase 3: Complete MLOps Pipeline (inference_branch)
- **Inference Pipeline**: Automated model loading and prediction
- **Multi-Job Workflow**: Three dependent jobs (test → train → inference)
- **Artifact Management**: Model artifacts passed between jobs
- **End-to-End Automation**: Complete pipeline from testing to inference

## Configuration

The model hyperparameters are defined in `config/config.json`:
```json
{
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000
}
```

## Performance Results

The model achieves excellent performance on the digits dataset:
- **Training Accuracy**: ~97.5%
- **Training F1-Score**: ~97.5%
- **Inference Accuracy**: ~99.5%
- **Inference F1-Score**: ~99.5%

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mlops-artifact-pipeline
   ```

2. **Create conda environment**:
   ```bash
   conda create -n mlops-env python=3.9
   conda activate mlops-env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run training**:
   ```bash
   python src/train.py
   ```

5. **Run tests**:
   ```bash
   python -m pytest tests/ -v
   ```

6. **Run inference**:
   ```bash
   python src/inference.py
   ```

## GitHub Actions Workflows

### 1. Training Workflow (`train.yml`)
- Triggers on push to `classification_branch`
- Sets up Python environment
- Installs dependencies
- Runs training script
- Uploads trained model as artifact

### 2. Testing Workflow (`test.yml`)
- Triggers on push to `test_branch`
- Runs comprehensive test suite
- Validates all components of the training pipeline

### 3. Complete MLOps Workflow (`inference.yml`)
- Triggers on push to `inference_branch`
- **Job 1 (test)**: Runs all test cases
- **Job 2 (train)**: Trains model (depends on test job)
- **Job 3 (inference)**: Downloads model and runs inference (depends on train job)

## Testing Coverage

The test suite includes:
- **Configuration Tests**: File existence, loading, parameter validation
- **Model Tests**: Creation, fitting, type validation
- **Performance Tests**: Accuracy threshold validation
- **Persistence Tests**: Model saving and loading verification

## Key Features

- ✅ **Parameterized Training**: JSON-based configuration
- ✅ **Comprehensive Testing**: 9 test cases with pytest
- ✅ **CI/CD Pipeline**: Automated workflows with job dependencies
- ✅ **Artifact Management**: Model artifacts passed between jobs
- ✅ **Performance Monitoring**: Accuracy and F1-score tracking
- ✅ **Code Modularity**: Clean separation of concerns
- ✅ **Reproducibility**: Consistent results with fixed random seeds

## Authors
- Shubham Bagwari: p22cs201@iitj.ac.in
- Divyaansh Mertia: m23cse013@iitj.ac.in