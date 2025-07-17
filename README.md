# MLOps Artifact Pipeline

This repository contains a complete MLOps pipeline for digit classification using Logistic Regression with automated CI/CD workflows using GitHub Actions.

## Project Structure
```
.
├── src/
│   ├── train.py
│   ├── inference.py
│   └── utils.py
├── config/
│   └── config.json
├── tests/
│   └── test_train.py
├── .github/
│   └── workflows/
│       ├── train.yml
│       ├── test.yml
│       └── inference.yml
├── requirements.txt
└── README.md
```

## Dataset and Model
- **Dataset**: sklearn.datasets.load_digits (built-in, no download required)
- **Task**: Multiclass classification (digits 0–9)
- **Features**: 64 grayscale pixel values (flattened 8x8 images)
- **Model**: LogisticRegression from sklearn.linear_model

## Branching Strategy
The project follows a linear branching approach:
main → classification_branch → test_branch → inference_branch

## Setup Instructions
1. Clone the repository
2. Create a conda environment
3. Install dependencies: `pip install -r requirements.txt`
4. Run training: `python src/train.py`
5. Run tests: `pytest tests/`
6. Run inference: `python src/inference.py`