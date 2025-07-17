"""
Utility functions for the MLOps pipeline
"""
import json
import pickle
import numpy as np
from sklearn.datasets import load_digits


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_model(model, filepath):
    """Save model using pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """Load model using pickle"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def get_digits_data():
    """Load and return digits dataset"""
    digits = load_digits()
    return digits.data, digits.target