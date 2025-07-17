"""
Training script for digit classification using Logistic Regression
"""
import json
import pickle
import os
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def train_model(X, y, config):
    """Train logistic regression model with given configuration"""
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter'],
        random_state=42
    )
    model.fit(X, y)
    return model


def main():
    # Load configuration
    config = load_config('config/config.json')
    
    # Load digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_model(X_train, y_train, config)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Training completed!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('model_train.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved as model_train.pkl")


if __name__ == "__main__":
    main()