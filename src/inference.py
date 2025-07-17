"""
Inference script for digit classification using trained Logistic Regression model
"""
import pickle
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_model(model_path):
    """Load trained model from pickle file"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def main():
    # Load the trained model
    try:
        model = load_model('model_train.pkl')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: model_train.pkl not found. Please run training first.")
        return
    
    # Load digits dataset for inference
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions, average='weighted')
    
    print("\n=== Inference Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Show some sample predictions
    print("\n=== Sample Predictions ===")
    sample_indices = [0, 100, 200, 300, 400]
    for i in sample_indices:
        print(f"Sample {i}: True={y[i]}, Predicted={predictions[i]}")
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y, predictions))
    
    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()