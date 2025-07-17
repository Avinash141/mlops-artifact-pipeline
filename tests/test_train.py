"""
Unit tests for the training pipeline
"""
import json
import os
import pytest
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from src.train import load_config, train_model


class TestConfigurationLoading:
    """Test configuration file loading"""
    
    def test_config_file_exists(self):
        """Test that config file exists"""
        assert os.path.exists('config/config.json'), "Configuration file does not exist"
    
    def test_config_loads_successfully(self):
        """Test that configuration loads without errors"""
        config = load_config('config/config.json')
        assert config is not None, "Configuration failed to load"
        assert isinstance(config, dict), "Configuration should be a dictionary"
    
    def test_required_hyperparameters_exist(self):
        """Test that all required hyperparameters exist"""
        config = load_config('config/config.json')
        
        required_params = ['C', 'solver', 'max_iter']
        for param in required_params:
            assert param in config, f"Required parameter '{param}' missing from config"
    
    def test_hyperparameter_data_types(self):
        """Test that hyperparameters have correct data types"""
        config = load_config('config/config.json')
        
        assert isinstance(config['C'], (int, float)), "C should be a number"
        assert isinstance(config['solver'], str), "solver should be a string"
        assert isinstance(config['max_iter'], int), "max_iter should be an integer"


class TestModelCreation:
    """Test model creation and training"""
    
    def setup_method(self):
        """Setup test data"""
        self.digits = load_digits()
        self.X, self.y = self.digits.data, self.digits.target
        self.config = load_config('config/config.json')
    
    def test_train_model_returns_logistic_regression(self):
        """Test that train_model returns a LogisticRegression object"""
        model = train_model(self.X, self.y, self.config)
        assert isinstance(model, LogisticRegression), "train_model should return LogisticRegression object"
    
    def test_model_is_fitted(self):
        """Test that the returned model is fitted"""
        model = train_model(self.X, self.y, self.config)
        
        # Check if model has been fitted by verifying fitted attributes exist
        assert hasattr(model, 'coef_'), "Model should have coef_ attribute after fitting"
        assert hasattr(model, 'classes_'), "Model should have classes_ attribute after fitting"
        assert model.coef_ is not None, "Model coefficients should not be None"
        assert model.classes_ is not None, "Model classes should not be None"


class TestModelAccuracy:
    """Test model accuracy and performance"""
    
    def setup_method(self):
        """Setup test data"""
        self.digits = load_digits()
        self.X, self.y = self.digits.data, self.digits.target
        self.config = load_config('config/config.json')
    
    def test_model_accuracy_threshold(self):
        """Test that model accuracy is above threshold"""
        model = train_model(self.X, self.y, self.config)
        accuracy = model.score(self.X, self.y)
        
        # Model should achieve at least 90% accuracy on training data
        assert accuracy > 0.90, f"Model accuracy {accuracy:.4f} is below threshold of 0.90"
    
    def test_model_can_predict(self):
        """Test that model can make predictions"""
        model = train_model(self.X, self.y, self.config)
        predictions = model.predict(self.X[:10])  # Test on first 10 samples
        
        assert len(predictions) == 10, "Should return 10 predictions"
        assert all(0 <= pred <= 9 for pred in predictions), "Predictions should be between 0 and 9"


class TestModelPersistence:
    """Test model saving and loading"""
    
    def setup_method(self):
        """Setup test data"""
        self.digits = load_digits()
        self.X, self.y = self.digits.data, self.digits.target
        self.config = load_config('config/config.json')
        self.test_model_path = 'test_model.pkl'
    
    def teardown_method(self):
        """Clean up test files"""
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)
    
    def test_model_can_be_saved_and_loaded(self):
        """Test that model can be saved and loaded correctly"""
        # Train and save model
        model = train_model(self.X, self.y, self.config)
        with open(self.test_model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Load model
        with open(self.test_model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Test that loaded model works
        original_pred = model.predict(self.X[:5])
        loaded_pred = loaded_model.predict(self.X[:5])
        
        assert all(original_pred == loaded_pred), "Loaded model should make same predictions as original"