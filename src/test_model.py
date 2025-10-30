"""
Test Trained Model on Testing Data
Evaluates model performance on unseen test data
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_processor import DataProcessor
import warnings
warnings.filterwarnings('ignore')


class ModelTester:
    """Test trained intrusion detection model"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def load_model(self):
        """Load trained model and artifacts"""
        print("Loading trained model...")
        
        # Load model
        model_path = os.path.join(self.models_dir, 'intrusion_detection_model.pkl')
        with open(model_path, 'rb') as f:
            artifacts = joblib.load(f)
        
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        print(f"✓ Model loaded: {artifacts['model_name']}")
        
        # Load feature names
        feature_names_path = os.path.join(self.models_dir, 'feature_names.json')
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)
        print(f"✓ Features loaded: {len(self.feature_names)}")
    
    def evaluate_on_test_data(self, test_df: pd.DataFrame, max_samples: int = None):
        """
        Evaluate model on test data
        
        Args:
            test_df: Test data
            max_samples: Maximum number of samples to test on
        """
        print("="*70)
        print("MODEL EVALUATION ON TEST DATA")
        print("="*70)
        
        # Sample test data if needed
        if max_samples and len(test_df) > max_samples:
            print(f"\nSampling {max_samples:,} test records...")
            test_df = test_df.sample(n=max_samples, random_state=42)
        
        # Process test data
        processor = DataProcessor()
        
        print("\nPreprocessing test data...")
        X_test, y_test = processor.preprocess_data(test_df)
        
        # Ensure feature alignment
        missing_features = set(self.feature_names) - set(X_test.columns)
        extra_features = set(X_test.columns) - set(self.feature_names)
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing in test data")
            for feat in missing_features:
                X_test[feat] = 0
        
        if extra_features:
            print(f"Warning: {len(extra_features)} extra features in test data")
            X_test = X_test.drop(columns=list(extra_features))
        
        # Reorder columns to match training
        X_test = X_test[self.feature_names]
        
        # Scale features
        print("Scaling features...")
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        print("Making predictions...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Display results
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        
        print(f"\nTest Records: {len(y_test):,}")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        print("\nClassification Report:")
        print("-" * 70)
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
        
        print("\nConfusion Matrix:")
        print("-" * 70)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"True Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives: {cm[1,1]}")
        
        # Calculate attack detection rate
        if cm[1,1] + cm[1,0] > 0:
            attack_detection_rate = cm[1,1] / (cm[1,1] + cm[1,0])
            print(f"\nAttack Detection Rate: {attack_detection_rate:.4f} ({attack_detection_rate*100:.2f}%)")
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }


def main():
    """Main testing pipeline"""
    # Initialize
    processor = DataProcessor()
    tester = ModelTester()
    
    # Load model
    tester.load_model()
    
    # Load test data
    print("\nLoading test data...")
    test_df = processor.load_test_data()
    print(f"Test data shape: {test_df.shape}")
    
    # Evaluate
    results = tester.evaluate_on_test_data(test_df, max_samples=10000)
    
    print(f"\n✓ Model evaluation completed!")
    print(f"Final Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")


if __name__ == "__main__":
    main()

