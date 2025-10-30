"""
Train Intrusion Detection ML Models
Tests multiple algorithms and selects the best performing one
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_processor import DataProcessor
import warnings
warnings.filterwarnings('ignore')


class IntrusionDetectionTrainer:
    """Train and evaluate intrusion detection models"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.scaler = None
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        os.makedirs(models_dir, exist_ok=True)
    
    def train_and_evaluate(self, train_df: pd.DataFrame, n_samples: int = 100000):
        """
        Train and evaluate multiple models
        
        Args:
            train_df: Training data
            n_samples: Number of samples to use for training
        """
        print("="*70)
        print("INTRUSION DETECTION MODEL TRAINING")
        print("="*70)
        
        # Process data
        processor = DataProcessor()
        
        # Create sample dataset if needed
        if n_samples and len(train_df) > n_samples:
            train_df = processor.create_sample_dataset(train_df, n_samples=n_samples)
        
        # Preprocess
        print("\n1. PREPROCESSING DATA")
        print("-" * 70)
        X, y = processor.preprocess_data(train_df)
        self.feature_names = processor.get_feature_names()
        
        # Train-test split
        print("\n2. SPLITTING DATA")
        print("-" * 70)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Scale features
        print("\n3. SCALING FEATURES")
        print("-" * 70)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("✓ Features scaled")
        
        # Train multiple models
        print("\n4. TRAINING MODELS")
        print("-" * 70)
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=0
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=0
            ),
            'NeuralNetwork': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                verbose=False
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, average='weighted')
            recall = recall_score(y_test, y_pred_test, average='weighted')
            f1 = f1_score(y_test, y_pred_test, average='weighted')
            
            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy: {test_acc:.4f}")
            print(f"  F1-Score: {f1:.4f}")
        
        # Select best model
        print("\n5. MODEL SELECTION")
        print("-" * 70)
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"Best Model: {best_model_name}")
        print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
        print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
        
        # Detailed evaluation of best model
        print("\n6. DETAILED EVALUATION (Best Model)")
        print("-" * 70)
        y_pred = results[best_model_name]['model'].predict(X_test_scaled)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Save model
        print("\n7. SAVING MODEL")
        print("-" * 70)
        self.save_model(results[best_model_name])
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
    
    def save_model(self, model_info: dict):
        """Save the best model and artifacts"""
        # Save model and scaler
        model_path = os.path.join(self.models_dir, 'intrusion_detection_model.pkl')
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name
        }, model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Save feature names
        feature_names_path = os.path.join(self.models_dir, 'feature_names.json')
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"✓ Feature names saved to: {feature_names_path}")
        
        # Save model info
        model_info_path = os.path.join(self.models_dir, 'model_info.json')
        info = {
            'model_type': self.best_model_name,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'metrics': {
                'accuracy': model_info['test_accuracy'],
                'precision': model_info['precision'],
                'recall': model_info['recall'],
                'f1_score': model_info['f1_score']
            }
        }
        with open(model_info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✓ Model info saved to: {model_info_path}")


def main():
    """Main training pipeline"""
    # Initialize
    processor = DataProcessor()
    trainer = IntrusionDetectionTrainer()
    
    # Load training data
    print("Loading training data...")
    train_df = processor.load_training_data()
    
    # Train model
    trainer.train_and_evaluate(train_df, n_samples=100000)
    
    print("\n✓ Model training completed successfully!")
    print("You can now use the trained model for predictions.")


if __name__ == "__main__":
    main()
