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
        self.label_encoder = None
        self.label_names = None
        self.classification_mode = 'binary'  # Default to binary
        os.makedirs(models_dir, exist_ok=True)
    
    def train_and_evaluate(self, train_df: pd.DataFrame, n_samples: int = 15000, classification_mode: str = None):
        """
        Train and evaluate multiple models
        
        Args:
            train_df: Training data
            n_samples: Number of samples to use for training
            classification_mode: 'binary' or 'multiclass' (defaults to self.classification_mode)
        """
        print("="*70)
        print("INTRUSION DETECTION MODEL TRAINING")
        print("="*70)
        
        # Use provided classification_mode or default from self
        if classification_mode is not None:
            self.classification_mode = classification_mode
        
        # Process data
        processor = DataProcessor()
        processor.classification_mode = self.classification_mode
        
        # Create sample dataset if needed
        if n_samples and len(train_df) > n_samples:
            train_df = processor.create_sample_dataset(train_df, n_samples=n_samples)
        
        # Preprocess
        print("\n1. PREPROCESSING DATA")
        print("-" * 70)
        print(f"Classification Mode: {self.classification_mode.upper()}")
        X, y = processor.preprocess_data(train_df, classification_mode=self.classification_mode)
        self.feature_names = processor.get_feature_names()
        
        # Save label encoder and names for multiclass mode
        if self.classification_mode == 'multiclass':
            self.label_encoder = processor.label_encoder
            self.label_names = processor.get_all_label_names()
        
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
        
        # Calculate class weights for imbalanced multiclass datasets
        if self.classification_mode == 'multiclass':
            from sklearn.utils.class_weight import compute_class_weight
            unique_classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
            class_weight_dict = dict(zip(unique_classes, class_weights))
            print(f"  Using balanced class weights for imbalanced multiclass dataset")
            print(f"  Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        else:
            class_weight_dict = 'balanced'  # For binary classification
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight_dict if self.classification_mode == 'multiclass' else 'balanced',
                random_state=42,
                n_jobs=-1,
                verbose=0
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=50 if self.classification_mode == 'multiclass' else 100,  # Reduce for multiclass speed
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=1 if self.classification_mode == 'multiclass' else 0  # Show progress for multiclass
            ),
            'NeuralNetwork': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                verbose=False,
                early_stopping=True,  # Prevent overfitting
                validation_fraction=0.1  # Use 10% for validation
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
        # Use actual label names if available, otherwise use generic names
        if self.classification_mode == 'multiclass' and self.label_names:
            target_names = self.label_names
        elif self.classification_mode == 'binary':
            target_names = ['Benign', 'Attack']
        else:
            # Fallback: use class indices
            n_classes = len(np.unique(y_test))
            target_names = [f'Class_{i}' for i in range(n_classes)]
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Save model
        print("\n7. SAVING MODEL")
        print("-" * 70)
        self.save_model(results)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
    
    def save_model(self, results: dict):
        """Save trained models and artifacts"""
        best_info = results[self.best_model_name]

        # Save best model and scaler
        model_path = os.path.join(self.models_dir, 'intrusion_detection_model.pkl')
        model_artifacts = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'classification_mode': self.classification_mode
        }
        # Only save label encoder and names for multiclass mode
        if self.classification_mode == 'multiclass':
            model_artifacts['label_encoder'] = self.label_encoder
            model_artifacts['label_names'] = self.label_names
        joblib.dump(model_artifacts, model_path)
        print(f"✓ Model saved to: {model_path}")

        # Save feature names
        feature_names_path = os.path.join(self.models_dir, 'feature_names.json')
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"✓ Feature names saved to: {feature_names_path}")

        # Save all trained models for comparison
        candidates_path = os.path.join(self.models_dir, 'model_candidates.pkl')
        candidates_artifacts = {
            'models': {name: info['model'] for name, info in results.items()},
            'scaler': self.scaler,
            'best_model': self.best_model_name,
            'feature_names': self.feature_names,
            'classification_mode': self.classification_mode
        }
        # Only save label encoder and names for multiclass mode
        if self.classification_mode == 'multiclass':
            candidates_artifacts['label_encoder'] = self.label_encoder
            candidates_artifacts['label_names'] = self.label_names
        joblib.dump(candidates_artifacts, candidates_path)
        print(f"✓ All trained models saved to: {candidates_path}")

        # Prepare metrics summary for JSON
        all_metrics = {
            name: {
                'train_accuracy': info['train_accuracy'],
                'test_accuracy': info['test_accuracy'],
                'precision': info['precision'],
                'recall': info['recall'],
                'f1_score': info['f1_score']
            }
            for name, info in results.items()
        }

        # Save model info summary
        model_info_path = os.path.join(self.models_dir, 'model_info.json')
        info = {
            'model_type': self.best_model_name,
            'classification_mode': self.classification_mode,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'metrics': {
                'accuracy': best_info['test_accuracy'],
                'precision': best_info['precision'],
                'recall': best_info['recall'],
                'f1_score': best_info['f1_score']
            },
            'all_models_metrics': all_metrics
        }
        # Add label information for multiclass mode
        if self.classification_mode == 'multiclass' and self.label_names:
            info['label_names'] = self.label_names
            info['n_classes'] = len(self.label_names)
        with open(model_info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✓ Model info saved to: {model_info_path}")


def main():
    """Main training pipeline"""
    import sys
    
    # Check command-line arguments
    classification_mode = 'binary'  # Default to binary
    models_dir = 'models'  # Default directory
    n_samples = 100000  # Default sample size
    
    if '--multiclass' in sys.argv:
        classification_mode = 'multiclass'
        models_dir = 'models_multiclass_size_100000'  # Default for multiclass
    elif '--binary' in sys.argv:
        classification_mode = 'binary'
        models_dir = 'models_size_100000'  # Default for binary
    
    # Check for custom models directory
    if '--models-dir' in sys.argv:
        idx = sys.argv.index('--models-dir')
        if idx + 1 < len(sys.argv):
            models_dir = sys.argv[idx + 1]
    
    # Check for custom sample size
    if '--n-samples' in sys.argv:
        idx = sys.argv.index('--n-samples')
        if idx + 1 < len(sys.argv):
            try:
                n_samples = int(sys.argv[idx + 1])
            except ValueError:
                print("Warning: Invalid n_samples, using default 100000")
                n_samples = 100000
    
    print(f"Classification Mode: {classification_mode.upper()}")
    print(f"Models Directory: {models_dir}")
    print(f"Sample Size: {n_samples:,}")
    
    # Initialize
    processor = DataProcessor()
    processor.classification_mode = classification_mode
    trainer = IntrusionDetectionTrainer(models_dir=models_dir)
    trainer.classification_mode = classification_mode
    
    # Load training data
    print("\nLoading training data...")
    train_df = processor.load_training_data()
    
    # Train model
    trainer.train_and_evaluate(train_df, n_samples=n_samples, classification_mode=classification_mode)
    
    print("\n✓ Model training completed successfully!")
    print(f"You can now use the trained {classification_mode} model for predictions.")


if __name__ == "__main__":
    main()
