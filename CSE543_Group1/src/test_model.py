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
from visualize_metrics import (plot_confusion_matrix_heatmap, plot_model_comparison,
                               plot_metrics_table, generate_all_visualizations)
import warnings
warnings.filterwarnings('ignore')


class ModelTester:
    """Test trained intrusion detection model"""
    
    def __init__(self, models_dir: str = 'models', classification_mode: str = 'auto'):
        self.models_dir = models_dir
        self.classification_mode = classification_mode  # 'auto', 'binary', or 'multiclass'
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.all_models = None
        self.best_model_name = None
        self.label_encoder = None
        self.label_names = None
    
    def load_model(self):
        """Load trained model and artifacts"""
        print("Loading trained model...")
        
        # Load model
        model_path = os.path.join(self.models_dir, 'intrusion_detection_model.pkl')
        with open(model_path, 'rb') as f:
            artifacts = joblib.load(f)
        
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.best_model_name = artifacts.get('model_name', 'LoadedModel')
        
        # Auto-detect or use specified classification mode
        if self.classification_mode == 'auto':
            # Check if label_encoder exists (multiclass) or not (binary)
            if 'label_encoder' in artifacts and artifacts['label_encoder'] is not None:
                self.classification_mode = 'multiclass'
            elif 'classification_mode' in artifacts:
                self.classification_mode = artifacts['classification_mode']
            else:
                # Fallback: assume binary for old models without classification_mode field
                self.classification_mode = 'binary'
        else:
            # Use specified mode
            self.classification_mode = self.classification_mode
        
        # Load label encoder and names for multiclass mode
        if self.classification_mode == 'multiclass':
            self.label_encoder = artifacts.get('label_encoder', None)
            self.label_names = artifacts.get('label_names', None)
        
        print(f"✓ Model loaded: {self.best_model_name} ({self.classification_mode.upper()} mode)")
        if self.label_names:
            print(f"✓ Label classes: {self.label_names}")
        
        # Load feature names
        feature_names_path = os.path.join(self.models_dir, 'feature_names.json')
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)
        print(f"✓ Features loaded: {len(self.feature_names)}")

        # Load all trained models if available
        candidates_path = os.path.join(self.models_dir, 'model_candidates.pkl')
        if os.path.exists(candidates_path):
            try:
                with open(candidates_path, 'rb') as f:
                    candidates = joblib.load(f)
                models_dict = candidates.get('models')
                if models_dict:
                    self.all_models = models_dict
                    print(f"✓ Loaded {len(self.all_models)} trained models for comparison")
                candidate_scaler = candidates.get('scaler')
                if self.scaler is None and candidate_scaler is not None:
                    self.scaler = candidate_scaler
                candidate_features = candidates.get('feature_names')
                if candidate_features and not self.feature_names:
                    self.feature_names = candidate_features
                candidate_label_encoder = candidates.get('label_encoder')
                if candidate_label_encoder and not self.label_encoder:
                    self.label_encoder = candidate_label_encoder
                candidate_label_names = candidates.get('label_names')
                if candidate_label_names and not self.label_names:
                    self.label_names = candidate_label_names
                candidate_mode = candidates.get('classification_mode')
                if candidate_mode and self.classification_mode == 'auto':
                    self.classification_mode = candidate_mode
            except Exception as exc:
                print(f"⚠️ Could not load model_candidates.pkl: {exc}")
    
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

        if max_samples and len(test_df) > max_samples:
            print(f"\nSampling {max_samples:,} test records...")
            test_df = test_df.sample(n=max_samples, random_state=42)

        processor = DataProcessor()
        
        # Use detected or specified classification mode
        classification_mode = self.classification_mode if self.classification_mode != 'auto' else 'binary'
        processor.classification_mode = classification_mode
        
        # Load label encoder from saved model if available (for multiclass mode)
        if classification_mode == 'multiclass' and self.label_encoder is not None:
            processor.label_encoder = self.label_encoder
            processor.label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)} if hasattr(self.label_encoder, 'classes_') else None

        print("\nPreprocessing test data...")
        print(f"Classification Mode: {classification_mode.upper()}")
        X_test, y_test = processor.preprocess_data(test_df, classification_mode=classification_mode)

        missing_features = set(self.feature_names) - set(X_test.columns)
        extra_features = set(X_test.columns) - set(self.feature_names)

        if missing_features:
            print(f"Warning: {len(missing_features)} features missing in test data")
            for feat in missing_features:
                X_test[feat] = 0

        if extra_features:
            print(f"Warning: {len(extra_features)} extra features in test data")
            X_test = X_test.drop(columns=list(extra_features))

        X_test = X_test[self.feature_names]

        # Ensure all features are numeric (convert strings to numeric, errors become NaN)
        print("  Converting features to numeric...")
        for col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        
        # Fill any NaN created by conversion
        X_test = X_test.fillna(0)
        
        # Final validation: ensure all columns are numeric
        non_numeric_cols = X_test.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"  Warning: Found {len(non_numeric_cols)} non-numeric columns: {list(non_numeric_cols)}")
            for col in non_numeric_cols:
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
        
        # Ensure X_test is purely numeric
        X_test = X_test.select_dtypes(include=[np.number])

        print("Scaling features...")
        X_test_scaled = self.scaler.transform(X_test)

        if self.all_models:
            models_to_evaluate = self.all_models
        else:
            fallback_name = self.best_model_name or self.model.__class__.__name__
            models_to_evaluate = {fallback_name: self.model}

        print("Evaluating models...")

        comparison_results = {}
        detailed_model_name = self.best_model_name or next(iter(models_to_evaluate.keys()))
        best_confusion = None
        best_predictions = None
        best_metrics = None

        for model_name, model in models_to_evaluate.items():
            print(f"\n▶ {model_name}")
            y_pred = model.predict(X_test_scaled)

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            comparison_results[model_name] = metrics

            if model_name == detailed_model_name:
                best_confusion = confusion_matrix(y_test, y_pred)
                best_predictions = y_pred
                best_metrics = metrics

        print("="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(f"{'Model':<18}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1-Score':>12}")
        print('-' * 70)
        for name, metrics in comparison_results.items():
            print(f"{name:<18}{metrics['accuracy']:>10.4f}{metrics['precision']:>12.4f}{metrics['recall']:>10.4f}{metrics['f1_score']:>12.4f}")

        if best_metrics is None:
            detailed_model_name, best_metrics = next(iter(comparison_results.items()))
            model = models_to_evaluate[detailed_model_name]
            best_predictions = model.predict(X_test_scaled)
            best_confusion = confusion_matrix(y_test, best_predictions)

        print(f"\nDetailed results for: {detailed_model_name}")
        print('-' * 70)
        print(f"Test Records: {len(y_test):,}")
        print(f"Accuracy:  {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall:    {best_metrics['recall']:.4f}")
        print(f"F1-Score:  {best_metrics['f1_score']:.4f}")

        print("\nClassification Report:")
        print('-' * 70)
        # Use actual label names if available, otherwise use generic names
        if classification_mode == 'multiclass' and self.label_names:
            target_names = self.label_names
        elif classification_mode == 'binary':
            target_names = ['Benign', 'Attack']
        else:
            # Fallback: use class indices
            n_classes = len(np.unique(y_test))
            target_names = [f'Class_{i}' for i in range(n_classes)]
        print(classification_report(y_test, best_predictions, target_names=target_names, zero_division=0))

        print("\nConfusion Matrix:")
        print('-' * 70)
        print(best_confusion)
        
        # Only show TN/FP/FN/TP for binary classification
        current_mode = getattr(self, '_current_classification_mode', classification_mode)
        if current_mode == 'binary' and best_confusion.shape == (2, 2):
            print(f"True Negatives: {best_confusion[0,0]}")
            print(f"False Positives: {best_confusion[0,1]}")
            print(f"False Negatives: {best_confusion[1,0]}")
            print(f"True Positives: {best_confusion[1,1]}")
            
            if best_confusion[1,1] + best_confusion[1,0] > 0:
                attack_detection_rate = best_confusion[1,1] / (best_confusion[1,1] + best_confusion[1,0])
                print(f"\nAttack Detection Rate: {attack_detection_rate:.4f} ({attack_detection_rate*100:.2f}%)")
        elif current_mode == 'multiclass' or best_confusion.shape[0] > 2:
            # For multiclass, show per-class statistics
            print(f"\nPer-Class Accuracy:")
            for i in range(len(best_confusion)):
                row_sum = best_confusion[i].sum()
                label_name = target_names[i] if i < len(target_names) else f'Class_{i}'
                if row_sum > 0:
                    class_accuracy = best_confusion[i, i] / row_sum
                    print(f"  {label_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - "
                          f"{best_confusion[i, i]} correct out of {row_sum} samples")
            
            # Overall attack detection rate (all non-benign classes)
            if self.label_names and 'BENIGN' in self.label_names:
                try:
                    benign_idx = self.label_names.index('BENIGN')
                    if benign_idx < len(best_confusion):
                        total_attacks = len(y_test) - (y_test == benign_idx).sum()
                        detected_attacks = ((best_predictions != benign_idx) & (y_test != benign_idx)).sum()
                        if total_attacks > 0:
                            attack_detection_rate = detected_attacks / total_attacks
                            print(f"\nOverall Attack Detection Rate: {attack_detection_rate:.4f} ({attack_detection_rate*100:.2f}%)")
                except (ValueError, IndexError):
                    pass

        print("="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)

        return {
            'comparison': comparison_results,
            'best_model': detailed_model_name,
            'best_metrics': best_metrics,
            'confusion_matrix': best_confusion,
            'predictions': best_predictions,
            'y_test': y_test,
            'test_records': int(len(y_test))
        }

def main():
    """Main testing pipeline"""
    import sys
    
    # Check command-line arguments
    models_dir = 'models'  # Default directory
    classification_mode = 'auto'  # Default to auto-detect
    
    if '--models-dir' in sys.argv:
        idx = sys.argv.index('--models-dir')
        if idx + 1 < len(sys.argv):
            models_dir = sys.argv[idx + 1]
    
    if '--binary' in sys.argv:
        classification_mode = 'binary'
    elif '--multiclass' in sys.argv:
        classification_mode = 'multiclass'
    
    print(f"Models Directory: {models_dir}")
    print(f"Classification Mode: {classification_mode.upper()}")
    
    # Initialize
    processor = DataProcessor()
    tester = ModelTester(models_dir=models_dir, classification_mode=classification_mode)
    
    # Load model
    tester.load_model()
    
    # Load test data
    print("\nLoading test data...")
    test_df = processor.load_test_data()
    print(f"Test data shape: {test_df.shape}")

    # Align processor with training features
    processor.feature_columns = tester.feature_names

    # Evaluate
    results = tester.evaluate_on_test_data(test_df, max_samples=10000)
    
    best_metrics = results['best_metrics']
    best_model_name = results['best_model']
    print(f"\n✓ Model evaluation completed!")
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1-Score: {best_metrics['f1_score']:.4f}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    os.makedirs('visualizations', exist_ok=True)
    
    # Confusion matrix
    if results['confusion_matrix'] is not None:
        # Get label names for multiclass visualization
        label_names_for_plot = None
        if tester.label_names:
            label_names_for_plot = tester.label_names
        elif tester.classification_mode == 'multiclass':
            # Try to get from model artifacts
            try:
                model_path = os.path.join(tester.models_dir, 'intrusion_detection_model.pkl')
                with open(model_path, 'rb') as f:
                    artifacts = joblib.load(f)
                    label_names_for_plot = artifacts.get('label_names', None)
            except:
                pass
        
        plot_confusion_matrix_heatmap(
            results['y_test'], 
            results['predictions'],
            model_name=best_model_name,
            save_path='visualizations/confusion_matrix.png',
            label_names=label_names_for_plot
        )
    
    # Model comparison
    if results['comparison']:
        plot_model_comparison(
            results['comparison'],
            save_path='visualizations/model_comparison.png'
        )
        plot_metrics_table(
            results['comparison'],
            save_path='visualizations/metrics_table.png'
        )
    
    print("\n✓ All visualizations saved to 'visualizations/' directory")


if __name__ == "__main__":
    main()

