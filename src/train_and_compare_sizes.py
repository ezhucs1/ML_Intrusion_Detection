"""
Train models with different training sample sizes and compare results
This script trains models with various training sizes, tests each on the same test data,
and generates comparison visualizations.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
import shutil
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train_model import IntrusionDetectionTrainer
from test_model import ModelTester
from data_processor import DataProcessor
from visualize_metrics import plot_training_size_comparison

def train_with_size(trainer, train_df, training_size, models_dir_base='models'):
    """
    Train models with a specific training sample size
    
    Args:
        trainer: IntrusionDetectionTrainer instance
        train_df: Full training dataframe
        training_size: Number of samples to use for training
        models_dir_base: Base directory for models
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*70}")
    print(f"TRAINING WITH {training_size:,} SAMPLES")
    print(f"{'='*70}")
    
    # Create temporary models directory for this training size
    temp_models_dir = f"{models_dir_base}_size_{training_size}"
    os.makedirs(temp_models_dir, exist_ok=True)
    
    # Set trainer to use temporary directory
    trainer.models_dir = temp_models_dir
    
    # Train model
    trainer.train_and_evaluate(train_df, n_samples=training_size)
    
    # Load results from saved model info
    model_info_path = os.path.join(temp_models_dir, 'model_info.json')
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # Extract test metrics (these are from the validation split during training)
    results = {}
    if 'all_models_metrics' in model_info:
        for model_name, metrics in model_info['all_models_metrics'].items():
            results[model_name] = {
                'accuracy': metrics.get('test_accuracy', 0),
                'precision': metrics.get('test_precision', 0),
                'recall': metrics.get('test_recall', 0),
                'f1_score': metrics.get('test_f1_score', 0)
            }
    
    return results, temp_models_dir

def test_trained_model(test_df, models_dir, test_sample_size=10000):
    """
    Test a trained model on test data by directly loading from the models directory
    
    Args:
        test_df: Test dataframe
        models_dir: Directory containing trained models
        test_sample_size: Number of test samples to use
    
    Returns:
        Dictionary with test metrics for all models
    """
    try:
        # Load model directly from the specified directory
        import joblib
        
        # Load model candidates
        candidates_path = os.path.join(models_dir, 'model_candidates.pkl')
        if not os.path.exists(candidates_path):
            print(f"Warning: {candidates_path} not found")
            return {}
        
        with open(candidates_path, 'rb') as f:
            model_data = joblib.load(f)
        
        models = model_data['models']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        # Preprocess test data
        processor = DataProcessor()
        processor.feature_columns = feature_names
        
        # Sample test data
        if test_sample_size and len(test_df) > test_sample_size:
            test_df_sample = test_df.sample(n=test_sample_size, random_state=42)
        else:
            test_df_sample = test_df
        
        X_test, y_test = processor.preprocess_data(test_df_sample)
        
        # Ensure feature alignment
        missing_features = set(feature_names) - set(X_test.columns)
        for feat in missing_features:
            X_test[feat] = 0
        
        extra_features = set(X_test.columns) - set(feature_names)
        if extra_features:
            X_test = X_test.drop(columns=list(extra_features))
        
        X_test = X_test[feature_names]
        
        # Ensure numeric
        for col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        X_test = X_test.fillna(0)
        X_test = X_test.select_dtypes(include=[np.number])
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Evaluate all models
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results = {}
        for model_name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            
            results[model_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        
        return results
    
    except Exception as e:
        print(f"Error testing model from {models_dir}: {e}")
        import traceback
        traceback.print_exc()
        return {}

def train_and_compare_sizes(training_sizes=[50000, 100000, 150000, 200000, 250000], 
                            test_sample_size=10000):
    """
    Train models with different training sizes and compare results
    
    Args:
        training_sizes: List of training sample sizes to test
        test_sample_size: Number of test samples to use for evaluation
    
    Returns:
        Dictionary of results for each training size
    """
    print("="*70)
    print("TRAINING SIZE COMPARISON ANALYSIS")
    print("="*70)
    print(f"Training sizes to test: {training_sizes}")
    print(f"Test sample size: {test_sample_size:,}")
    print()
    
    # Initialize
    processor = DataProcessor()
    trainer = IntrusionDetectionTrainer()
    
    # Load training data
    print("Loading training data...")
    train_df = processor.load_training_data()
    print(f"Total training records available: {len(train_df):,}\n")
    
    # Load test data (same for all comparisons)
    print("Loading test data...")
    test_df = processor.load_test_data()
    print(f"Total test records available: {len(test_df):,}\n")
    
    # Store results
    all_results = {}
    model_directories = {}
    
    for training_size in training_sizes:
        if training_size > len(train_df):
            print(f"⚠️  Skipping {training_size} (exceeds available data: {len(train_df):,})")
            continue
        
        # Train with this size
        train_results, temp_models_dir = train_with_size(trainer, train_df, training_size)
        model_directories[training_size] = temp_models_dir
        
        # Test on independent test data
        print(f"\nTesting model trained with {training_size:,} samples on test data...")
        test_results = test_trained_model(test_df, temp_models_dir, test_sample_size)
        
        # Store combined results (prioritize test results, fallback to train validation results)
        combined_results = {}
        for model_name in ['RandomForest', 'GradientBoosting', 'NeuralNetwork']:
            if model_name in test_results:
                # Use test results (more reliable)
                combined_results[model_name] = test_results[model_name]
            elif model_name in train_results:
                # Fallback to training validation results
                combined_results[model_name] = train_results[model_name]
        
        all_results[training_size] = combined_results
        
        print(f"\n✓ Completed training and testing with {training_size:,} samples")
        if combined_results:
            best_model = max(combined_results.keys(), 
                           key=lambda k: combined_results[k].get('f1_score', 0))
            best_acc = combined_results[best_model].get('accuracy', 0) * 100
            print(f"  Best Model: {best_model}")
            print(f"  Best Test Accuracy: {best_acc:.2f}%")
    
    return all_results, model_directories

def create_training_size_visualizations(all_results, output_dir='visualizations'):
    """
    Create visualizations comparing different training sizes
    
    Args:
        all_results: Dictionary of results from train_and_compare_sizes
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING TRAINING SIZE COMPARISON VISUALIZATIONS")
    print("="*70)
    
    # Prepare data for plotting
    training_size_results = {}
    for size, results in all_results.items():
        training_size_results[size] = results
    
    # Create training size comparison plot
    if training_size_results:
        fig = plot_training_size_comparison(
            training_size_results,
            save_path=os.path.join(output_dir, 'training_size_comparison.png')
        )
        plt.close(fig)
        print("✓ Training size comparison plot saved")
    
    # Create summary table
    summary_data = []
    for size in sorted(all_results.keys()):
        result = all_results[size]
        if result:
            # Find best model for this training size
            best_model = max(result.keys(), 
                           key=lambda k: result[k].get('f1_score', 0))
            best_metrics = result[best_model]
            
            summary_data.append({
                'Training Size': f"{size:,}",
                'Best Model': best_model,
                'Test Accuracy (%)': f"{best_metrics.get('accuracy', 0)*100:.2f}",
                'Test Precision (%)': f"{best_metrics.get('precision', 0)*100:.2f}",
                'Test Recall (%)': f"{best_metrics.get('recall', 0)*100:.2f}",
                'Test F1-Score (%)': f"{best_metrics.get('f1_score', 0)*100:.2f}"
            })
    
    # Save summary to CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'training_size_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Summary table saved to: {summary_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY: Performance by Training Size")
        print("="*70)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    return None

def cleanup_temp_models(model_directories, keep_best=True):
    """
    Clean up temporary model directories
    
    Args:
        model_directories: Dictionary of {training_size: models_dir}
        keep_best: If True, keep the best performing model
    """
    print("\n" + "="*70)
    print("CLEANUP")
    print("="*70)
    
    if keep_best:
        # Find best training size (you might want to customize this logic)
        print("Keeping all trained models for reference.")
        print("To remove temporary model directories, run:")
        for size, dir_path in model_directories.items():
            print(f"  rm -rf {dir_path}")
    else:
        # Remove all temporary directories
        for size, dir_path in model_directories.items():
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"✓ Removed {dir_path}")

def main():
    """Main function"""
    training_sizes = [50000, 100000, 150000, 200000, 250000]
    test_sample_size = 10000
    
    print("\n" + "="*70)
    print("TRAINING SIZE ANALYSIS")
    print("="*70)
    print("This script will:")
    print("  1. Train models with different training sample sizes")
    print("  2. Test each trained model on the same test data")
    print("  3. Generate comparison visualizations")
    print()
    print(f"Training sizes: {training_sizes}")
    print(f"Test sample size: {test_sample_size:,}")
    print()
    print("⚠️  WARNING: This will take significant time as it trains multiple models!")
    print("    Estimated time: ~10-30 minutes depending on your system")
    print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Train and compare
    all_results, model_directories = train_and_compare_sizes(
        training_sizes=training_sizes,
        test_sample_size=test_sample_size
    )
    
    # Generate visualizations
    summary_df = create_training_size_visualizations(all_results)
    
    # Cleanup (keep models for now)
    cleanup_temp_models(model_directories, keep_best=True)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print(f"All visualizations saved to: visualizations/")
    print(f"  - training_size_comparison.png")
    if summary_df is not None:
        print(f"  - training_size_summary.csv")
    print()
    print("Trained models are saved in:")
    for size, dir_path in model_directories.items():
        print(f"  - {dir_path}")
    print()

if __name__ == "__main__":
    main()

