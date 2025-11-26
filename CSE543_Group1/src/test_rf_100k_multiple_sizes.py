"""
Test RandomForest model (trained with 100K samples) on 2018 data
with different test sample sizes and generate comprehensive visualizations
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from src directory (shared with main binary evaluation script)
try:
    from data_processor import DataProcessor
    from visualize_metrics import (plot_confusion_matrix_heatmap, plot_roc_curve,
                                  plot_metrics_table, plot_sample_size_comparison)
except ImportError:
    # Try with src prefix
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from data_processor import DataProcessor
    from visualize_metrics import (plot_confusion_matrix_heatmap, plot_roc_curve,
                                  plot_metrics_table, plot_sample_size_comparison)

def load_rf_model_from_100k(models_dir='models'):
    """
    Load RandomForest model trained with 100K samples
    
    Args:
        models_dir: Directory containing the trained model
    
    Returns:
        Tuple of (model, scaler, feature_names)
    """
    print("="*70)
    print("LOADING RANDOMFOREST MODEL (Main binary model from 'models/')")
    print("="*70)
    
    candidates_path = os.path.join(models_dir, 'model_candidates.pkl')
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(f"Model not found: {candidates_path}")
    
    with open(candidates_path, 'rb') as f:
        model_data = joblib.load(f)
    
    models = model_data['models']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    if 'RandomForest' not in models:
        raise ValueError("RandomForest model not found in saved models")
    
    rf_model = models['RandomForest']
    
    print(f"✓ Model loaded from: {models_dir}")
    print(f"✓ Features: {len(feature_names)}")
    print()
    
    return rf_model, scaler, feature_names

def test_rf_with_size(rf_model, scaler, feature_names, test_df, test_sample_size):
    """
    Test RandomForest model with a specific test sample size
    
    Args:
        rf_model: RandomForest model
        scaler: Feature scaler
        feature_names: List of feature names
        test_df: Full test dataframe
        test_sample_size: Number of samples to test
    
    Returns:
        Dictionary with metrics, predictions, and true labels
    """
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
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]  # Probability of attack class
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return {
        'metrics': metrics,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'test_size': len(y_test)
    }

def test_multiple_sizes(rf_model, scaler, feature_names, test_df, 
                        test_sizes=[5000, 10000, 15000, 20000, 25000, 30000]):
    """
    Test RandomForest model with multiple test sample sizes
    
    Args:
        rf_model: RandomForest model
        scaler: Feature scaler
        feature_names: List of feature names
        test_df: Full test dataframe
        test_sizes: List of test sample sizes
    
    Returns:
        Dictionary of results for each test size
    """
    print("="*70)
    print("TESTING WITH MULTIPLE SAMPLE SIZES")
    print("="*70)
    print(f"Test sizes: {test_sizes}")
    print()
    
    all_results = {}
    
    for test_size in test_sizes:
        if test_size > len(test_df):
            print(f"⚠️  Skipping {test_size} (exceeds available data: {len(test_df):,})")
            continue
        
        print(f"Testing with {test_size:,} samples...")
        result = test_rf_with_size(rf_model, scaler, feature_names, test_df, test_size)
        all_results[test_size] = result
        
        metrics = result['metrics']
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
        print()
    
    return all_results

def create_comprehensive_visualizations(all_results, output_dir='visualizations'):
    """
    Create all visualizations for RandomForest model testing
    
    Args:
        all_results: Dictionary of results from test_multiple_sizes
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*70)
    
    # 1. Sample size comparison plot
    sample_size_results = {}
    for size, result in all_results.items():
        sample_size_results[size] = {
            'RandomForest': result['metrics']
        }
    
    if sample_size_results:
        fig = plot_sample_size_comparison(
            sample_size_results,
            save_path=os.path.join(output_dir, 'rf_100k_sample_size_comparison.png')
        )
        plt.close(fig)
        print("✓ Sample size comparison plot saved")
    
    # 2. Confusion matrix for largest test size (most reliable)
    largest_size = max(all_results.keys())
    largest_result = all_results[largest_size]
    
    fig = plot_confusion_matrix_heatmap(
        largest_result['y_test'],
        largest_result['y_pred'],
        model_name=f'RandomForest (100K trained, {largest_size:,} tested)',
        save_path=os.path.join(output_dir, 'rf_100k_confusion_matrix.png')
    )
    plt.close(fig)
    print("✓ Confusion matrix saved")
    
    # 3. ROC curve for largest test size
    fig = plot_roc_curve(
        largest_result['y_test'],
        largest_result['y_pred_proba'],
        model_name=f'RandomForest (100K trained, {largest_size:,} tested)',
        save_path=os.path.join(output_dir, 'rf_100k_roc_curve.png')
    )
    plt.close(fig)
    print("✓ ROC curve saved")
    
    # 4. Metrics table for all test sizes
    metrics_data = {}
    for size, result in all_results.items():
        metrics_data[f'{size:,} samples'] = result['metrics']
    
    if metrics_data:
        fig = plot_metrics_table(
            metrics_data,
            save_path=os.path.join(output_dir, 'rf_100k_metrics_table.png')
        )
        plt.close(fig)
        print("✓ Metrics table saved")
    
    # 5. Summary CSV
    summary_data = []
    for size in sorted(all_results.keys()):
        result = all_results[size]
        metrics = result['metrics']
        cm = metrics['confusion_matrix']
        summary_data.append({
            'Test Sample Size': f"{size:,}",
            'Accuracy (%)': f"{metrics['accuracy']*100:.2f}",
            'Precision (%)': f"{metrics['precision']*100:.2f}",
            'Recall (%)': f"{metrics['recall']*100:.2f}",
            'F1-Score (%)': f"{metrics['f1_score']*100:.2f}",
            'True Negatives': int(cm[0, 0]),
            'False Positives': int(cm[0, 1]),
            'False Negatives': int(cm[1, 0]),
            'True Positives': int(cm[1, 1])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'rf_100k_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary CSV saved to: {summary_path}")
    
    # Print detailed summary
    print("\n" + "="*70)
    print("DETAILED SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    # Print classification report for largest test size
    print("\n" + "="*70)
    print(f"CLASSIFICATION REPORT (Test Size: {largest_size:,})")
    print("="*70)
    y_true_largest = largest_result['y_test']
    y_pred_largest = largest_result['y_pred']
    print(classification_report(
        y_true_largest,
        y_pred_largest,
        target_names=['Benign', 'Attack'],
        zero_division=0
    ))

    # Attack Detection Rate (binary: TP / (TP + FN)) for largest test size
    cm_largest = confusion_matrix(y_true_largest, y_pred_largest)
    if cm_largest.shape == (2, 2):
        tn, fp, fn, tp = cm_largest.ravel()
        if tp + fn > 0:
            attack_detection_rate = tp / (tp + fn)
            print(f"\nAttack Detection Rate: {attack_detection_rate:.4f} ({attack_detection_rate*100:.2f}%)")
    
    return summary_df

def main():
    """Main function"""
    # Use the same directory as src/test_model.py so results are consistent
    models_dir = 'models'
    test_sizes = [5000, 10000, 15000, 20000, 25000, 30000]
    
    print("\n" + "="*70)
    print("RANDOMFOREST MODEL TESTING (100K Training Samples)")
    print("="*70)
    print("This script will:")
    print("  1. Load RandomForest model trained with 100K samples")
    print("  2. Test on 2018 data with different test sample sizes")
    print("  3. Generate comprehensive visualizations")
    print()
    
    # Load model
    rf_model, scaler, feature_names = load_rf_model_from_100k(models_dir)
    
    # Load test data (2018 data)
    print("Loading 2018 test data...")
    processor = DataProcessor()
    test_df = processor.load_test_data()
    print(f"Total test records available: {len(test_df):,}\n")
    
    # Test with multiple sizes
    all_results = test_multiple_sizes(
        rf_model, scaler, feature_names, test_df, test_sizes
    )
    
    # Generate visualizations
    summary_df = create_comprehensive_visualizations(all_results)
    
    print("\n" + "="*70)
    print("✓ TESTING COMPLETE")
    print("="*70)
    print(f"All visualizations saved to: visualizations/")
    print("  - rf_100k_sample_size_comparison.png")
    print("  - rf_100k_confusion_matrix.png")
    print("  - rf_100k_roc_curve.png")
    print("  - rf_100k_metrics_table.png")
    print("  - rf_100k_summary.csv")
    print()

if __name__ == "__main__":
    main()

