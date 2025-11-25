"""
Test model with different sample sizes and generate comparison visualizations
This script tests the model with various sample sizes and creates plots showing
how performance changes with sample size.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from test_model import ModelTester
from data_processor import DataProcessor
from visualize_metrics import plot_sample_size_comparison

def test_multiple_sample_sizes(sample_sizes=[5000, 10000, 15000, 20000, 25000, 30000]):
    """
    Test model with different sample sizes and collect results
    
    Args:
        sample_sizes: List of sample sizes to test
    
    Returns:
        Dictionary of results for each sample size
    """
    print("="*70)
    print("TESTING MULTIPLE SAMPLE SIZES")
    print("="*70)
    print(f"Sample sizes to test: {sample_sizes}")
    print()
    
    # Initialize
    processor = DataProcessor()
    tester = ModelTester()
    tester.load_model()
    
    # Load test data
    print("Loading test data...")
    test_df = processor.load_test_data()
    print(f"Total test records available: {len(test_df):,}\n")
    
    # Align processor with training features
    processor.feature_columns = tester.feature_names
    
    # Store results for each sample size
    all_results = {}
    
    for sample_size in sample_sizes:
        if sample_size > len(test_df):
            print(f"⚠️  Skipping {sample_size} (exceeds available data: {len(test_df):,})")
            continue
        
        print(f"\n{'='*70}")
        print(f"Testing with sample size: {sample_size:,}")
        print(f"{'='*70}")
        
        # Evaluate with this sample size
        results = tester.evaluate_on_test_data(test_df, max_samples=sample_size)
        
        # Store results
        all_results[sample_size] = {
            'comparison': results['comparison'],
            'best_model': results['best_model'],
            'best_metrics': results['best_metrics']
        }
        
        print(f"\n✓ Completed testing with {sample_size:,} samples")
        print(f"  Best Model: {results['best_model']}")
        print(f"  Accuracy: {results['best_metrics']['accuracy']*100:.2f}%")
    
    return all_results

def create_sample_size_visualizations(all_results, output_dir='visualizations'):
    """
    Create visualizations comparing different sample sizes
    
    Args:
        all_results: Dictionary of results from test_multiple_sample_sizes
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING SAMPLE SIZE COMPARISON VISUALIZATIONS")
    print("="*70)
    
    # Prepare data for plotting
    sample_size_results = {}
    for size, results in all_results.items():
        sample_size_results[size] = results['comparison']
    
    # Create sample size comparison plot
    if sample_size_results:
        fig = plot_sample_size_comparison(
            sample_size_results,
            save_path=os.path.join(output_dir, 'sample_size_comparison.png')
        )
        plt.close(fig)
        print("✓ Sample size comparison plot saved")
    
    # Create summary table
    summary_data = []
    for size in sorted(all_results.keys()):
        result = all_results[size]
        summary_data.append({
            'Sample Size': f"{size:,}",
            'Best Model': result['best_model'],
            'Accuracy (%)': f"{result['best_metrics']['accuracy']*100:.2f}",
            'Precision (%)': f"{result['best_metrics']['precision']*100:.2f}",
            'Recall (%)': f"{result['best_metrics']['recall']*100:.2f}",
            'F1-Score (%)': f"{result['best_metrics']['f1_score']*100:.2f}"
        })
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'sample_size_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary table saved to: {summary_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Performance by Sample Size")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    return summary_df

def main():
    """Main function"""
    # Test with different sample sizes
    sample_sizes = [5000, 10000, 15000, 20000, 25000, 30000]
    
    print("\n" + "="*70)
    print("SAMPLE SIZE ANALYSIS")
    print("="*70)
    print("This script will test the model with different sample sizes")
    print("and generate comparison visualizations.")
    print()
    
    # Test multiple sample sizes
    all_results = test_multiple_sample_sizes(sample_sizes)
    
    # Generate visualizations
    summary_df = create_sample_size_visualizations(all_results)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print(f"All visualizations saved to: visualizations/")
    print(f"  - sample_size_comparison.png")
    print(f"  - sample_size_summary.csv")
    print()

if __name__ == "__main__":
    main()

