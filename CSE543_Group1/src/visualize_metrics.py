"""
Visualization module for model evaluation metrics
Generates publication-quality plots for presentations and reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def plot_confusion_matrix_heatmap(y_true, y_pred, model_name="Model", save_path=None, label_names=None):
    """
    Plot confusion matrix as a heatmap - supports both binary and multiclass
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the figure (optional)
        label_names: List of label names (for multiclass) or None for auto-detect
    
    Returns:
        matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(cm)
    
    # Auto-detect label names if not provided
    if label_names is None:
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        if len(unique_labels) == 2:
            label_names = ['Benign', 'Attack']
        else:
            label_names = [f'Class_{i}' for i in unique_labels]
    
    # Ensure label_names matches the number of classes in confusion matrix
    if len(label_names) != n_classes:
        # Use indices if mismatch
        label_names = [f'Class_{i}' for i in range(n_classes)]
    
    # Calculate percentages (avoid division by zero)
    row_sums = cm.sum(axis=1)
    cm_percent = np.zeros_like(cm, dtype=float)
    for i in range(n_classes):
        if row_sums[i] > 0:
            cm_percent[i] = cm[i].astype('float') / row_sums[i] * 100
    
    # Adjust figure size based on number of classes
    if n_classes > 10:
        fig_size = (16, 14)
        font_size = 8
        annotation_size = 7
    elif n_classes > 5:
        fig_size = (12, 10)
        font_size = 9
        annotation_size = 8
    else:
        fig_size = (10, 8)
        font_size = 10
        annotation_size = 9
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create heatmap with counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names,
                yticklabels=label_names,
                cbar_kws={'label': 'Count'},
                ax=ax,
                annot_kws={'size': annotation_size},
                linewidths=0.5,
                linecolor='gray')
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=font_size)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=font_size)
    
    # Add percentage annotations (only if figure is not too crowded)
    if n_classes <= 10:
        for i in range(n_classes):
            for j in range(n_classes):
                if cm[i, j] > 0:  # Only show percentage if count > 0
                    # Calculate position for percentage text (above count)
                    text = ax.text(j+0.5, i+0.25, f'({cm_percent[i, j]:.1f}%)',
                                  ha="center", va="center", 
                                  color="red", fontweight='bold', 
                                  fontsize=max(6, annotation_size-2))
    
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    return fig

def plot_model_comparison(metrics_dict, save_path=None):
    """
    Plot bar chart comparing all models across multiple metrics
    
    Args:
        metrics_dict: Dictionary of {model_name: {metric: value}}
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib figure
    """
    models = list(metrics_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = []
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [metrics_dict[model].get(metric, 0) * 100 for model in models]
        bar = ax.bar(x + i*width, values, width, label=label, alpha=0.85, color=colors[i])
        bars.append(bar)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            ax.text(j + i*width, v + 1, f'{v:.1f}%', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison saved to: {save_path}")
    
    return fig

def plot_sample_size_comparison(sample_size_results, save_path=None):
    """
    Plot how metrics change with different sample sizes
    
    Args:
        sample_size_results: Dictionary of {sample_size: {model: {metric: value}}}
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib figure
    """
    sample_sizes = sorted(sample_size_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        for model_name in ['RandomForest', 'GradientBoosting', 'NeuralNetwork']:
            values = []
            for size in sample_sizes:
                if model_name in sample_size_results[size]:
                    values.append(sample_size_results[size][model_name].get(metric, 0) * 100)
                else:
                    values.append(None)
            
            # Filter out None values
            valid_sizes = [s for s, v in zip(sample_sizes, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            if valid_values:
                ax.plot(valid_sizes, valid_values, marker='o', linewidth=2, 
                       markersize=8, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Sample Size', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{label} (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{label} vs Sample Size', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Set y-axis range to focus on the data range (80-100%) for better visibility
        all_values = []
        for model_name in ['RandomForest', 'GradientBoosting', 'NeuralNetwork']:
            for size in sample_sizes:
                if model_name in sample_size_results[size]:
                    val = sample_size_results[size][model_name].get(metric, 0) * 100
                    if val is not None:
                        all_values.append(val)
        
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            # Add padding: 2% below min, 2% above max, but at least show 80-100 range
            y_min = max(0, min(75, min_val - 2))
            y_max = min(100, max_val + 2)
            # Ensure we show meaningful range (at least 80-100 if data is in that range)
            if min_val >= 75:
                y_min = 75
            if max_val <= 100:
                y_max = 100
            ax.set_ylim([y_min, y_max])
        else:
            ax.set_ylim([75, 100])  # Default focused range
    
    plt.suptitle('Model Performance vs Test Sample Size', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sample size comparison saved to: {save_path}")
    
    return fig

def plot_training_size_comparison(training_size_results, save_path=None):
    """
    Plot how metrics change with different training sample sizes
    
    Args:
        training_size_results: Dictionary of {training_size: {model: {metric: value}}}
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib figure
    """
    training_sizes = sorted(training_size_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        for model_name in ['RandomForest', 'GradientBoosting', 'NeuralNetwork']:
            values = []
            for size in training_sizes:
                if model_name in training_size_results[size]:
                    values.append(training_size_results[size][model_name].get(metric, 0) * 100)
                else:
                    values.append(None)
            
            # Filter out None values
            valid_sizes = [s for s, v in zip(training_sizes, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            if valid_values:
                ax.plot(valid_sizes, valid_values, marker='o', linewidth=2, 
                       markersize=8, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Training Sample Size', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Test {label} (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Test {label} vs Training Sample Size', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Set y-axis range to focus on the data range (80-100%) for better visibility
        all_values = []
        for model_name in ['RandomForest', 'GradientBoosting', 'NeuralNetwork']:
            for size in training_sizes:
                if model_name in training_size_results[size]:
                    val = training_size_results[size][model_name].get(metric, 0) * 100
                    if val is not None:
                        all_values.append(val)
        
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            # Add padding: 2% below min, 2% above max, but at least show 75-100 range
            y_min = max(0, min(75, min_val - 2))
            y_max = min(100, max_val + 2)
            # Ensure we show meaningful range (at least 75-100 if data is in that range)
            if min_val >= 75:
                y_min = 75
            if max_val <= 100:
                y_max = 100
            ax.set_ylim([y_min, y_max])
        else:
            ax.set_ylim([75, 100])  # Default focused range
    
    plt.suptitle('Model Performance vs Training Sample Size (Tested on Same Test Data)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training size comparison saved to: {save_path}")
    
    return fig

def plot_metrics_table(metrics_dict, save_path=None):
    """
    Create a styled metrics table
    
    Args:
        metrics_dict: Dictionary of {model_name: {metric: value}}
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib figure
    """
    # Convert to DataFrame
    df = pd.DataFrame(metrics_dict).T
    df = df * 100  # Convert to percentage
    df = df.round(2)
    df.columns = [col.capitalize().replace('_', ' ') for col in df.columns]
    
    # Format values as strings with exactly 2 decimal places
    # Convert DataFrame values to formatted strings
    formatted_values = []
    for idx, row in df.iterrows():
        formatted_row = []
        for val in row:
            try:
                if pd.isna(val) or val is None:
                    formatted_row.append('N/A')
                else:
                    formatted_row.append(f'{float(val):.2f}')
            except (ValueError, TypeError):
                formatted_row.append(str(val))
        formatted_values.append(formatted_row)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=formatted_values,
                     rowLabels=df.index.tolist(),
                     colLabels=df.columns.tolist(),
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(df) + 1):
        if i % 2 == 0:
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title('Model Evaluation Metrics (%)', fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics table saved to: {save_path}")
    
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name="Model", save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (for positive class)
        model_name: Name of the model
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {save_path}")
    
    return plt.gcf()  # Return the current figure instead of plt module

def generate_all_visualizations(test_results, output_dir='visualizations', model_name=None):
    """
    Generate all visualization plots from test results
    
    Args:
        test_results: Dictionary containing test results
        output_dir: Directory to save plots
        model_name: Name of the best model
    
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # Extract data
    comparison = test_results.get('comparison', {})
    best_model = test_results.get('best_model', 'Unknown')
    best_metrics = test_results.get('best_metrics', {})
    confusion_matrix_data = test_results.get('confusion_matrix', None)
    
    # 1. Model comparison bar chart
    if comparison:
        fig = plot_model_comparison(comparison, 
                                   save_path=os.path.join(output_dir, 'model_comparison.png'))
        plt.close(fig)
        saved_files.append('model_comparison.png')
    
    # 2. Metrics table
    if comparison:
        fig = plot_metrics_table(comparison,
                                save_path=os.path.join(output_dir, 'metrics_table.png'))
        plt.close(fig)
        saved_files.append('metrics_table.png')
    
    return saved_files

