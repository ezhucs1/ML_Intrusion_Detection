# Visualization Guide

This guide explains how to generate evaluation metric visualizations for your intrusion detection system.

## Overview

The visualization system uses **Matplotlib and Seaborn** (already installed) to create publication-quality plots for presentations and reports. No TensorFlow or additional libraries needed!

## Quick Start

### 1. Generate Standard Visualizations

Run the test script to automatically generate visualizations:

```bash
python src/test_model.py
```

This will create:
- `visualizations/confusion_matrix.png` - Confusion matrix heatmap
- `visualizations/model_comparison.png` - Bar chart comparing all 3 models
- `visualizations/metrics_table.png` - Formatted metrics table

### 2. Test Different Test Sample Sizes

To see how performance changes with different **test** sample sizes:

```bash
python src/test_sample_sizes.py
```

This will:
- Test the model with sample sizes: 5K, 10K, 15K, 20K, 25K, 30K
- Generate `visualizations/sample_size_comparison.png` - 4-panel plot showing how Accuracy, Precision, Recall, and F1-Score change with sample size
- Create `visualizations/sample_size_summary.csv` - Summary table of results

### 3. Compare Different Training Sample Sizes

To see how performance changes with different **training** sample sizes:

```bash
python src/train_and_compare_sizes.py
```

This will:
- **Train** models with different training sizes: 50K, 100K, 150K, 200K, 250K
- **Test** each trained model on the same test data (10K samples)
- Generate `visualizations/training_size_comparison.png` - 4-panel plot showing how test metrics change with training size
- Create `visualizations/training_size_summary.csv` - Summary table of results

⚠️ **Note**: This takes significant time (10-30 minutes) as it trains multiple models!

## Generated Visualizations

### Confusion Matrix
- Shows True Positives, False Positives, True Negatives, False Negatives
- Includes percentages for each cell
- Perfect for understanding model performance breakdown

### Model Comparison Chart
- Bar chart comparing RandomForest, GradientBoosting, and NeuralNetwork
- Shows Accuracy, Precision, Recall, and F1-Score side-by-side
- Great for presentations to show which model performs best

### Metrics Table
- Clean, formatted table of all metrics
- Easy to include in reports or presentations

### Test Sample Size Comparison
- 4-panel plot showing how each metric changes with **test** sample size
- Helps demonstrate statistical reliability
- Shows if increasing test sample size improves metric stability

### Training Size Comparison
- 4-panel plot showing how **test** metrics change with **training** sample size
- Shows if more training data improves model performance
- Helps identify optimal training size
- All models tested on the same test data for fair comparison

## Customization

### Change Sample Sizes

Edit `src/test_sample_sizes.py`:

```python
sample_sizes = [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000]
```

### Change Output Directory

Modify the `output_dir` parameter in the visualization functions:

```python
plot_model_comparison(
    results['comparison'],
    save_path='my_custom_folder/model_comparison.png'
)
```

### Adjust Plot Styles

Edit `src/visualize_metrics.py` to customize:
- Colors
- Figure sizes
- Font sizes
- Plot styles

## Using in Presentations

All plots are saved as high-resolution PNG files (300 DPI) suitable for:
- PowerPoint presentations
- LaTeX papers
- Reports
- Posters

Simply include the images from the `visualizations/` directory in your presentation.

## Example Workflow

1. **Train your model (standard):**
   ```bash
   python src/train_model.py
   ```

2. **Test with standard sample size and generate plots:**
   ```bash
   python src/test_model.py
   ```

3. **Test with multiple test sample sizes:**
   ```bash
   python src/test_sample_sizes.py
   ```

4. **Compare different training sizes (optional, takes time):**
   ```bash
   python src/train_and_compare_sizes.py
   ```

5. **Use the generated images:**
   - Open `visualizations/` folder
   - Copy images to your presentation/report
   - All images are ready to use!

## Troubleshooting

### "ModuleNotFoundError: No module named 'visualize_metrics'"
- Make sure you're running from the project root directory
- Check that `src/visualize_metrics.py` exists

### Plots not generating
- Check that `visualizations/` directory is writable
- Ensure matplotlib and seaborn are installed: `pip install matplotlib seaborn`

### Empty or incorrect plots
- Make sure you've trained a model first: `python src/train_model.py`
- Verify test data exists in `Testing_data/` directory

## File Structure

```
CSE543_Group1/
├── src/
│   ├── visualize_metrics.py       # Visualization functions
│   ├── test_model.py              # Standard testing + visualization
│   ├── test_sample_sizes.py       # Test sample size comparison
│   └── train_and_compare_sizes.py # Training size comparison
└── visualizations/                # Generated plots (auto-created)
    ├── confusion_matrix.png
    ├── model_comparison.png
    ├── metrics_table.png
    ├── sample_size_comparison.png      # Test sample size comparison
    ├── sample_size_summary.csv
    ├── training_size_comparison.png    # Training size comparison
    └── training_size_summary.csv
```

## Understanding the Two Comparisons

### Test Sample Size Comparison
- **What it shows**: How metrics change when testing with more/fewer samples
- **X-axis**: Test sample size (5K, 10K, 15K, ...)
- **Y-axis**: Metrics (%)
- **Purpose**: Shows statistical reliability and stability of metrics

### Training Size Comparison  
- **What it shows**: How model performance improves with more training data
- **X-axis**: Training sample size (50K, 100K, 150K, ...)
- **Y-axis**: Test metrics (%) - all tested on same test data
- **Purpose**: Shows if more training data improves model quality
- **Time**: Takes longer (trains multiple models)

