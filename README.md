# 🛡️ Network Intrusion Detection System

A machine learning-based intrusion detection system trained on the **CIC-IDS-2017** dataset to identify network attacks in real-time.

## 🎯 Overview

This system uses machine learning to analyze network traffic and detect intrusions with **87.49% accuracy** on held-out test data. It trains three different algorithms (Random Forest, Gradient Boosting, Neural Network) and automatically selects the best performer.

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **RandomForest** (Best) | 87.49% | 0.8970 | 0.8749 | 0.8747 |
| GradientBoosting | 82.89% | 0.8324 | 0.8289 | 0.8293 |
| NeuralNetwork | 78.52% | 0.8122 | 0.7852 | 0.7841 |

**Test Results:**
- Attack Detection Rate: 78.34%
- True Positives: 4,298 out of 5,486 attacks
- Low False Positives: 63 out of 4,514 benign traffic

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- 8GB+ RAM (for training)
- 10GB+ disk space (for dataset)

### 2. Installation

```bash
# Clone repository
git clone https://github.com/ezhucs1/ML_Intrusion_Detection.git
cd ML_Intrusion_Detection

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Dataset

Download **CIC-IDS-2017** dataset from: https://www.unb.ca/cic/datasets/ids-2017.html

Place training CSV files in `data_original/`:
```
data_original/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Place test CSV files in `Testing_data/`:
```
Testing_data/
├── 02-14-2018.csv
├── 02-15-2018.csv
├── 02-16-2018.csv
├── 02-20-2018.csv
├── 02-21-2018.csv
├── 02-22-2018.csv
├── 02-23-2018.csv
├── 02-28-2018.csv
├── 03-01-2018.csv
└── 03-02-2018.csv
```

### 4. Train Model

```bash
python src/train_model.py
```

This will:
- Load training data from `data_original/`
- Preprocess and clean the data
- Train 3 models (Random Forest, Gradient Boosting, Neural Network)
- Select the best performing model
- Save model artifacts to `models/`

**Expected time:** 10-30 minutes (depending on hardware)

### 5. Test Model

```bash
python src/test_model.py
```

This will:
- Load the trained model
- Evaluate on test data from `Testing_data/`
- Display comparison metrics for all 3 models
- Show detailed results for the best model

### 6. Run Full Pipeline

```bash
# Using the convenience script
./run_full_pipeline.sh

# Or manually
python src/train_model.py && python src/test_model.py
```

### 7. Launch Web Demo 🎨

```bash
# Using the convenience script
./run_web_demo.sh

# Or manually
streamlit run src/web_demo.py
```

This will start an interactive web interface at `http://localhost:8501` where you can:
- 🎲 Test with sample network flow data
- 📝 Manually input network flow features
- 📁 Upload CSV files for analysis
- 📊 View real-time predictions with confidence scores
- 📈 Compare model performance metrics

**Perfect for presentations and demonstrations!**

## 📁 Project Structure

```
CSE543_Group1/
├── src/
│   ├── data_processor.py   # Data loading and preprocessing
│   ├── train_model.py      # Model training script
│   ├── test_model.py       # Model evaluation script
│   └── web_demo.py         # Interactive web demo (Streamlit)
├── models/                 # Trained models (gitignored)
│   ├── intrusion_detection_model.pkl
│   ├── model_candidates.pkl
│   ├── model_info.json
│   └── feature_names.json
├── data_original/          # Training data (gitignored)
├── Testing_data/           # Test data (gitignored)
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
├── run_full_pipeline.sh   # Full pipeline script
├── run_web_demo.sh        # Web demo launcher
├── README.md              # This file
└── PRESENTATION_OUTLINE.md # Presentation guide (optional)
```

## 🔧 How It Works

### 1. Data Preprocessing
- Loads network flow data from CSV files
- Standardizes column names across different dataset formats
- Handles missing values and infinities
- Removes constant columns
- Extracts 70 network flow features

### 2. Feature Engineering
Extracts 70 features including:
- **Flow characteristics**: Duration, packet counts, bytes
- **Packet statistics**: Mean, max, min, standard deviation
- **Inter-arrival times**: Forward and backward packet timing
- **Protocol flags**: SYN, FIN, ACK counts
- **Window sizes**: TCP window scaling

### 3. Model Training
- Trains 3 algorithms: Random Forest, Gradient Boosting, Neural Network
- Uses stratified sampling to maintain class balance
- Performs train-test split (80/20)
- Automatically selects best model based on F1-score

### 4. Evaluation
- Tests on held-out test data
- Compares all three models
- Calculates accuracy, precision, recall, F1-score
- Generates confusion matrix and classification report

## 💡 Usage Example

```python
import joblib
import numpy as np

# Load trained model
artifacts = joblib.load('models/intrusion_detection_model.pkl')
model = artifacts['model']
scaler = artifacts['scaler']
feature_names = artifacts.get('feature_names', [])

# Predict on new network flow (70 features)
features = [80, 9392887, 4, 0, 24, 0, ...]  # Your network flow features
scaled_features = scaler.transform([features])
prediction = model.predict(scaled_features)

print("🚨 ATTACK DETECTED!" if prediction[0] == 1 else "✅ Normal traffic")
```

## 📊 Dataset Information

**CIC-IDS-2017 Dataset:**
- Training: 2,830,743 network flow records
- Test: Separate held-out data (CSE-CIC-IDS2018)
- Attack types: DDoS, Brute Force, Web Attacks, Infiltration, Port Scan
- Features: 70 network flow characteristics
- Source: Canadian Institute for Cybersecurity

## 🎤 Presentation Guide

For detailed presentation instructions, see `PRESENTATION_OUTLINE.md`.

**Quick Talking Points:**
1. "We trained on the industry-standard CIC-IDS-2017 dataset"
2. "87.49% accuracy in detecting network intrusions on test data"
3. "Three algorithms tested, Random Forest performed best"
4. "Processes 70 network flow features in real-time"
5. "Successfully distinguishes attacks from normal traffic"

## 🛠️ Requirements

See `requirements.txt` for full list. Main dependencies:
- pandas
- numpy
- scikit-learn
- joblib

## ⚠️ Troubleshooting

**Issue**: `FileNotFoundError` for CSV files  
**Solution**: Ensure dataset is placed in `data_original/` and `Testing_data/` directories

**Issue**: Memory error during training  
**Solution**: Edit `src/train_model.py` and reduce `n_samples` parameter (default: 100,000)

**Issue**: Column mismatch errors  
**Solution**: The code automatically handles different column name formats between training and test datasets

**Issue**: Missing modules  
**Solution**: `pip install -r requirements.txt`

## 📖 Technical Details

### Algorithms Used
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble learning
- **Neural Network**: Multi-layer perceptron (MLP)

### Data Preprocessing
- Feature scaling using StandardScaler
- Missing value imputation with median
- Handling of infinite values
- Constant column removal
- Column name standardization across datasets

### Model Selection
Best model selected based on F1-score to balance precision and recall.

## 📚 References

- [CIC-IDS-2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## 📄 License

Academic/Research use - CSE543 Group Project

---

**Built with ❤️ for Network Security**
