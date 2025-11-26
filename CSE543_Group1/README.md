# 🛡️ Binary Network Intrusion Detection (CIC-IDS-2017 → CIC-IDS-2018)

This project trains a **binary intrusion detection model** on a **100,000‑sample subset of CIC‑IDS‑2017** and evaluates it on **30,000 records from CIC‑IDS‑2018** using a **RandomForest** classifier.

The main goal:  
- **Train**: CIC‑IDS‑2017 → 100,000 sampled flows (BENIGN vs ATTACK)  
- **Test**: CIC‑IDS‑2018 → up to 30,000 flows  
- **Result at 30,000 test records** (from `src/test_rf_100k_multiple_sizes.py`):
  - Accuracy: **87.65%**
  - Precision: **89.76%**
  - Recall: **87.65%**
  - F1‑Score: **87.60%**
  - Attack Detection Rate: **78.19%**

---

## 📂 Project Structure (Python Files and Purposes)

All core code lives in `CSE543_Group1/src/`:

- `data_processor.py`  
  - Loads CSVs from `data_original/` (CIC‑IDS‑2017) and `Testing_data/` (CIC‑IDS‑2018)  
  - Standardizes column names across days / datasets  
  - Cleans data (NaNs, infinities, header rows inside files)  
  - Builds feature matrix (68 features actually used by the final model)  
  - Encodes labels for **binary** classification: `BENIGN` → 0, Attack (everything else) → 1  

- `train_model.py`  
  - Trains 3 models on a **stratified 100,000‑sample subset** of CIC‑IDS‑2017:
    - `RandomForestClassifier`
    - `GradientBoostingClassifier`
    - `MLPClassifier` (Neural Network)  
  - Selects the **best** model by F1‑score (RandomForest in practice)  
  - Saves:
    - `models/intrusion_detection_model.pkl` (best model + scaler + metadata)
    - `models/model_candidates.pkl` (all trained models + scaler + feature names)
    - `models/feature_names.json`, `models/model_info.json` (local only; not in Git)

- `test_model.py`  
  - Loads the best model from `models/` (auto‑detects binary mode)  
  - Samples **10,000 records** from CIC‑IDS‑2018 (`Testing_data/`)  
  - Evaluates all three models and prints a comparison table  
  - Prints detailed metrics and confusion matrix for the best model  
  - Generates:
    - `visualizations/confusion_matrix.png`
    - `visualizations/model_comparison.png`
    - `visualizations/metrics_table.png`

- `test_rf_100k_multiple_sizes.py`  
  - Loads the **same binary RandomForest model from `models/`**  
  - Loads up to ~49,999 test records from CIC‑IDS‑2018  
  - Evaluates the model on multiple test sizes:
    - 5,000 / 10,000 / 15,000 / 20,000 / 25,000 / **30,000**  
  - Prints metrics per size and a detailed summary table  
  - For the largest test size (30,000), prints:
    - Classification report (Benign vs Attack)
    - **Attack Detection Rate** = TP / (TP + FN)  
  - Saves:
    - `visualizations/rf_100k_sample_size_comparison.png`
    - `visualizations/rf_100k_confusion_matrix.png`
    - `visualizations/rf_100k_roc_curve.png`
    - `visualizations/rf_100k_metrics_table.png`
    - `visualizations/rf_100k_summary.csv`

- `test_sample_sizes.py`  
  - Helper script to explore how test sample size affects metrics using a more generic pipeline.  
  - Not required for the main 100k → 30k experiment, but useful for additional analysis.

- `train_and_compare_sizes.py`  
  - Trains models with **different training sizes** (e.g., 50k, 100k, 150k, …) and compares performance.  
  - Good for answering: “How much training data do we really need?”  
  - Not required for the main 100k training experiment.

- `visualize_metrics.py`  
  - Shared plotting utilities:
    - Confusion matrix heatmap (binary or multiclass)
    - ROC curve
    - Metrics tables
    - Training/test size comparison plots

---

## 📊 Data Layout

Place the raw CSVs under `CSE543_Group1/`:

```text
CSE543_Group1/
├── data_original/      # CIC-IDS-2017 (training data)
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── ...
│   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
└── Testing_data/       # CIC-IDS-2018 (test data)
    ├── 02-14-2018.csv
    ├── 02-15-2018.csv
    ├── ...
    └── 03-02-2018.csv
```

These folders and the large model files are **not** tracked by Git (they are in `.gitignore`).

---

## 🌐 Dataset Download Links

- **CIC-IDS-2017**: [Official dataset page](https://www.unb.ca/cic/datasets/ids-2017.html)  
- **CIC-IDS-2018**: [Official dataset page](https://www.unb.ca/cic/datasets/ids-2018.html)

---

## ⚙️ Environment Setup

From `CSE543_Group1/`:

```bash
cd <YOUR_PROJECT_PATH>/CSE543_Group1

# (Optional) create venv
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Step 1 – Train Binary Model on CIC‑IDS‑2017 (100,000 samples)

Train the binary model using a 100,000‑sample stratified subset of CIC‑IDS‑2017:

```bash
cd <YOUR_PROJECT_PATH>/CSE543_Group1
source .venv/bin/activate   # if using virtualenv

python src/train_model.py --binary --models-dir models --n-samples 100000
```

What this does:

- Loads all `data_original/*.csv`
- Creates a stratified 100,000‑sample training set
- Trains RandomForest, GradientBoosting, and Neural Network
- Selects the best by F1‑Score (RandomForest)
- Saves best model and candidates to `models/` (local only)

---

## 🧪 Step 2 – Quick Evaluation on 10,000 CIC‑IDS‑2018 Records

Run the standard evaluation on a **10,000‑record** subset of CIC‑IDS‑2018:

```bash
cd <YOUR_PROJECT_PATH>/CSE543_Group1
python src/test_model.py
```

Expected behavior:

- Samples 10,000 test records from `Testing_data/`
- Evaluates all 3 models; output (approx):

```text
Model               Accuracy   Precision    Recall    F1-Score
----------------------------------------------------------------------
RandomForest          0.8757      0.8977    0.8757      0.8754
GradientBoosting      0.8309      0.8347    0.8309      0.8313
NeuralNetwork         0.7843      0.8108    0.7843      0.7832
```

Detailed report for RandomForest (best model):

- **Accuracy** ≈ **87.57%**
- **Attack Detection Rate** (on this 10k subset) ≈ **78.37%**

Visual outputs (in `visualizations/`):

- `confusion_matrix.png`
- `model_comparison.png`
- `metrics_table.png`

---

## 🧪 Step 3 – Evaluation on CIC‑IDS‑2018 up to 30,000 Records (Main Result)

Now evaluate the same **RandomForest** model from `models/` across multiple test sizes, including **30,000**:

```bash
cd <YOUR_PROJECT_PATH>/CSE543_Group1
python src/test_rf_100k_multiple_sizes.py
```

This script:

- Loads the best RandomForest model from `models/`
- Loads ~49,999 CIC‑IDS‑2018 records from `Testing_data/`
- Evaluates at test sizes: 5k / 10k / 15k / 20k / 25k / **30k**
- Prints metrics for each size and saves plots + a summary CSV

### Key Metrics by Test Size (from a reference run)

```text
Test Sample Size Accuracy (%) Precision (%) Recall (%) F1-Score (%)  True Negatives  False Positives  False Negatives  True Positives
5,000               87.82          89.90        87.82        87.78            2272               30              579            2119
10,000              87.57          89.77        87.57        87.54            4468               59             1184            4289
15,000              87.70          89.83        87.70        87.66            6778               89             1756            6377
20,000              87.81          89.91        87.81        87.77            9082              116             2322            8480
25,000              87.76          89.87        87.76        87.72           11365              147             2912           10576
30,000              87.65          89.76        87.65        87.60           13686              186             3518           12610
```

### Highlight: CIC‑IDS‑2018 – 30,000 Record Evaluation

From the classification report at **30,000**:

- **Accuracy**: **87.65%**
- **Precision (weighted)**: 89.76%
- **Recall (weighted)**: 87.65%
- **F1‑Score (weighted)**: 87.60%

Per‑class (30,000 records):

- **Benign**
  - Precision: 0.80
  - Recall: 0.99
  - F1: 0.88  
  - Support: 13,872
- **Attack**
  - Precision: 0.99
  - Recall: 0.78
  - F1: 0.87  
  - Support: 16,128

**Attack Detection Rate** (TP / (TP + FN)) at 30,000 records:

- Confusion matrix row for Attack: FN = 3,518, TP = 12,610
- \( \text{ADR} = 12610 / (12610 + 3518) \approx 0.7819 \)  
- **Attack Detection Rate: 78.19%**

Visual outputs (in `visualizations/`):

- `rf_100k_sample_size_comparison.png`
- `rf_100k_confusion_matrix.png`
- `rf_100k_roc_curve.png`
- `rf_100k_metrics_table.png`
- `rf_100k_summary.csv`

---

## 🔁 Quick Reproduction Checklist

1. Put CIC‑IDS‑2017 CSVs in `CSE543_Group1/data_original/`.
2. Put CIC‑IDS‑2018 CSVs in `CSE543_Group1/Testing_data/`.
3. Create & activate a Python environment, then:

```bash
cd <YOUR_PROJECT_PATH>/CSE543_Group1
pip install -r requirements.txt

# Train on 100,000 CIC-IDS-2017 samples (binary)
python src/train_model.py --binary --models-dir models --n-samples 100000

# Quick test on 10,000 CIC-IDS-2018 records
python src/test_model.py

# Main test on up to 30,000 CIC-IDS-2018 records + plots
python src/test_rf_100k_multiple_sizes.py
```

This will reproduce the **100k training / 30k testing** results and regenerate all graphs for your presentation.


