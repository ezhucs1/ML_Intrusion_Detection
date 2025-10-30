# 🚀 Setup Guide for GitHub Clone

After cloning this repository, follow these steps to set up the intrusion detection system.

## ⚙️ Setup Steps

### 1. Download the Dataset

**Required:** Download CIC-IDS-2017 dataset from:
https://www.unb.ca/cic/datasets/ids-2017.html

### 2. Create Required Directories

```bash
mkdir -p data_original
mkdir -p Testing_data
mkdir -p models
```

### 3. Place Training Data

Copy your training CSV files to `data_original/`:
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

### 4. Place Testing Data

Copy your testing CSV files to `Testing_data/`:
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

### 5. Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 6. Train the Model

```bash
python src/train_model.py
```

This will:
- Load training data from `data_original/`
- Preprocess and clean the data
- Train 3 ML models
- Select the best one
- Save to `models/intrusion_detection_model.pkl`

### 7. Test the Model

```bash
python src/test_model.py
```

## 📊 Expected Output

After training, you should see:
- Model type: Gradient Boosting
- Accuracy: ~99%+
- Model saved to `models/intrusion_detection_model.pkl`

## ⚠️ Important Notes

1. **Dataset Required**: You must download CIC-IDS-2017 dataset separately
2. **Training Time**: First training may take 10-30 minutes
3. **Disk Space**: Dataset requires ~7GB total
4. **Model File**: After training, model saved in `models/` (gitignored)

## 🧪 Quick Test

After training, test with:
```bash
python -c "
import joblib
artifacts = joblib.load('models/intrusion_detection_model.pkl')
print('Model:', artifacts['model_name'])
print('✓ Model loaded successfully!')
"
```

## 📚 Documentation

- **README.md** - Overview and usage
- **SIMPLE_EXPLANATION.md** - How it works
- **PRESENTATION_OUTLINE.md** - Presentation guide

## ❓ Troubleshooting

**Issue**: `FileNotFoundError` for CSV files  
**Solution**: Make sure dataset is placed in correct directories

**Issue**: Memory error during training  
**Solution**: Edit `src/train_model.py` and reduce `n_samples` parameter

**Issue**: Missing modules  
**Solution**: `pip install -r requirements.txt`

---

**Happy Training! 🎉**

