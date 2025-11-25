"""
Data Processing Module
Handles loading, preprocessing, and feature engineering for CIC-IDS-2017 dataset
"""

import pandas as pd
import numpy as np
import os
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

CANONICAL_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min"
]

COLUMN_RENAME_MAP = {
    'dst port': "Destination Port",
    'destination port': "Destination Port",
    'protocol': None,
    'timestamp': None,
    'flow duration': "Flow Duration",
    'tot fwd pkts': "Total Fwd Packets",
    'total fwd packets': "Total Fwd Packets",
    'tot bwd pkts': "Total Backward Packets",
    'total backward packets': "Total Backward Packets",
    'totlen fwd pkts': "Total Length of Fwd Packets",
    'totlen bwd pkts': "Total Length of Bwd Packets",
    'fwd pkt len max': "Fwd Packet Length Max",
    'fwd pkt len min': "Fwd Packet Length Min",
    'fwd pkt len mean': "Fwd Packet Length Mean",
    'fwd pkt len std': "Fwd Packet Length Std",
    'bwd pkt len max': "Bwd Packet Length Max",
    'bwd pkt len min': "Bwd Packet Length Min",
    'bwd pkt len mean': "Bwd Packet Length Mean",
    'bwd pkt len std': "Bwd Packet Length Std",
    'flow byts/s': "Flow Bytes/s",
    'flow bytes/s': "Flow Bytes/s",
    'flow pkts/s': "Flow Packets/s",
    'flow packets/s': "Flow Packets/s",
    'fwd iat tot': "Fwd IAT Total",
    'fwd iat total': "Fwd IAT Total",
    'bwd iat tot': "Bwd IAT Total",
    'bwd iat total': "Bwd IAT Total",
    'fwd iat mean': "Fwd IAT Mean",
    'fwd iat std': "Fwd IAT Std",
    'fwd iat max': "Fwd IAT Max",
    'fwd iat min': "Fwd IAT Min",
    'bwd iat mean': "Bwd IAT Mean",
    'bwd iat std': "Bwd IAT Std",
    'bwd iat max': "Bwd IAT Max",
    'bwd iat min': "Bwd IAT Min",
    'fwd psh flags': "Fwd PSH Flags",
    'fwd urg flags': "Fwd URG Flags",
    'bwd psh flags': "Bwd PSH Flags",
    'bwd urg flags': "Bwd URG Flags",
    'fwd header len': "Fwd Header Length",
    'fwd header length': "Fwd Header Length",
    'fwd header length.1': "Fwd Header Length.1",
    'bwd header len': "Bwd Header Length",
    'bwd header length': "Bwd Header Length",
    'fwd pkts/s': "Fwd Packets/s",
    'bwd pkts/s': "Bwd Packets/s",
    'pkt len min': "Min Packet Length",
    'pkt len max': "Max Packet Length",
    'pkt len mean': "Packet Length Mean",
    'pkt len std': "Packet Length Std",
    'pkt len var': "Packet Length Variance",
    'fin flag cnt': "FIN Flag Count",
    'syn flag cnt': "SYN Flag Count",
    'rst flag cnt': "RST Flag Count",
    'psh flag cnt': "PSH Flag Count",
    'ack flag cnt': "ACK Flag Count",
    'urg flag cnt': "URG Flag Count",
    'cwe flag count': "CWE Flag Count",
    'ece flag cnt': "ECE Flag Count",
    'down/up ratio': "Down/Up Ratio",
    'pkt size avg': "Average Packet Size",
    'fwd seg size avg': "Avg Fwd Segment Size",
    'bwd seg size avg': "Avg Bwd Segment Size",
    'fwd byts/b avg': "Fwd Avg Bytes/Bulk",
    'fwd pkts/b avg': "Fwd Avg Packets/Bulk",
    'fwd blk rate avg': "Fwd Avg Bulk Rate",
    'bwd byts/b avg': "Bwd Avg Bytes/Bulk",
    'bwd pkts/b avg': "Bwd Avg Packets/Bulk",
    'bwd blk rate avg': "Bwd Avg Bulk Rate",
    'subflow fwd pkts': "Subflow Fwd Packets",
    'subflow fwd byts': "Subflow Fwd Bytes",
    'subflow bwd pkts': "Subflow Bwd Packets",
    'subflow bwd byts': "Subflow Bwd Bytes",
    'init fwd win byts': "Init_Win_bytes_forward",
    'init bwd win byts': "Init_Win_bytes_backward",
    'fwd act data pkts': "act_data_pkt_fwd",
    'fwd seg size min': "min_seg_size_forward",
    'active mean': "Active Mean",
    'active std': "Active Std",
    'active max': "Active Max",
    'active min': "Active Min",
    'idle mean': "Idle Mean",
    'idle std': "Idle Std",
    'idle max': "Idle Max",
    'idle min': "Idle Min",
    'label': 'Label',
    ' total fwd packets': "Total Fwd Packets"
}

DROP_COLUMNS = {
    'Protocol',
    'Timestamp',
}


class DataProcessor:
    """Process CIC-IDS-2017 dataset files"""
    
    def __init__(self):
        self.feature_columns = None
        self.label_column = 'Label'  # Try ' Label' if not found
        self.label_encoder = LabelEncoder()
        self.label_mapping = None  # Will store {encoded_value: label_name}
        self.classification_mode = 'binary'  # Default to binary (to match train_model.py)
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and ensure required features exist"""
        rename_map = {}
        for col in df.columns:
            key = col.strip().lower()
            if key in COLUMN_RENAME_MAP:
                new_name = COLUMN_RENAME_MAP[key]
                if new_name is None:
                    rename_map[col] = None
                else:
                    rename_map[col] = new_name
        # Apply renames
        for col, new_name in list(rename_map.items()):
            if new_name is None:
                df = df.drop(columns=col, errors='ignore')
            else:
                if new_name in df.columns and new_name != col:
                    df[new_name] = df[new_name].combine_first(df[col])
                    df = df.drop(columns=col)
                else:
                    df = df.rename(columns={col: new_name})
        # Drop explicitly flagged columns
        df = df.drop(columns=[c for c in df.columns if c in DROP_COLUMNS], errors='ignore')
        # Ensure label column normalized
        if 'Label' not in df.columns:
            for candidate in ['label', ' Label']:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: 'Label'})
                    break
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        # If Fwd Header Length.1 missing, copy from Fwd Header Length
        if 'Fwd Header Length.1' not in df.columns and 'Fwd Header Length' in df.columns:
            df['Fwd Header Length.1'] = df['Fwd Header Length']
        # Ensure all canonical features exist
        for feature in CANONICAL_FEATURES:
            if feature not in df.columns:
                df[feature] = 0
        # Reorder columns (keep label if present)
        ordered_cols = [col for col in CANONICAL_FEATURES if col in df.columns]
        other_cols = [col for col in df.columns if col not in ordered_cols and col != 'Label']
        if 'Label' in df.columns:
            df = df[ordered_cols + other_cols + ['Label']]
        else:
            df = df[ordered_cols + other_cols]
        return df

    def load_training_data(self, data_dir: str = 'data_original') -> pd.DataFrame:
        """
        Load all CSV files from training directory
        
        Args:
            data_dir: Directory containing training CSV files
            
        Returns:
            Combined DataFrame with all training data
        """
        print("Loading training data from data_original/...")
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files")
        
        dfs = []
        for file in csv_files:
            print(f"  Loading {file}...")
            df = pd.read_csv(os.path.join(data_dir, file))
            
            # Remove any rows that look like header rows (contain column names as values)
            if len(df) > 0 and len(df.columns) > 0:
                first_col = df.columns[0]
                if df[first_col].dtype == 'object':
                    # Check if any row has column names as values
                    header_patterns = ['dst port', 'flow duration', 'label', 'protocol', 'timestamp']
                    mask = df[first_col].astype(str).str.lower().str.strip().isin([p.lower() for p in header_patterns])
                    if mask.any():
                        rows_to_drop = df[mask].index
                        print(f"    Removing {len(rows_to_drop)} header-like rows from {file}")
                        df = df.drop(index=rows_to_drop).reset_index(drop=True)
            
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Total training records: {len(combined_df):,}")
        return combined_df
    
    def load_test_data(self, data_dir: str = 'Testing_data', max_records_per_file: int = 5000) -> pd.DataFrame:
        """
        Load test data
        
        Args:
            data_dir: Directory containing test CSV files
            max_records_per_file: Maximum records to load from each file
            
        Returns:
            Combined DataFrame with all test data
        """
        print(f"Loading test data from {data_dir}/...")
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files")
        
        dfs = []
        for file in csv_files:
            print(f"  Loading {file} (max {max_records_per_file:,} records)...")
            try:
                df = pd.read_csv(os.path.join(data_dir, file), nrows=max_records_per_file)
                
                # Remove any rows that look like header rows (contain column names as values)
                # Check if first data column contains column name patterns
                if len(df) > 0 and len(df.columns) > 0:
                    first_col = df.columns[0]
                    if df[first_col].dtype == 'object':
                        # Check if any row has column names as values (e.g., 'Dst Port', 'Flow Duration')
                        header_patterns = ['dst port', 'flow duration', 'label', 'protocol', 'timestamp']
                        mask = df[first_col].astype(str).str.lower().str.strip().isin([p.lower() for p in header_patterns])
                        if mask.any():
                            rows_to_drop = df[mask].index
                            print(f"    Removing {len(rows_to_drop)} header-like rows from {file}")
                            df = df.drop(index=rows_to_drop).reset_index(drop=True)
                
                dfs.append(df)
            except Exception as e:
                print(f"    Error loading {file}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No test data files could be loaded")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Total test records: {len(combined_df):,}")
        return combined_df
    
    def preprocess_data(self, df: pd.DataFrame, classification_mode: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data: clean, normalize features
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        print(f"\nPreprocessing data (shape: {df.shape})...")

        df.columns = df.columns.str.strip()
        df = self._standardize_dataframe(df)

        is_training = self.feature_columns is None

        if self.feature_columns is None:
            self.feature_columns = [col for col in CANONICAL_FEATURES if col in df.columns]

        missing_features = [col for col in self.feature_columns if col not in df.columns]
        for col in missing_features:
            df[col] = 0

        X = df[self.feature_columns].copy()
        y = df['Label'].copy() if 'Label' in df.columns else pd.Series(np.zeros(len(df)))

        # Ensure all features are numeric first
        print("  Converting features to numeric...")
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        print("  Handling missing values and infinities...")
        X = X.replace([np.inf, -np.inf, "Infinity", "-Infinity", "inf", "-inf", "NaN", "nan"], np.nan)
        
        # Fill missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Fill any remaining NaN (categorical or converted from strings)
        X = X.fillna(0)
        
        # Final check: ensure only numeric columns remain
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"  Warning: Dropping {len(non_numeric_cols)} non-numeric columns: {list(non_numeric_cols)}")
            X = X.drop(columns=non_numeric_cols)
            # Update feature_columns to match
            self.feature_columns = [col for col in self.feature_columns if col in X.columns]
        
        # Remove columns with all zeros or constants (training phase only)
        if is_training:
            print("  Removing constant columns...")
            constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
            if constant_cols:
                print(f"  Removed {len(constant_cols)} constant columns")
                X = X.drop(columns=constant_cols)
                self.feature_columns = [col for col in self.feature_columns if col not in constant_cols]
        
        # Clean labels - use specified classification mode
        print("  Processing labels...")
        if classification_mode is None:
            classification_mode = getattr(self, 'classification_mode', 'binary')
        # Determine if encoder should be fitted (fit during training, transform during testing)
        fit_encoder = is_training
        y = self.process_labels(y, classification_mode=classification_mode, fit_encoder=fit_encoder)
        
        print(f"Preprocessed data shape: {X.shape}")
        print(f"Number of features: {len(X.columns)}")
        print(f"Label distribution:\n{y.value_counts()}")
        
        return X, y
    
    def process_labels(self, y: pd.Series, classification_mode: str = 'binary', fit_encoder: bool = True) -> pd.Series:
        """
        Process labels for binary or multi-class classification
        
        Args:
            y: Label series
            classification_mode: 'binary' or 'multiclass'
            fit_encoder: Whether to fit the label encoder (True for training, False for testing)
            
        Returns:
            Encoded label series (binary: 0=benign, 1=attack; multiclass: numeric labels)
        """
        # Strip whitespace and convert to uppercase for consistency
        y_cleaned = y.str.strip().str.upper()
        
        if classification_mode == 'binary':
            # Binary classification (anything not BENIGN is an attack)
            y_binary = (y_cleaned != 'BENIGN').astype(int)
            return pd.Series(y_binary, index=y.index)
        
        else:  # multiclass
            # Handle encoding issues and normalize labels
            y_cleaned = y_cleaned.str.replace(r'[^\x00-\x7F]', '-', regex=True)  # Replace non-ASCII characters
            y_cleaned = y_cleaned.str.replace('  ', ' ', regex=False)  # Fix double spaces
            y_cleaned = y_cleaned.str.replace('WEB ATTACK', 'WEB-ATTACK', regex=False)  # Normalize Web Attack
            y_cleaned = y_cleaned.str.replace('BRUTE FORCE', 'BRUTE-FORCE', regex=False)  # Normalize Brute Force
            y_cleaned = y_cleaned.str.replace('WEB-ATTACK -', 'WEB-ATTACK', regex=False)  # Fix double dash
            y_cleaned = y_cleaned.str.replace('WEB-ATTACK-BRUTE-FORCE', 'WEB-ATTACK-BRUTE-FORCE', regex=False)  # Keep as is
            
            # Fit or transform using LabelEncoder
            if fit_encoder:
                y_encoded = self.label_encoder.fit_transform(y_cleaned)
                # Create mapping for reverse transformation
                self.label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
                print(f"  Label classes: {list(self.label_encoder.classes_)}")
            else:
                # For testing, use known classes and handle unseen labels
                try:
                    y_encoded = self.label_encoder.transform(y_cleaned)
                except ValueError:
                    # Handle unseen labels by mapping them to the most common class (usually 0 for BENIGN)
                    # or to a default "UNKNOWN" class
                    print(f"  Warning: Some test labels not seen during training")
                    y_encoded = []
                    for label in y_cleaned:
                        if label in self.label_encoder.classes_:
                            y_encoded.append(self.label_encoder.transform([label])[0])
                        else:
                            # Map unseen labels to 0 (BENIGN) if available, otherwise to first class
                            default_class = 0 if 'BENIGN' in self.label_encoder.classes_ else 0
                            y_encoded.append(default_class)
                            print(f"    Mapped unseen label '{label}' to class {default_class}")
                    y_encoded = np.array(y_encoded)
            
            return pd.Series(y_encoded, index=y.index)
    
    def get_label_name(self, encoded_value: int) -> str:
        """Convert encoded label value back to original label name"""
        if self.label_mapping is not None:
            return self.label_mapping.get(encoded_value, f"Unknown({encoded_value})")
        elif hasattr(self.label_encoder, 'classes_'):
            if encoded_value < len(self.label_encoder.classes_):
                return self.label_encoder.classes_[encoded_value]
        return f"Unknown({encoded_value})"
    
    def get_all_label_names(self) -> List[str]:
        """Get all label class names"""
        if hasattr(self.label_encoder, 'classes_'):
            return list(self.label_encoder.classes_)
        elif self.label_mapping is not None:
            return list(self.label_mapping.values())
        return []
    
    def create_sample_dataset(self, df: pd.DataFrame, n_samples: int = 100000, 
                             random_state: int = 42) -> pd.DataFrame:
        """
        Create a sample dataset for faster training/testing
        
        Args:
            df: Full dataset
            n_samples: Number of samples to create
            random_state: Random seed
            
        Returns:
            Sampled DataFrame
        """
        if len(df) <= n_samples:
            return df
        
        # Ensure label column exists
        if 'Label' not in df.columns:
            for col in df.columns:
                if col.strip().lower() == 'label':
                    df = df.rename(columns={col: 'Label'})
                    break
        
        print(f"\nCreating sample dataset ({n_samples:,} samples)...")
        
        # Improved stratified sampling for multiclass with minimum samples per class
        try:
            n_classes = df['Label'].nunique()
            min_samples_per_class = max(100, n_samples // (n_classes * 5))  # At least 100 samples per class, or 1/5th of n_samples/classes
            samples_per_class = max(min_samples_per_class, n_samples // n_classes)
            
            print(f"  Target samples per class: {samples_per_class} (minimum: {min_samples_per_class})")
            
            sampled_df = df.groupby('Label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), samples_per_class),
                                 random_state=random_state)
            )
            
            # If we got fewer samples than requested, fill the rest randomly
            if len(sampled_df) < n_samples:
                remaining = n_samples - len(sampled_df)
                remaining_df = df[~df.index.isin(sampled_df.index)]
                if len(remaining_df) > 0:
                    additional = remaining_df.sample(min(remaining, len(remaining_df)), 
                                                   random_state=random_state)
                    sampled_df = pd.concat([sampled_df, additional], ignore_index=True)
            
            # Final random shuffle
            sampled_df = sampled_df.sample(n=min(len(sampled_df), n_samples), 
                                         random_state=random_state).reset_index(drop=True)
            
        except Exception as e:
            print(f"  Warning: Stratified sampling failed: {e}, using random sampling")
            sampled_df = df.sample(n=min(n_samples, len(df)), random_state=random_state)
        
        print(f"Sample dataset shape: {sampled_df.shape}")
        label_dist = sampled_df['Label'].value_counts()
        print(f"Label distribution in sample:")
        for label, count in label_dist.items():
            print(f"  {label}: {count}")
        
        return sampled_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_columns if self.feature_columns is not None else []


def main():
    """Test data processing"""
    processor = DataProcessor()
    
    print("="*60)
    print("DATA PROCESSING TEST")
    print("="*60)
    
    # Load training data
    train_df = processor.load_training_data()
    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Columns: {len(train_df.columns)}")
    
    # Create sample for testing
    sample_df = processor.create_sample_dataset(train_df, n_samples=50000)
    
    # Preprocess
    X, y = processor.preprocess_data(sample_df)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Label distribution:\n{y.value_counts()}")
    print("="*60)


if __name__ == "__main__":
    main()

