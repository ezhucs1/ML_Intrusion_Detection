"""
Data Processing Module
Handles loading, preprocessing, and feature engineering for CIC-IDS-2017 dataset
"""

import pandas as pd
import numpy as np
import os
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Process CIC-IDS-2017 dataset files"""
    
    def __init__(self):
        self.feature_columns = None
        self.label_column = 'Label'  # Try ' Label' if not found
    
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
                dfs.append(df)
            except Exception as e:
                print(f"    Error loading {file}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No test data files could be loaded")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Total test records: {len(combined_df):,}")
        return combined_df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data: clean, normalize features
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        print(f"\nPreprocessing data (shape: {df.shape})...")
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Store feature column names
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != 'Label']
        
        # Extract features and labels
        X = df[self.feature_columns].copy()
        y = df['Label'].copy()
        
        # Handle missing values and infinities
        print("  Handling missing values and infinities...")
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Fill any remaining NaN (categorical)
        X = X.fillna(0)
        
        # Remove columns with all zeros or constants
        print("  Removing constant columns...")
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            print(f"  Removed {len(constant_cols)} constant columns")
            X = X.drop(columns=constant_cols)
            self.feature_columns = [col for col in self.feature_columns if col not in constant_cols]
        
        # Clean labels - convert to binary (attack vs benign)
        print("  Processing labels...")
        y = self.process_labels(y)
        
        print(f"Preprocessed data shape: {X.shape}")
        print(f"Number of features: {len(X.columns)}")
        print(f"Label distribution:\n{y.value_counts()}")
        
        return X, y
    
    def process_labels(self, y: pd.Series) -> pd.Series:
        """
        Process labels to binary classification (attack vs benign)
        
        Args:
            y: Label series
            
        Returns:
            Binary label series (0=benign, 1=attack)
        """
        # Strip whitespace and convert to uppercase for consistency
        y = y.str.strip().str.upper()
        
        # Convert to binary (anything not BENIGN is an attack)
        y_binary = (y != 'BENIGN').astype(int)
        
        return y_binary
    
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
        
        print(f"\nCreating sample dataset ({n_samples:,} samples)...")
        
        # Stratified sampling to maintain label distribution
        try:
            sampled_df = df.groupby('Label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), n_samples // df['Label'].nunique()),
                                 random_state=random_state)
            )
        except:
            sampled_df = df.sample(n=n_samples, random_state=random_state)
        
        print(f"Sample dataset shape: {sampled_df.shape}")
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

