"""
Data preprocessing module for the Crypto Sentiment Dashboard.
Handles data loading, cleaning, and merging.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
from utils import setup_logging, create_output_directories, validate_dataframe_columns

logger = setup_logging()


class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing."""
    
    def __init__(self, fear_greed_path: str, trading_data_path: str):
        """
        Initialize the DataPreprocessor.
        
        Args:
            fear_greed_path: Path to fear/greed index CSV
            trading_data_path: Path to trading data CSV
        """
        self.fear_greed_path = fear_greed_path
        self.trading_data_path = trading_data_path
        self.fear_greed_df = None
        self.trading_df = None
        self.merged_df = None
        
        logger.info("DataPreprocessor initialized")
    
    def load_fear_greed_data(self) -> pd.DataFrame:
        """
        Load and clean fear/greed index data.
        
        Returns:
            Cleaned fear/greed dataframe
        """
        try:
            logger.info(f"Loading fear/greed data from {self.fear_greed_path}")
            df = pd.read_csv(self.fear_greed_path)
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Handle missing values
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['classification'] = df['classification'].str.lower().str.strip()
            
            # Remove rows with missing critical data
            df = df.dropna(subset=['date', 'value', 'classification'])
            
            # Remove duplicates based on date
            df = df.drop_duplicates(subset=['date'], keep='first')
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Convert sentiment to numeric
            df['sentiment'] = df['classification'].apply(
                lambda x: 0 if x == 'fear' else 1
            )
            
            logger.info(f"Fear/Greed data loaded successfully: {len(df)} records")
            self.fear_greed_df = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading fear/greed data: {str(e)}")
            raise
    
    def load_trading_data(self) -> pd.DataFrame:
        """
        Load and clean trading data with memory optimization.
        
        Returns:
            Cleaned trading dataframe
        """
        try:
            logger.info(f"Loading trading data from {self.trading_data_path}")
            
            # Specify dtypes upfront to reduce memory usage
            dtype_dict = {
                'execution price': 'float32',
                'execution_price': 'float32',
                'size tokens': 'float32',
                'size': 'float32',
                'size usd': 'float32',
                'closed pnl': 'float32',
                'closedpnl': 'float32',
                'leverage': 'float32',
                'lev': 'float32',
                'start position': 'float32',
                'start_position': 'float32',
            }
            
            df = pd.read_csv(
                self.trading_data_path,
                dtype=dtype_dict,
                low_memory=True,
                engine='c'
            )
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Column mapping to handle variations in CSV column names
            column_mapping = {
                'time': ['time', 'timestamp ist', 'timestamp', 'datetime'],
                'execution_price': ['execution_price', 'execution price', 'price'],
                'size': ['size', 'size tokens', 'size usd'],
                'closedpnl': ['closedpnl', 'closed pnl'],
                'leverage': ['leverage', 'lev'],
                'account': ['account', 'trader'],
                'symbol': ['symbol', 'coin'],
                'side': ['side', 'direction']
            }
            
            # Helper function to find actual column name
            def find_column(df_cols, expected_names):
                for possible_name in expected_names:
                    if possible_name in df_cols:
                        return possible_name
                return None
            
            # Rename columns to standard names
            for standard_name, possible_names in column_mapping.items():
                actual_col = find_column(df.columns, possible_names)
                if actual_col and actual_col != standard_name:
                    df = df.rename(columns={actual_col: standard_name})
                    logger.info(f"Renamed column '{actual_col}' to '{standard_name}'")
            
            # Convert time to datetime
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], format='mixed', dayfirst=True, errors='coerce')
            else:
                raise ValueError("'time' column not found after mapping")
            
            # Extract date for merging
            df['date'] = df['time'].dt.date
            df['date'] = pd.to_datetime(df['date'])
            
            # Convert numeric columns that weren't caught by dtype dict
            numeric_cols = ['execution_price', 'size', 'closedpnl', 'leverage', 'start_position']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create leverage column if missing
            if 'leverage' not in df.columns:
                df['leverage'] = 1.0
            else:
                df['leverage'] = df['leverage'].fillna(1.0)
            
            # Handle missing values - drop entirely empty rows first
            df = df.dropna(subset=['time', 'account'], how='any')
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['time', 'account'], keep='first')
            
            # Sort by time
            df = df.sort_values('time').reset_index(drop=True)
            
            logger.info(f"Trading data loaded successfully: {len(df)} records")
            self.trading_df = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading trading data: {str(e)}")
            raise
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge fear/greed and trading data on date.
        
        Returns:
            Merged dataframe
        """
        try:
            if self.fear_greed_df is None:
                self.load_fear_greed_data()
            if self.trading_df is None:
                self.load_trading_data()
            
            logger.info("Merging datasets on date")
            
            # Merge on date
            merged_df = pd.merge(
                self.trading_df,
                self.fear_greed_df[['date', 'value', 'classification', 'sentiment']],
                on='date',
                how='left'
            )
            
            # Forward fill sentiment values for missing dates
            merged_df['sentiment'] = merged_df['sentiment'].fillna(method='ffill')
            merged_df['value'] = merged_df['value'].fillna(method='ffill')
            merged_df['classification'] = merged_df['classification'].fillna(method='ffill')
            
            # Fill any remaining NaN with Fear (0)
            merged_df['sentiment'] = merged_df['sentiment'].fillna(0)
            merged_df['value'] = merged_df['value'].fillna(merged_df['value'].mean())
            merged_df['classification'] = merged_df['classification'].fillna('fear')
            
            logger.info(f"Datasets merged successfully: {len(merged_df)} records")
            self.merged_df = merged_df
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            raise
    
    def clean_outliers(self, df: pd.DataFrame, columns: list, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from dataframe.
        
        Args:
            df: Input dataframe
            columns: Columns to check for outliers
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            Dataframe with outliers removed
        """
        try:
            if method == 'iqr':
                for col in columns:
                    if col in df.columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            elif method == 'zscore':
                from scipy import stats
                for col in columns:
                    if col in df.columns:
                        z_scores = np.abs(stats.zscore(df[col]))
                        df = df[z_scores < 3]
            
            logger.info(f"Outliers removed using {method}: {len(df)} records remaining")
            return df
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            return df
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the complete processed dataset.
        
        Returns:
            Processed dataframe
        """
        try:
            if self.merged_df is None:
                self.merge_datasets()
            
            # Remove outliers
            processed_df = self.clean_outliers(
                self.merged_df.copy(),
                ['closedpnl', 'leverage', 'size'],
                method='iqr'
            )
            
            logger.info(f"Data preprocessing complete: {len(processed_df)} final records")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error getting processed data: {str(e)}")
            raise
    
    def save_processed_data(self, output_path: str = "outputs/cleaned_data/processed_data.csv") -> None:
        """
        Save processed data to CSV.
        
        Args:
            output_path: Path to save the processed data
        """
        try:
            if self.merged_df is None:
                self.merge_datasets()
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self.merged_df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise


def preprocess_pipeline(fear_greed_path: str, trading_data_path: str) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        fear_greed_path: Path to fear/greed index CSV
        trading_data_path: Path to trading data CSV
        
    Returns:
        Processed dataframe
    """
    preprocessor = DataPreprocessor(fear_greed_path, trading_data_path)
    processed_data = preprocessor.get_processed_data()
    preprocessor.save_processed_data()
    return processed_data
