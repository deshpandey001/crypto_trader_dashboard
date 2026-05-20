"""
Feature engineering module for advanced trading intelligence.
Creates sophisticated features from raw trading data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple
from scipy import stats

logger = logging.getLogger("crypto_sentiment_dashboard")


class FeatureEngineer:
    """Creates advanced trading intelligence features."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer.
        
        Args:
            df: Input dataframe with trading data
        """
        self.df = df.copy()
        logger.info("FeatureEngineer initialized")
    
    def create_pnl_features(self) -> pd.DataFrame:
        """Create PnL-related features."""
        
        # PnL ratio (profit/loss relative to size)
        self.df['pnl_ratio'] = np.where(
            self.df['size'] != 0,
            self.df['closedpnl'] / (self.df['size'] * self.df['execution_price']),
            0
        )
        self.df['pnl_ratio'] = self.df['pnl_ratio'].fillna(0)
        
        # Win/Loss flag
        self.df['win_loss_flag'] = (self.df['closedpnl'] > 0).astype(int)
        
        # Profit per dollar risked (accounting for leverage)
        self.df['profit_per_leverage'] = np.where(
            self.df['leverage'] != 0,
            self.df['closedpnl'] / self.df['leverage'],
            0
        )
        self.df['profit_per_leverage'] = self.df['profit_per_leverage'].fillna(0)
        
        logger.info("PnL features created")
        return self.df
    
    def create_leverage_features(self) -> pd.DataFrame:
        """Create leverage-related features."""
        
        # Leverage risk score (higher leverage = higher risk)
        self.df['leverage_risk_score'] = (
            (self.df['leverage'] / self.df['leverage'].max()) * 100
        ).fillna(0)
        
        # Leverage category
        def categorize_leverage(lev):
            if lev <= 2:
                return "Low"
            elif lev <= 5:
                return "Medium"
            else:
                return "High"
        
        self.df['leverage_category'] = self.df['leverage'].apply(categorize_leverage)
        
        # High leverage exposure indicator
        self.df['high_leverage_exposure'] = (self.df['leverage'] > 5).astype(int)
        
        logger.info("Leverage features created")
        return self.df
    
    def create_position_features(self) -> pd.DataFrame:
        """Create position-related features."""
        
        # Position size category
        def categorize_position_size(size):
            if size < 1:
                return "Micro"
            elif size < 10:
                return "Small"
            elif size < 50:
                return "Medium"
            else:
                return "Large"
        
        self.df['position_size_category'] = self.df['size'].apply(categorize_position_size)
        
        # Trade direction
        self.df['trade_direction'] = self.df['side'].str.lower()
        
        # Position value
        self.df['position_value'] = self.df['execution_price'] * self.df['size']
        
        logger.info("Position features created")
        return self.df
    
    def create_volatility_features(self) -> pd.DataFrame:
        """Create volatility-related features."""
        
        # Volatility bucket based on price movement
        self.df['volatility_bucket'] = pd.cut(
            self.df['execution_price'],
            bins=5,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        logger.info("Volatility features created")
        return self.df
    
    def create_consistency_features(self) -> pd.DataFrame:
        """Create trader consistency features."""
        
        # Trader consistency score (win rate per trader)
        trader_win_rates = self.df.groupby('account')['win_loss_flag'].agg(['sum', 'count'])
        trader_win_rates['win_rate'] = (
            trader_win_rates['sum'] / trader_win_rates['count']
        ) * 100
        
        self.df['trader_consistency_score'] = self.df['account'].map(
            trader_win_rates['win_rate']
        ).fillna(50)
        
        logger.info("Consistency features created")
        return self.df
    
    def create_sentiment_features(self) -> pd.DataFrame:
        """Create sentiment-related features."""
        
        # Sentiment shift indicator
        self.df['sentiment_shift'] = self.df['sentiment'].diff().fillna(0)
        
        logger.info("Sentiment features created")
        return self.df
    
    def create_risk_features(self) -> pd.DataFrame:
        """Create risk-related features."""
        
        # Risk-reward ratio
        self.df['risk_reward_ratio'] = np.where(
            self.df['leverage'] != 0,
            np.abs(self.df['closedpnl']) / (self.df['leverage'] * self.df['execution_price']),
            0
        )
        self.df['risk_reward_ratio'] = self.df['risk_reward_ratio'].fillna(0)
        
        logger.info("Risk features created")
        return self.df
    
    def create_rolling_features(self, window: int = 10) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            window: Window size for rolling calculations
        """
        
        # Rolling average PnL per trader
        self.df['rolling_avg_pnl'] = (
            self.df.groupby('account')['closedpnl'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        )
        
        # Rolling standard deviation of PnL
        self.df['rolling_std_pnl'] = (
            self.df.groupby('account')['closedpnl'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        ).fillna(0)
        
        logger.info(f"Rolling features created with window={window}")
        return self.df
    
    def create_zscore_features(self) -> pd.DataFrame:
        """Create Z-score normalized features."""
        
        # PnL Z-score (standardized profit/loss)
        self.df['pnl_zscore'] = (
            self.df.groupby('account')['closedpnl'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
        ).fillna(0)
        
        logger.info("Z-score features created")
        return self.df
    
    def classify_traders(self) -> pd.DataFrame:
        """
        Classify traders into risk categories.
        
        Returns:
            Dataframe with trader classifications
        """
        
        # Calculate trader metrics
        trader_metrics = self.df.groupby('account').agg({
            'leverage': 'mean',
            'closedpnl': ['mean', 'std'],
            'win_loss_flag': 'mean',
            'size': 'mean'
        }).reset_index()
        
        trader_metrics.columns = ['account', 'avg_leverage', 'avg_pnl', 'std_pnl', 'win_rate', 'avg_size']
        
        # Classify as retail or whale based on size
        trader_metrics['trader_type'] = np.where(
            trader_metrics['avg_size'] > trader_metrics['avg_size'].median(),
            'Whale',
            'Retail'
        )
        
        # Classify risk level
        def classify_risk(row):
            leverage_score = row['avg_leverage'] / 10 * 100  # Normalized to 0-100
            volatility_score = row['std_pnl'] / abs(row['avg_pnl'] + 1e-8) * 100
            
            risk_score = (leverage_score + volatility_score) / 2
            
            if risk_score > 60:
                return "High Risk"
            elif risk_score > 30:
                return "Medium Risk"
            else:
                return "Low Risk"
        
        trader_metrics['risk_level'] = trader_metrics.apply(classify_risk, axis=1)
        
        # Merge back to main dataframe
        self.df = self.df.merge(
            trader_metrics[['account', 'trader_type', 'risk_level']],
            on='account',
            how='left'
        )
        
        logger.info("Traders classified into risk categories")
        return self.df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """
        Create all features in sequence.
        
        Returns:
            Dataframe with all engineered features
        """
        
        self.create_pnl_features()
        self.create_leverage_features()
        self.create_position_features()
        self.create_volatility_features()
        self.create_consistency_features()
        self.create_sentiment_features()
        self.create_risk_features()
        self.create_rolling_features()
        self.create_zscore_features()
        self.classify_traders()
        
        # Handle any remaining NaN values (numeric vs categorical)
        # Fill numeric columns with 0
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)
        
        # Fill categorical columns - convert to string first to avoid dtype issues
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in ['account', 'symbol', 'side', 'trade_direction']:
                # Convert categorical to string to allow fillna with any value
                if self.df[col].dtype == 'category':
                    self.df[col] = self.df[col].astype(str)
                
                mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                self.df[col] = self.df[col].fillna(mode_val)
        
        logger.info(f"All features engineered successfully: {len(self.df.columns)} features")
        return self.df
    
    def get_feature_summary(self) -> dict:
        """
        Get summary of engineered features.
        
        Returns:
            Dictionary with feature information
        """
        
        features = {
            'pnl_features': [
                'pnl_ratio', 'win_loss_flag', 'profit_per_leverage', 'closedpnl'
            ],
            'leverage_features': [
                'leverage_risk_score', 'leverage_category', 'high_leverage_exposure', 'leverage'
            ],
            'position_features': [
                'position_size_category', 'trade_direction', 'position_value', 'size'
            ],
            'consistency_features': [
                'trader_consistency_score'
            ],
            'sentiment_features': [
                'sentiment_shift', 'sentiment', 'classification'
            ],
            'risk_features': [
                'risk_reward_ratio', 'pnl_zscore', 'rolling_std_pnl'
            ],
            'trader_classification': [
                'trader_type', 'risk_level'
            ]
        }
        
        return features


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Apply feature engineering pipeline.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (engineered dataframe, feature summary)
    """
    engineer = FeatureEngineer(df)
    engineered_df = engineer.engineer_all_features()
    feature_summary = engineer.get_feature_summary()
    
    return engineered_df, feature_summary
