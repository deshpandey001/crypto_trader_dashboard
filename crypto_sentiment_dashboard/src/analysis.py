"""
Advanced analytics module for trading intelligence.
Performs comprehensive statistical analysis on trading data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("crypto_sentiment_dashboard")


class AdvancedAnalytics:
    """Performs advanced trading analytics and analysis."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analytics engine.
        
        Args:
            df: Input dataframe with engineered features
        """
        self.df = df
        self.analysis_results = {}
        logger.info("AdvancedAnalytics initialized")
    
    def calculate_win_rate_analysis(self) -> Dict:
        """
        Analyze win rates across different dimensions.
        
        Returns:
            Dictionary with win rate statistics
        """
        
        results = {
            'overall_win_rate': (self.df['win_loss_flag'].sum() / len(self.df)) * 100,
            'overall_loss_rate': ((1 - self.df['win_loss_flag']).sum() / len(self.df)) * 100,
            'by_sentiment': {},
            'by_leverage_category': {},
            'by_trader_type': {},
            'by_symbol': {}
        }
        
        # Win rate by sentiment
        for sentiment in [0, 1]:
            sentiment_label = "Fear" if sentiment == 0 else "Greed"
            mask = self.df['sentiment'] == sentiment
            if mask.sum() > 0:
                wr = (self.df[mask]['win_loss_flag'].sum() / mask.sum()) * 100
                results['by_sentiment'][sentiment_label] = wr
        
        # Win rate by leverage category
        for cat in self.df['leverage_category'].unique():
            if pd.notna(cat):
                mask = self.df['leverage_category'] == cat
                if mask.sum() > 0:
                    wr = (self.df[mask]['win_loss_flag'].sum() / mask.sum()) * 100
                    results['by_leverage_category'][cat] = wr
        
        # Win rate by trader type
        for trader_type in self.df['trader_type'].unique():
            if pd.notna(trader_type):
                mask = self.df['trader_type'] == trader_type
                if mask.sum() > 0:
                    wr = (self.df[mask]['win_loss_flag'].sum() / mask.sum()) * 100
                    results['by_trader_type'][trader_type] = wr
        
        # Win rate by symbol
        for symbol in self.df['symbol'].unique():
            if pd.notna(symbol):
                mask = self.df['symbol'] == symbol
                if mask.sum() > 0:
                    wr = (self.df[mask]['win_loss_flag'].sum() / mask.sum()) * 100
                    results['by_symbol'][symbol] = wr
        
        logger.info("Win rate analysis completed")
        self.analysis_results['win_rate'] = results
        return results
    
    def calculate_profitability_analysis(self) -> Dict:
        """
        Analyze profitability across different dimensions.
        
        Returns:
            Dictionary with profitability statistics
        """
        
        results = {
            'total_pnl': self.df['closedpnl'].sum(),
            'avg_pnl': self.df['closedpnl'].mean(),
            'median_pnl': self.df['closedpnl'].median(),
            'max_profit': self.df['closedpnl'].max(),
            'max_loss': self.df['closedpnl'].min(),
            'by_sentiment': {},
            'by_leverage_category': {},
            'by_trader_type': {},
            'by_symbol': {}
        }
        
        # Profitability by sentiment
        for sentiment in [0, 1]:
            sentiment_label = "Fear" if sentiment == 0 else "Greed"
            mask = self.df['sentiment'] == sentiment
            if mask.sum() > 0:
                results['by_sentiment'][sentiment_label] = {
                    'total_pnl': self.df[mask]['closedpnl'].sum(),
                    'avg_pnl': self.df[mask]['closedpnl'].mean(),
                    'count': mask.sum()
                }
        
        # Profitability by leverage category
        for cat in self.df['leverage_category'].unique():
            if pd.notna(cat):
                mask = self.df['leverage_category'] == cat
                if mask.sum() > 0:
                    results['by_leverage_category'][cat] = {
                        'total_pnl': self.df[mask]['closedpnl'].sum(),
                        'avg_pnl': self.df[mask]['closedpnl'].mean(),
                        'count': mask.sum()
                    }
        
        # Profitability by trader type
        for trader_type in self.df['trader_type'].unique():
            if pd.notna(trader_type):
                mask = self.df['trader_type'] == trader_type
                if mask.sum() > 0:
                    results['by_trader_type'][trader_type] = {
                        'total_pnl': self.df[mask]['closedpnl'].sum(),
                        'avg_pnl': self.df[mask]['closedpnl'].mean(),
                        'count': mask.sum()
                    }
        
        # Profitability by symbol
        for symbol in self.df['symbol'].unique():
            if pd.notna(symbol):
                mask = self.df['symbol'] == symbol
                if mask.sum() > 0:
                    results['by_symbol'][symbol] = {
                        'total_pnl': self.df[mask]['closedpnl'].sum(),
                        'avg_pnl': self.df[mask]['closedpnl'].mean(),
                        'count': mask.sum()
                    }
        
        logger.info("Profitability analysis completed")
        self.analysis_results['profitability'] = results
        return results
    
    def calculate_leverage_analysis(self) -> Dict:
        """
        Analyze leverage usage and its impact.
        
        Returns:
            Dictionary with leverage statistics
        """
        
        results = {
            'avg_leverage': self.df['leverage'].mean(),
            'max_leverage': self.df['leverage'].max(),
            'min_leverage': self.df['leverage'].min(),
            'high_leverage_trades': (self.df['leverage'] > 5).sum(),
            'high_leverage_percentage': ((self.df['leverage'] > 5).sum() / len(self.df)) * 100,
            'avg_pnl_by_leverage': self.df.groupby('leverage_category')['closedpnl'].mean().to_dict(),
            'correlation_leverage_pnl': self.df['leverage'].corr(self.df['closedpnl']),
            'correlation_leverage_risk': self.df['leverage'].corr(self.df['pnl_zscore'].abs())
        }
        
        logger.info("Leverage analysis completed")
        self.analysis_results['leverage'] = results
        return results
    
    def calculate_trader_segmentation(self) -> Dict:
        """
        Segment traders by performance and behavior.
        
        Returns:
            Dictionary with trader segmentation
        """
        
        trader_stats = self.df.groupby('account').agg({
            'closedpnl': ['sum', 'mean', 'std', 'count'],
            'win_loss_flag': 'mean',
            'leverage': 'mean',
            'size': 'mean',
            'trader_consistency_score': 'first',
            'trader_type': 'first',
            'risk_level': 'first'
        }).reset_index()
        
        trader_stats.columns = [
            'trader', 'total_pnl', 'avg_pnl', 'std_pnl', 'trades_count',
            'win_rate', 'avg_leverage', 'avg_size', 'consistency_score',
            'trader_type', 'risk_level'
        ]
        
        results = {
            'total_traders': len(trader_stats),
            'top_performers': trader_stats.nlargest(5, 'total_pnl')[
                ['trader', 'total_pnl', 'win_rate', 'trades_count']
            ].to_dict('records'),
            'worst_performers': trader_stats.nsmallest(5, 'total_pnl')[
                ['trader', 'total_pnl', 'win_rate', 'trades_count']
            ].to_dict('records'),
            'by_risk_level': trader_stats.groupby('risk_level').agg({
                'total_pnl': 'mean',
                'win_rate': 'mean',
                'trader': 'count'
            }).to_dict('index'),
            'by_trader_type': trader_stats.groupby('trader_type').agg({
                'total_pnl': 'mean',
                'win_rate': 'mean',
                'trader': 'count'
            }).to_dict('index')
        }
        
        logger.info("Trader segmentation completed")
        self.analysis_results['trader_segmentation'] = results
        return results
    
    def calculate_volatility_analysis(self) -> Dict:
        """
        Analyze market volatility and its impact on trading.
        
        Returns:
            Dictionary with volatility statistics
        """
        
        results = {
            'pnl_volatility': self.df['closedpnl'].std(),
            'price_volatility': self.df['execution_price'].std(),
            'by_volatility_bucket': self.df.groupby('volatility_bucket').agg({
                'closedpnl': 'mean',
                'win_loss_flag': 'mean',
                'leverage': 'mean'
            }).to_dict('index')
        }
        
        logger.info("Volatility analysis completed")
        self.analysis_results['volatility'] = results
        return results
    
    def calculate_trade_frequency_analysis(self) -> Dict:
        """
        Analyze trade frequency patterns.
        
        Returns:
            Dictionary with trade frequency statistics
        """
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        daily_trades = self.df.groupby('date').size()
        
        results = {
            'total_trades': len(self.df),
            'avg_daily_trades': daily_trades.mean(),
            'max_daily_trades': daily_trades.max(),
            'min_daily_trades': daily_trades.min(),
            'trades_by_symbol': self.df['symbol'].value_counts().to_dict(),
            'trades_by_sentiment': self.df['sentiment'].value_counts().to_dict(),
            'trades_by_side': self.df['side'].value_counts().to_dict()
        }
        
        logger.info("Trade frequency analysis completed")
        self.analysis_results['trade_frequency'] = results
        return results
    
    def calculate_drawdown_analysis(self) -> Dict:
        """
        Analyze maximum drawdown and recovery.
        
        Returns:
            Dictionary with drawdown statistics
        """
        
        # Calculate cumulative PnL
        self.df = self.df.sort_values('time')
        cumulative_pnl = self.df['closedpnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / (running_max + 1e-8)
        
        results = {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown.mean(),
            'max_cumulative_pnl': cumulative_pnl.max(),
            'min_cumulative_pnl': cumulative_pnl.min(),
            'final_cumulative_pnl': cumulative_pnl.iloc[-1]
        }
        
        logger.info("Drawdown analysis completed")
        self.analysis_results['drawdown'] = results
        return results
    
    def calculate_liquidation_risk(self) -> Dict:
        """
        Analyze liquidation risk indicators.
        
        Returns:
            Dictionary with liquidation risk statistics
        """
        
        # Check if 'event' column exists; if not, skip liquidation analysis
        if 'event' not in self.df.columns:
            logger.warning("'event' column not found in data, skipping liquidation analysis")
            results = {
                'total_liquidations': 0,
                'liquidation_rate': 0,
                'avg_leverage_at_liquidation': 0,
                'pnl_at_liquidation': 0,
                'liquidations_by_trader': {},
                'high_risk_traders': self.df[self.df['risk_level'] == 'High Risk']['account'].nunique() if 'risk_level' in self.df.columns else 0
            }
        else:
            liquidations = self.df[self.df['event'] == 'Liquidation']
            
            results = {
                'total_liquidations': len(liquidations),
                'liquidation_rate': (len(liquidations) / len(self.df)) * 100,
                'avg_leverage_at_liquidation': liquidations['leverage'].mean(),
                'pnl_at_liquidation': liquidations['closedpnl'].mean(),
                'liquidations_by_trader': liquidations['account'].value_counts().to_dict(),
                'high_risk_traders': self.df[self.df['risk_level'] == 'High Risk']['account'].nunique() if 'risk_level' in self.df.columns else 0
            }
        
        logger.info("Liquidation risk analysis completed")
        self.analysis_results['liquidation_risk'] = results
        return results
    
    def run_all_analyses(self) -> Dict:
        """
        Run all analyses.
        
        Returns:
            Dictionary with all analysis results
        """
        
        self.calculate_win_rate_analysis()
        self.calculate_profitability_analysis()
        self.calculate_leverage_analysis()
        self.calculate_trader_segmentation()
        self.calculate_volatility_analysis()
        self.calculate_trade_frequency_analysis()
        self.calculate_drawdown_analysis()
        self.calculate_liquidation_risk()
        
        logger.info("All analyses completed successfully")
        return self.analysis_results


def perform_analytics(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """
    Perform comprehensive analytics on trading data.
    
    Args:
        df: Input dataframe with engineered features
        
    Returns:
        Tuple of (analysis results, updated dataframe)
    """
    
    analytics = AdvancedAnalytics(df)
    results = analytics.run_all_analyses()
    
    return results, analytics.df
