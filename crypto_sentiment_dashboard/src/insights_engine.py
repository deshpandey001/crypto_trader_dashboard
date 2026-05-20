"""
AI Insights Engine for generating trading intelligence recommendations.
Produces actionable insights from data analysis.
"""

import pandas as pd
import logging
from typing import Dict, List

logger = logging.getLogger("crypto_sentiment_dashboard")


class InsightsEngine:
    """Generates AI-powered trading insights and recommendations."""
    
    def __init__(self, df: pd.DataFrame, analysis_results: Dict):
        """
        Initialize insights engine.
        
        Args:
            df: Input dataframe with engineered features
            analysis_results: Results from analytics module
        """
        self.df = df
        self.analysis_results = analysis_results
        self.insights = []
        self.recommendations = []
        self.warnings = []
        
        logger.info("InsightsEngine initialized")
    
    def generate_sentiment_insights(self) -> None:
        """Generate insights about sentiment and trading behavior."""
        
        win_rates = self.analysis_results.get('win_rate', {}).get('by_sentiment', {})
        profitability = self.analysis_results.get('profitability', {}).get('by_sentiment', {})
        
        if win_rates:
            fear_wr = win_rates.get('Fear', 0)
            greed_wr = win_rates.get('Greed', 0)
            
            if fear_wr > greed_wr:
                self.insights.append(
                    f"Fear periods show {fear_wr-greed_wr:.1f}% higher win rate than greed periods. "
                    f"Traders tend to be more disciplined during market downturns."
                )
            else:
                self.insights.append(
                    f"Greed periods show {greed_wr-fear_wr:.1f}% higher win rate than fear periods. "
                    f"Traders capitalize on bullish sentiment."
                )
        
        if profitability:
            fear_pnl = profitability.get('Fear', {}).get('avg_pnl', 0)
            greed_pnl = profitability.get('Greed', {}).get('avg_pnl', 0)
            
            if fear_pnl > 0 and greed_pnl > 0:
                self.insights.append(
                    f"Average trade profits: Fear = ${fear_pnl:.2f}, Greed = ${greed_pnl:.2f}. "
                    f"Both sentiments remain profitable, but discipline varies."
                )
    
    def generate_leverage_insights(self) -> None:
        """Generate insights about leverage usage."""
        
        leverage_data = self.analysis_results.get('leverage', {})
        
        high_leverage_pct = leverage_data.get('high_leverage_percentage', 0)
        correlation = leverage_data.get('correlation_leverage_pnl', 0)
        
        if high_leverage_pct > 30:
            self.warnings.append(
                f"⚠️ HIGH LEVERAGE EXPOSURE: {high_leverage_pct:.1f}% of trades use leverage > 5x. "
                f"This indicates elevated liquidation risk."
            )
            self.recommendations.append(
                "Implement leverage position limits. Consider restricting leverage to 3-5x "
                "during high volatility periods."
            )
        
        if correlation < -0.3:
            self.insights.append(
                f"High leverage shows negative correlation with profits ({correlation:.2f}). "
                f"Reducing leverage could improve net returns."
            )
    
    def generate_trader_insights(self) -> None:
        """Generate insights about trader behavior and segmentation."""
        
        segmentation = self.analysis_results.get('trader_segmentation', {})
        total_traders = segmentation.get('total_traders', 0)
        
        if total_traders > 0:
            top_performers = segmentation.get('top_performers', [])
            if top_performers:
                top_pnl = top_performers[0]['total_pnl']
                self.insights.append(
                    f"Top performer ({top_performers[0]['trader']}) generated ${top_pnl:.2f} PnL. "
                    f"Study their trade patterns for best practices."
                )
        
        # Risk level analysis
        by_risk = segmentation.get('by_risk_level', {})
        if by_risk:
            high_risk = by_risk.get('High Risk', {})
            if high_risk.get('total_pnl', 0) < -100:
                self.warnings.append(
                    f"⚠️ HIGH-RISK TRADERS UNDERPERFORMING: "
                    f"Average PnL = ${high_risk.get('total_pnl', 0):.2f}. "
                    f"Implement risk controls."
                )
    
    def generate_symbol_insights(self) -> None:
        """Generate insights about symbol performance."""
        
        profitability = self.analysis_results.get('profitability', {}).get('by_symbol', {})
        
        if profitability:
            symbols_sorted = sorted(
                profitability.items(),
                key=lambda x: x[1]['avg_pnl'],
                reverse=True
            )
            
            best_symbol = symbols_sorted[0]
            worst_symbol = symbols_sorted[-1]
            
            self.insights.append(
                f"Best performing symbol: {best_symbol[0]} (Avg PnL: ${best_symbol[1]['avg_pnl']:.2f}). "
                f"Allocate more capital to high-performing pairs."
            )
            
            if worst_symbol[1]['avg_pnl'] < -50:
                self.warnings.append(
                    f"⚠️ {worst_symbol[0]} showing losses (${worst_symbol[1]['avg_pnl']:.2f} avg). "
                    f"Review or suspend trading this pair."
                )
    
    def generate_volatility_insights(self) -> None:
        """Generate insights about volatility and market conditions."""
        
        volatility = self.analysis_results.get('volatility', {})
        pnl_vol = volatility.get('pnl_volatility', 0)
        
        if pnl_vol > 500:
            self.insights.append(
                f"High PnL volatility (StdDev: ${pnl_vol:.2f}) indicates inconsistent trading results. "
                f"Consider position sizing adjustments."
            )
        else:
            self.insights.append(
                f"Stable PnL results (StdDev: ${pnl_vol:.2f}) indicate consistent trading execution."
            )
    
    def generate_liquidation_insights(self) -> None:
        """Generate insights about liquidation risks."""
        
        liquidation = self.analysis_results.get('liquidation_risk', {})
        liquidation_rate = liquidation.get('liquidation_rate', 0)
        high_risk_traders = liquidation.get('high_risk_traders', 0)
        
        if liquidation_rate > 5:
            self.warnings.append(
                f"⚠️ LIQUIDATION RISK: {liquidation_rate:.2f}% of trades were liquidated. "
                f"This is unsustainably high."
            )
            self.recommendations.append(
                "Implement strict stop-loss orders at 2-3% account equity. "
                "Use automated risk management to prevent catastrophic losses."
            )
        
        if high_risk_traders > 0:
            self.insights.append(
                f"{high_risk_traders} traders classified as 'High Risk'. "
                f"Monitor their activity closely and enforce position limits."
            )
    
    def generate_winning_strategies(self) -> None:
        """Identify winning strategies and patterns."""
        
        # Identify winning combinations
        win_data = self.df[self.df['win_loss_flag'] == 1]
        
        if len(win_data) > 0:
            # Best leverage for wins
            best_leverage_cat = win_data['leverage_category'].mode().values
            if len(best_leverage_cat) > 0:
                self.recommendations.append(
                    f"Winning trades most frequently use {best_leverage_cat[0]} leverage. "
                    f"Standardize on this leverage level for new traders."
                )
            
            # Best sentiment for wins
            best_sentiment = win_data['sentiment'].mode().values
            if len(best_sentiment) > 0:
                sentiment_label = "Fear" if best_sentiment[0] == 0 else "Greed"
                self.recommendations.append(
                    f"Most wins occur during {sentiment_label} periods. "
                    f"Adjust trading strategy based on sentiment regime."
                )
    
    def generate_risk_recommendations(self) -> None:
        """Generate risk management recommendations."""
        
        avg_leverage = self.df['leverage'].mean()
        
        if avg_leverage > 5:
            self.recommendations.append(
                f"Average leverage is {avg_leverage:.2f}x - REDUCE to 3-4x for sustainability. "
                f"Lower leverage reduces liquidation risk significantly."
            )
        
        # Position sizing
        avg_position_pct = (self.df['position_value'].sum() / 
                           (self.df['position_value'].sum() + 10000)) * 100
        
        if avg_position_pct > 50:
            self.recommendations.append(
                "Position sizes appear too concentrated. Diversify across more symbols "
                "and reduce per-trade exposure to 2-5% of account equity."
            )
    
    def generate_all_insights(self) -> Dict:
        """
        Generate all insights and recommendations.
        
        Returns:
            Dictionary with organized insights
        """
        
        self.generate_sentiment_insights()
        self.generate_leverage_insights()
        self.generate_trader_insights()
        self.generate_symbol_insights()
        self.generate_volatility_insights()
        self.generate_liquidation_insights()
        self.generate_winning_strategies()
        self.generate_risk_recommendations()
        
        results = {
            'insights': self.insights,
            'recommendations': self.recommendations,
            'warnings': self.warnings
        }
        
        logger.info(f"Generated {len(self.insights)} insights, "
                   f"{len(self.recommendations)} recommendations, "
                   f"{len(self.warnings)} warnings")
        
        return results


def generate_insights(df: pd.DataFrame, analysis_results: Dict) -> Dict:
    """
    Generate all AI insights.
    
    Args:
        df: Input dataframe
        analysis_results: Results from analytics
        
    Returns:
        Dictionary with all insights
    """
    engine = InsightsEngine(df, analysis_results)
    return engine.generate_all_insights()
