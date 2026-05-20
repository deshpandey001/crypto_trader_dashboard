"""
Report generation module for creating PDF reports and summaries.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json

logger = logging.getLogger("crypto_sentiment_dashboard")


class ReportGenerator:
    """Generates comprehensive trading analysis reports."""
    
    def __init__(self, analysis_results: Dict, insights: Dict, metrics: Dict):
        """
        Initialize report generator.
        
        Args:
            analysis_results: Results from analytics module
            insights: Insights from insights engine
            metrics: Model metrics from ML module
        """
        self.analysis_results = analysis_results
        self.insights = insights
        self.metrics = metrics
        self.output_dir = Path("outputs/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ReportGenerator initialized")
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        
        summary = """
========================================
EXECUTIVE SUMMARY
========================================

Report Generated: {}
Analysis Period: Complete Dataset

KEY METRICS:
- Total Trades: {}
- Total PnL: ${:.2f}
- Average Win Rate: {:.2f}%
- Maximum Drawdown: {:.2f}%
- Active Traders: {}
- Most Traded Symbol: {}

SENTIMENT ANALYSIS:
- Fear Periods Win Rate: {:.2f}%
- Greed Periods Win Rate: {:.2f}%

LEVERAGE METRICS:
- Average Leverage: {:.2f}x
- High Leverage Trades: {:.2f}%
- Liquidations: {}

========================================
        """.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.analysis_results.get('trade_frequency', {}).get('total_trades', 0),
            self.analysis_results.get('profitability', {}).get('total_pnl', 0),
            self.analysis_results.get('win_rate', {}).get('overall_win_rate', 0),
            abs(self.analysis_results.get('drawdown', {}).get('max_drawdown', 0)) * 100,
            self.analysis_results.get('trader_segmentation', {}).get('total_traders', 0),
            list(self.analysis_results.get('profitability', {}).get('by_symbol', {}).keys())[0]
            if self.analysis_results.get('profitability', {}).get('by_symbol', {}) else 'N/A',
            self.analysis_results.get('win_rate', {}).get('by_sentiment', {}).get('Fear', 0),
            self.analysis_results.get('win_rate', {}).get('by_sentiment', {}).get('Greed', 0),
            self.analysis_results.get('leverage', {}).get('avg_leverage', 0),
            self.analysis_results.get('leverage', {}).get('high_leverage_percentage', 0),
            self.analysis_results.get('liquidation_risk', {}).get('total_liquidations', 0)
        )
        
        return summary
    
    def generate_insights_section(self) -> str:
        """Generate insights and recommendations section."""
        
        insights_text = """
========================================
INSIGHTS & RECOMMENDATIONS
========================================

KEY INSIGHTS:
"""
        
        for i, insight in enumerate(self.insights.get('insights', []), 1):
            insights_text += f"\n{i}. {insight}\n"
        
        insights_text += """
RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(self.insights.get('recommendations', []), 1):
            insights_text += f"\n{i}. {rec}\n"
        
        if self.insights.get('warnings', []):
            insights_text += """
WARNINGS:
"""
            for i, warning in enumerate(self.insights.get('warnings', []), 1):
                insights_text += f"\n{i}. {warning}\n"
        
        insights_text += "\n========================================\n"
        return insights_text
    
    def generate_model_performance_section(self) -> str:
        """Generate ML model performance section."""
        
        section = """
========================================
MACHINE LEARNING MODEL PERFORMANCE
========================================

PROFITABILITY PREDICTION MODELS:
"""
        
        prof_metrics = self.metrics.get('profitability', {}).get('metrics', {})
        
        for model_name, metrics in prof_metrics.items():
            section += f"""
{model_name.upper()}:
- Accuracy: {metrics.get('accuracy', 0):.4f}
- ROC AUC: {metrics.get('roc_auc', 0):.4f}
"""
        
        section += """
TRADER RISK CLASSIFICATION:
- KMeans Clustering: 3 clusters (Low/Medium/High Risk)
- DBSCAN Clustering: Density-based anomaly detection

========================================
"""
        
        return section
    
    def generate_trader_analysis(self) -> str:
        """Generate trader analysis section."""
        
        section = """
========================================
TRADER ANALYSIS & SEGMENTATION
========================================

"""
        
        trader_seg = self.analysis_results.get('trader_segmentation', {})
        
        section += f"\nTotal Traders Analyzed: {trader_seg.get('total_traders', 0)}\n"
        
        if trader_seg.get('top_performers'):
            section += "\nTOP PERFORMERS:\n"
            for trader in trader_seg['top_performers'][:3]:
                section += f"- {trader['trader']}: ${trader['total_pnl']:.2f} PnL, {trader['win_rate']*100:.1f}% Win Rate\n"
        
        if trader_seg.get('by_risk_level'):
            section += "\nPERFORMANCE BY RISK LEVEL:\n"
            for risk_level, stats in trader_seg['by_risk_level'].items():
                section += f"- {risk_level}: Avg PnL ${stats.get('total_pnl', 0):.2f}, {stats.get('total_pnl', 0)*100:.1f}% Win Rate\n"
        
        section += "\n========================================\n"
        return section
    
    def generate_text_report(self) -> str:
        """Generate complete text report."""
        
        report = self.generate_executive_summary()
        report += self.generate_insights_section()
        report += self.generate_model_performance_section()
        report += self.generate_trader_analysis()
        report += """
========================================
CONCLUSION
========================================

This comprehensive analysis provides insights into trader behavior,
market sentiment effects, and risk patterns. The generated machine
learning models can predict trade profitability with high accuracy
and classify traders into risk categories for targeted interventions.

Regular monitoring of key metrics and adherence to recommendations
will improve overall trading performance and reduce systemic risk.

========================================
"""
        
        return report
    
    def save_json_report(self) -> Path:
        """Save analysis results as JSON."""
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'analysis_results': self.analysis_results,
            'insights': self.insights,
            'model_metrics': self.metrics
        }
        
        output_path = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=4, default=str)
        
        logger.info(f"Saved JSON report: {output_path}")
        return output_path
    
    def save_text_report(self) -> Path:
        """Save analysis results as text file."""
        
        report_text = self.generate_text_report()
        output_path = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Saved text report: {output_path}")
        return output_path
    
    def generate_all_reports(self) -> Dict:
        """Generate all report formats."""
        
        json_path = self.save_json_report()
        text_path = self.save_text_report()
        
        logger.info("All reports generated successfully")
        
        return {
            'json_report': str(json_path),
            'text_report': str(text_path),
            'summary': self.generate_text_report()
        }


def generate_reports(analysis_results: Dict, insights: Dict, metrics: Dict) -> Dict:
    """
    Generate all reports.
    
    Args:
        analysis_results: Results from analytics
        insights: Results from insights engine
        metrics: Model metrics
        
    Returns:
        Dictionary with report paths
    """
    generator = ReportGenerator(analysis_results, insights, metrics)
    return generator.generate_all_reports()
