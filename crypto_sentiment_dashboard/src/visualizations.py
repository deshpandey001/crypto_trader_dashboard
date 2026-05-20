"""
Visualizations module for interactive and static charts.
Creates professional trading analytics visualizations.
"""

import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger("crypto_sentiment_dashboard")

# Set style
sns.set_style("darkgrid")
plt.style.use('dark_background')


class TradingVisualizer:
    """Creates professional trading visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize visualizer.
        
        Args:
            df: Input dataframe with engineered features
        """
        self.df = df
        self.output_dir = Path("outputs/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("TradingVisualizer initialized")
    
    def create_sentiment_distribution(self) -> go.Figure:
        """Create sentiment distribution visualization."""
        
        sentiment_counts = self.df['sentiment'].value_counts()
        labels = ['Fear' if i == 0 else 'Greed' for i in sentiment_counts.index]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=sentiment_counts.values,
                hole=0.3,
                marker=dict(
                    colors=['#FF6B6B', '#4ECDC4'],
                    line=dict(color='#2C3E50', width=2)
                ),
                textposition='inside',
                textinfo='label+percent'
            )
        ])
        
        fig.update_layout(
            title="Market Sentiment Distribution",
            font=dict(size=12, family="Arial, sans-serif"),
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(25,25,25,1)',
            font_color='#FFFFFF'
        )
        
        return fig
    
    def create_pnl_distribution(self) -> go.Figure:
        """Create PnL distribution visualization."""
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=self.df['closedpnl'],
            nbinsx=50,
            name='PnL Distribution',
            marker=dict(color='#3498DB', line=dict(color='#2C3E50', width=1)),
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Closed PnL Distribution",
            xaxis_title="PnL ($)",
            yaxis_title="Frequency",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(25,25,25,1)',
            font_color='#FFFFFF',
            hovermode='x unified'
        )
        
        return fig
    
    def create_sentiment_vs_profitability(self) -> go.Figure:
        """Create sentiment vs profitability comparison."""
        
        sentiment_pnl = self.df.groupby('sentiment')['closedpnl'].agg(['mean', 'count'])
        sentiment_pnl['sentiment_label'] = ['Fear', 'Greed']
        
        fig = go.Figure(data=[
            go.Bar(
                x=sentiment_pnl['sentiment_label'],
                y=sentiment_pnl['mean'],
                marker=dict(
                    color=['#FF6B6B', '#4ECDC4'],
                    line=dict(color='#2C3E50', width=2)
                ),
                text=sentiment_pnl['mean'].round(2),
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Average PnL by Market Sentiment",
            xaxis_title="Sentiment",
            yaxis_title="Average PnL ($)",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(25,25,25,1)',
            font_color='#FFFFFF'
        )
        
        return fig
    
    def create_long_vs_short_performance(self) -> go.Figure:
        """Create long vs short performance comparison."""
        
        side_performance = self.df.groupby('side')['closedpnl'].agg(['mean', 'sum', 'count'])
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'pie'}, {'type': 'bar'}]],
            subplot_titles=('Trade Count', 'Average PnL')
        )
        
        fig.add_trace(
            go.Pie(
                labels=side_performance.index,
                values=side_performance['count'],
                marker=dict(colors=['#E74C3C', '#2ECC71']),
                name='Count'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=side_performance.index,
                y=side_performance['mean'],
                marker=dict(color=['#E74C3C', '#2ECC71']),
                text=side_performance['mean'].round(2),
                textposition='outside',
                name='Avg PnL'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Long vs Short Performance",
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(25,25,25,1)',
            font_color='#FFFFFF'
        )
        
        return fig
    
    def create_correlation_heatmap(self) -> None:
        """Create correlation heatmap and save."""
        
        numeric_cols = [
            'execution_price', 'size', 'leverage', 'closedpnl',
            'pnl_ratio', 'leverage_risk_score', 'win_loss_flag',
            'sentiment', 'trader_consistency_score'
        ]
        
        corr_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Correlation Matrix - Trading Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'correlation_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        
        logger.info(f"Saved: {output_path}")
    
    def create_leverage_vs_profit(self) -> go.Figure:
        """Create leverage vs profit scatterplot."""
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.df['leverage'],
            y=self.df['closedpnl'],
            mode='markers',
            marker=dict(
                size=5,
                color=self.df['win_loss_flag'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Win/Loss")
            ),
            text=self.df['account'],
            hovertemplate='<b>%{text}</b><br>Leverage: %{x}<br>PnL: $%{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Leverage vs Profit/Loss",
            xaxis_title="Leverage",
            yaxis_title="Closed PnL ($)",
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(25,25,25,1)',
            font_color='#FFFFFF'
        )
        
        return fig
    
    def create_trader_performance_ranking(self) -> go.Figure:
        """Create top traders ranking."""
        
        trader_stats = self.df.groupby('account')['closedpnl'].sum().nlargest(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=trader_stats.values,
                y=trader_stats.index,
                orientation='h',
                marker=dict(
                    color=trader_stats.values,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Total PnL")
                ),
                text=trader_stats.values.round(2),
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Top 10 Traders by Total PnL",
            xaxis_title="Total PnL ($)",
            yaxis_title="Trader",
            height=600,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(25,25,25,1)',
            font_color='#FFFFFF'
        )
        
        return fig
    
    def create_daily_volume_trends(self) -> go.Figure:
        """Create daily trading volume trends."""
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        daily_volume = self.df.groupby('date').size()
        daily_pnl = self.df.groupby('date')['closedpnl'].sum()
        
        fig = make_subplots(
            rows=2, cols=1,
            specs=[[{'secondary_y': False}], [{'secondary_y': False}]],
            subplot_titles=('Daily Trade Volume', 'Daily PnL'),
            vertical_spacing=0.12
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_volume.index,
                y=daily_volume.values,
                name='Trade Count',
                line=dict(color='#3498DB', width=2),
                fill='tozeroy'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_pnl.index,
                y=daily_pnl.values,
                name='Daily PnL',
                line=dict(color='#2ECC71', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="PnL ($)", row=2, col=1)
        
        fig.update_layout(
            title_text="Trading Trends Over Time",
            height=700,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(25,25,25,1)',
            font_color='#FFFFFF'
        )
        
        return fig
    
    def create_symbol_profitability(self) -> go.Figure:
        """Create symbol-wise profitability visualization."""
        
        symbol_stats = self.df.groupby('symbol')['closedpnl'].agg(['sum', 'mean', 'count'])
        symbol_stats = symbol_stats.sort_values('sum', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=symbol_stats['sum'],
            y=symbol_stats.index,
            orientation='h',
            marker=dict(
                color=symbol_stats['sum'],
                colorscale='RdYlGn',
                showscale=True
            ),
            text=symbol_stats['sum'].round(2),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Symbol-wise Total PnL",
            xaxis_title="Total PnL ($)",
            yaxis_title="Symbol",
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(25,25,25,1)',
            font_color='#FFFFFF'
        )
        
        return fig
    
    def create_risk_category_analysis(self) -> go.Figure:
        """Create risk category analysis."""
        
        risk_data = self.df.groupby('risk_level')['closedpnl'].agg(['mean', 'sum', 'count'])
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'bar'}, {'type': 'pie'}]],
            subplot_titles=('Avg PnL', 'Trade Distribution')
        )
        
        fig.add_trace(
            go.Bar(
                x=risk_data.index,
                y=risk_data['mean'],
                marker=dict(color=['#2ECC71', '#F39C12', '#E74C3C']),
                text=risk_data['mean'].round(2),
                textposition='outside'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Pie(
                labels=risk_data.index,
                values=risk_data['count'],
                marker=dict(colors=['#2ECC71', '#F39C12', '#E74C3C'])
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Analysis by Risk Level",
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(25,25,25,1)',
            font_color='#FFFFFF'
        )
        
        return fig
    
    def create_win_rate_comparison(self) -> go.Figure:
        """Create win rate comparison across categories."""
        
        categories = ['leverage_category', 'trader_type', 'risk_level']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[cat.replace('_', ' ').title() for cat in categories],
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
        )
        
        colors_list = ['#2ECC71', '#F39C12', '#E74C3C']
        
        for idx, cat in enumerate(categories, 1):
            win_rates = (self.df.groupby(cat)['win_loss_flag'].sum() / 
                        self.df.groupby(cat)['win_loss_flag'].count() * 100)
            
            fig.add_trace(
                go.Bar(
                    x=win_rates.index,
                    y=win_rates.values,
                    marker=dict(color=colors_list),
                    text=win_rates.round(1).astype(str) + '%',
                    textposition='outside',
                    showlegend=False
                ),
                row=1, col=idx
            )
            
            fig.update_yaxes(title_text="Win Rate (%)", row=1, col=idx)
        
        fig.update_layout(
            title_text="Win Rate Comparison Across Categories",
            height=500,
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(25,25,25,1)',
            font_color='#FFFFFF'
        )
        
        return fig
    
    def create_all_visualizations(self) -> dict:
        """Create and save all visualizations."""
        
        visualizations = {}
        
        # Interactive Plotly charts
        charts = {
            'sentiment_distribution': self.create_sentiment_distribution(),
            'pnl_distribution': self.create_pnl_distribution(),
            'sentiment_vs_profitability': self.create_sentiment_vs_profitability(),
            'long_vs_short': self.create_long_vs_short_performance(),
            'leverage_vs_profit': self.create_leverage_vs_profit(),
            'trader_ranking': self.create_trader_performance_ranking(),
            'daily_trends': self.create_daily_volume_trends(),
            'symbol_profitability': self.create_symbol_profitability(),
            'risk_analysis': self.create_risk_category_analysis(),
            'win_rate_comparison': self.create_win_rate_comparison()
        }
        
        for name, fig in charts.items():
            visualizations[name] = fig
            output_path = self.output_dir / f'{name}.html'
            fig.write_html(output_path)
            logger.info(f"Saved: {output_path}")
        
        # Static matplotlib charts
        self.create_correlation_heatmap()
        
        return visualizations


def create_all_visualizations(df: pd.DataFrame) -> dict:
    """
    Create all visualizations.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with all visualizations
    """
    visualizer = TradingVisualizer(df)
    return visualizer.create_all_visualizations()
