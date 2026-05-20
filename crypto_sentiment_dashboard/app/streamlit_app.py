"""
AI-Powered Crypto Trader Behavior Intelligence Dashboard
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import preprocess_pipeline
from feature_engineering import engineer_features
from analysis import perform_analytics
from visualizations import create_all_visualizations
from insights_engine import generate_insights
from report_generator import generate_reports
from utils import setup_logging, create_output_directories, format_large_number

# Setup logging
logger = setup_logging()
create_output_directories()

# Page config
st.set_page_config(
    page_title="Crypto Trader Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for Rich UI
def load_custom_css():
    """Load enhanced custom CSS styling."""
    custom_css = """
    <style>
        /* Hide sidebar completely */
        [data-testid="stSidebar"] {
            display: none;
        }
        
        /* Main page background */
        .main {
            background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%);
        }
        
        /* Text styling */
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 700 !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        }
        
        /* Metrics card styling */
        [data-testid="metric.container"] {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            padding: 20px !important;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3) !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] button {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
            border: 1px solid #334155 !important;
            border-radius: 8px 8px 0 0 !important;
            color: #94a3b8 !important;
            margin-right: 8px !important;
            font-weight: 600 !important;
            padding: 12px 20px !important;
            transition: all 0.3s ease !important;
        }
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: #ffffff !important;
            border: 1px solid #1e40af !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        }
        
        .stTabs [data-baseweb="tab-list"] button:hover {
            background: linear-gradient(135deg, #334155 0%, #1e293b 100%) !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4) !important;
            transform: translateY(-2px) !important;
        }
        
        /* Divider styling */
        hr {
            border: none !important;
            height: 1px !important;
            background: linear-gradient(90deg, transparent, #334155, transparent) !important;
            margin: 20px 0 !important;
        }
        
        /* Info box styling */
        .stAlert {
            background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%) !important;
            border-left: 4px solid #3b82f6 !important;
            border-radius: 8px !important;
        }
        
        /* Text input styling */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
            color: #e2e8f0 !important;
            padding: 10px 12px !important;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 8px !important;
            border: 1px solid #334155 !important;
            overflow: hidden !important;
        }
        
        .stDataFrame > div > div > table {
            background: #0f1419 !important;
        }
        
        /* Success/Error message styling */
        .stSuccess {
            background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%) !important;
            border-left: 4px solid #10b981 !important;
            border-radius: 8px !important;
        }
        
        .stError {
            background: linear-gradient(135deg, #7c2d12 0%, #0f172a 100%) !important;
            border-left: 4px solid #ef4444 !important;
            border-radius: 8px !important;
        }
        
        /* Container styling */
        .stContainer {
            background: transparent !important;
        }
        
        /* Column spacing */
        [data-testid="column"] {
            padding: 0 8px !important;
        }
        
        /* Metric label */
        .stMetric label {
            color: #94a3b8 !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }
        
        /* Metric value */
        .stMetric [data-testid="stMetricValue"] {
            color: #60a5fa !important;
            font-size: 24px !important;
            font-weight: 700 !important;
        }
        
        /* Overall font */
        body {
            color: #e2e8f0 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        }
        
        p, span, a {
            color: #cbd5e1 !important;
        }
        
        /* Header gradient */
        header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
            border-bottom: 1px solid #334155 !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

load_custom_css()

# Session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False


@st.cache_data
def load_and_process_data():
    """Load and process data pipeline."""
    
    with st.spinner("Loading and processing data..."):
        try:
            # Preprocessing - resolve paths relative to project root
            project_root = Path(__file__).parent.parent
            fear_greed_path = str(project_root / "data" / "fear_greed_index.csv")
            trading_data_path = str(project_root / "data" / "historical_data.csv")
            
            df = preprocess_pipeline(fear_greed_path, trading_data_path)
            
            # Feature engineering
            df, feature_summary = engineer_features(df)
            
            # Analytics
            analysis_results, df = perform_analytics(df)
            
            logger.info("Data loading and processing completed")
            
            return df, analysis_results, feature_summary
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            logger.error(f"Data processing error: {str(e)}")
            return None, None, None


def train_ml_models(df):
    """Train machine learning models."""
    st.error("ML model training has been removed.")
    return None


@st.cache_resource
def get_or_train_manager(df_bytes: bytes, fast: bool = True):
    """This function has been removed as ML model training is no longer supported."""
    return None


def create_dashboard_header():
    """Create professional dashboard header with rich styling."""
    
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    '>
        <h1 style='
            color: white;
            margin: 0;
            font-size: 2.5em;
            font-weight: 800;
            text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        '>📊 AI-Powered Crypto Trader Intelligence Dashboard</h1>
        <p style='
            color: rgba(255,255,255,0.9);
            margin: 10px 0 0 0;
            font-size: 1.1em;
        '>Real-time sentiment analysis & trader behavior insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Timestamp row
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col3:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 10px 15px;
            border-radius: 8px;
            border: 1px solid #334155;
            text-align: right;
        '>
            <span style='color: #94a3b8; font-size: 0.9em;'>Last Updated</span><br/>
            <span style='color: #60a5fa; font-weight: 600;'>{timestamp}</span>
        </div>
        """, unsafe_allow_html=True)


def display_executive_kpis(analysis_results):
    """Display executive KPIs with rich styling."""
    
    st.markdown("""
    <div style='margin-bottom: 20px;'>
        <h3 style='color: #60a5fa; margin-bottom: 15px;'>📈 Executive Overview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
    
    with col1:
        total_trades = analysis_results.get('trade_frequency', {}).get('total_trades', 0)
        st.metric("🎯 Total Trades", f"{total_trades:,.0f}", "All Time", 
                 help="Total number of trades in the dataset")
    
    with col2:
        total_pnl = analysis_results.get('profitability', {}).get('total_pnl', 0)
        delta_color = "green" if total_pnl > 0 else "red"
        st.metric("💰 Total PnL", f"${total_pnl:,.2f}", 
                 f"{'+' if total_pnl > 0 else ''}{total_pnl/1000:.1f}K",
                 help="Total Profit/Loss across all trades")
    
    with col3:
        win_rate = analysis_results.get('win_rate', {}).get('overall_win_rate', 0)
        st.metric("✅ Win Rate", f"{win_rate:.1f}%", "Success Ratio",
                 help="Percentage of profitable trades")
    
    with col4:
        avg_leverage = analysis_results.get('leverage', {}).get('avg_leverage', 0)
        st.metric("⚖️ Avg Leverage", f"{avg_leverage:.2f}x", "Risk Metric",
                 help="Average leverage used in trades")
    
    with col5:
        total_traders = analysis_results.get('trader_segmentation', {}).get('total_traders', 0)
        st.metric("👥 Active Traders", f"{total_traders}", "Unique Accounts",
                 help="Total number of unique trader accounts")


def display_sentiment_analytics(df, analysis_results):
    """Display sentiment analytics section with rich styling."""
    
    st.markdown("""
    <h3 style='color: #60a5fa; margin-bottom: 20px;'>🎯 Sentiment Analytics</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    border: 1px solid #334155; border-radius: 10px; padding: 15px;
                    text-align: center;'>
            <h4 style='color: #60a5fa; margin-top: 0;'>Sentiment Distribution</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        labels = ['Fear' if i == 0 else 'Greed' for i in sentiment_counts.index]
        
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=sentiment_counts.values,
            hole=0.4,
            marker=dict(colors=['#FF6B6B', '#4ECDC4']),
            textinfo='label+percent'
        )])
        fig.update_layout(
            showlegend=True,
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF', size=12),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    border: 1px solid #334155; border-radius: 10px; padding: 15px;'>
            <h4 style='color: #60a5fa; margin-top: 0;'>📊 Win Rate by Sentiment</h4>
        </div>
        """, unsafe_allow_html=True)
        
        win_rates = analysis_results.get('win_rate', {}).get('by_sentiment', {})
        
        if win_rates:
            for sentiment, rate in win_rates.items():
                sentiment_emoji = "😨" if sentiment == "Fear" else "🤑"
                progress = rate / 100
                st.markdown(f"""
                <div style='margin-bottom: 15px;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span style='color: #e2e8f0;'>{sentiment_emoji} {sentiment}</span>
                        <span style='color: #60a5fa; font-weight: 600;'>{rate:.1f}%</span>
                    </div>
                    <div style='background: #0f172a; border-radius: 6px; overflow: hidden; height: 8px;'>
                        <div style='background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%); 
                                    width: {progress*100}%; height: 100%;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Win rate data not available yet")
        
        if win_rates:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Fear Win Rate", f"{win_rates.get('Fear', 0):.1f}%")
            with col_b:
                st.metric("Greed Win Rate", f"{win_rates.get('Greed', 0):.1f}%")
        
        # Sentiment statistics
        profitability = analysis_results.get('profitability', {}).get('by_sentiment', {})
        if profitability:
            st.markdown("**Profitability by Sentiment**")
            for sentiment, stats in profitability.items():
                st.write(f"**{sentiment}**: ${stats['avg_pnl']:.2f} avg PnL ({stats['count']} trades)")


def display_trader_intelligence(df, analysis_results):
    """Display trader intelligence section."""
    
    st.markdown("### 👥 Trader Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Performers**")
        trader_stats = analysis_results.get('trader_segmentation', {}).get('top_performers', [])
        
        if trader_stats:
            for i, trader in enumerate(trader_stats[:5], 1):
                st.write(f"{i}. **{trader['trader']}**")
                st.write(f"   PnL: ${trader['total_pnl']:,.2f} | Win Rate: {trader['win_rate']*100:.1f}%")
        else:
            st.info("No trader data available")
    
    with col2:
        st.markdown("**Risk Classification**")
        risk_by_level = analysis_results.get('trader_segmentation', {}).get('by_risk_level', {})
        
        if risk_by_level:
            for risk_level, stats in risk_by_level.items():
                col_left, col_right = st.columns(2)
                with col_left:
                    st.write(f"**{risk_level}**")
                with col_right:
                    st.write(f"Avg PnL: ${stats.get('total_pnl', 0):.2f}")


def display_predictive_analytics(ml_results):
    """Display ML model performance."""
    
    st.markdown("### 🤖 Predictive Analytics")
    
    prof_metrics = ml_results.get('profitability', {}).get('metrics', {})
    
    if prof_metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Random Forest Model**")
            rf_metrics = prof_metrics.get('random_forest', {})
            if rf_metrics:
                st.metric("Accuracy", f"{rf_metrics.get('accuracy', 0):.4f}")
                st.metric("ROC AUC", f"{rf_metrics.get('roc_auc', 0):.4f}")
        
        with col2:
            st.markdown("**XGBoost Model**")
            xgb_metrics = prof_metrics.get('xgboost', {})
            if xgb_metrics:
                st.metric("Accuracy", f"{xgb_metrics.get('accuracy', 0):.4f}")
                st.metric("ROC AUC", f"{xgb_metrics.get('roc_auc', 0):.4f}")


def display_ai_insights(insights):
    """Display AI-generated insights."""
    
    st.markdown("### 💡 AI Insights & Recommendations")
    
    # Insights
    if insights.get('insights'):
        st.markdown("**Key Insights:**")
        for insight in insights['insights'][:3]:
            st.info(f"📌 {insight}")
    
    # Recommendations
    if insights.get('recommendations'):
        st.markdown("**Strategic Recommendations:**")
        for rec in insights['recommendations'][:3]:
            st.success(f"✅ {rec}")
    
    # Warnings
    if insights.get('warnings'):
        st.markdown("**Warnings:**")
        for warning in insights['warnings']:
            st.warning(f"⚠️ {warning}")


def main():
    """Main dashboard application."""
    
    create_dashboard_header()
    
    # Main tabs (ML Models removed)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "📈 Charts", "💡 Insights", "📥 Export"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.markdown("""
        <h3 style='color: #60a5fa; margin-bottom: 20px;'>📊 Dashboard Overview</h3>
        """, unsafe_allow_html=True)
        
        # Load data button with centered layout
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Load & Process Data", key="load_data", use_container_width=True):
                df, analysis_results, feature_summary = load_and_process_data()
                
                if df is not None:
                    st.session_state.df = df
                    st.session_state.analysis_results = analysis_results
                    st.session_state.feature_summary = feature_summary
                    st.session_state.data_loaded = True
                    st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state.data_loaded and 'analysis_results' in st.session_state:
            display_executive_kpis(st.session_state.analysis_results)
            
            st.markdown("<br><hr><br>", unsafe_allow_html=True)
            
            display_sentiment_analytics(st.session_state.df, st.session_state.analysis_results)
            
            st.markdown("<br><hr><br>", unsafe_allow_html=True)
            
            display_trader_intelligence(st.session_state.df, st.session_state.analysis_results)
        else:
            st.markdown("""
            <div style='
                background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
                border-left: 4px solid #3b82f6;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
            '>
                <p style='color: #60a5fa; font-size: 1.1em; margin: 0;'>
                    👆 Click <strong>'Load & Process Data'</strong> above to begin analysis
                </p>
                <p style='color: #94a3b8; font-size: 0.95em; margin: 10px 0 0 0;'>
                    This will load trading data and sentiment indices for analysis
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # (ML Models tab removed per request)
    
    # Tab 2: Visualizations
    with tab2:
        st.markdown("### Interactive Charts")
        
        if st.button("📊 Generate Visualizations", key="gen_charts"):
            if st.session_state.data_loaded and 'df' in st.session_state:
                with st.spinner("Generating visualizations..."):
                    try:
                        visualizations = create_all_visualizations(st.session_state.df)
                        st.session_state.visualizations = visualizations
                        st.success("✅ Visualizations generated!")
                    except Exception as e:
                        st.error(f"Error generating visualizations: {str(e)}")
            else:
                st.error("Please load data first")
        
        if 'visualizations' in st.session_state:
            visualizations = st.session_state.visualizations
            
            chart_cols = st.columns(2)
            chart_names = list(visualizations.keys())
            
            for idx, chart_name in enumerate(chart_names):
                col_idx = idx % 2
                with chart_cols[col_idx]:
                    st.markdown(f"**{chart_name.replace('_', ' ').title()}**")
                    st.plotly_chart(visualizations[chart_name], use_container_width=True)
    
    # Tab 3: AI Insights
    with tab3:
        st.markdown("### AI-Powered Insights")
        
        if st.button("💡 Generate Insights", key="gen_insights"):
            if st.session_state.data_loaded and 'analysis_results' in st.session_state:
                with st.spinner("Generating AI insights..."):
                    try:
                        insights = generate_insights(
                            st.session_state.df,
                            st.session_state.analysis_results
                        )
                        st.session_state.insights = insights
                        st.success("✅ Insights generated!")
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
            else:
                st.error("Please load data first")
        
        if 'insights' in st.session_state:
            display_ai_insights(st.session_state.insights)
    
    # Tab 4: Export
    with tab4:
        st.markdown("### 📥 Export Center")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate Report", key="gen_report"):
                if st.session_state.data_loaded:
                    with st.spinner("Generating comprehensive report..."):
                        try:
                            ml_results = st.session_state.ml_results if 'ml_results' in st.session_state else {}
                            insights = st.session_state.insights if 'insights' in st.session_state else {}
                            
                            reports = generate_reports(
                                st.session_state.analysis_results,
                                insights,
                                ml_results
                            )
                            
                            st.session_state.reports = reports
                            st.success("✅ Report generated!")
                        except Exception as e:
                            st.error(f"Error generating report: {str(e)}")
                else:
                    st.error("Please load data first")
        
        with col2:
            if st.button("💾 Export Data", key="export_data"):
                if st.session_state.data_loaded and 'df' in st.session_state:
                    csv = st.session_state.df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"trading_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("📊 Export Charts", key="export_charts"):
                st.info("Charts are automatically saved to outputs/charts/")
        
        if 'reports' in st.session_state:
            st.markdown("**Generated Reports**")
            reports = st.session_state.reports
            
            if 'summary' in reports:
                st.markdown("**Report Summary**")
                st.text(reports['summary'])


if __name__ == "__main__":
    main()
