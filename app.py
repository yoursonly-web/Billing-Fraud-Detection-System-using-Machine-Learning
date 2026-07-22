"""
Billing Fraud Detection System — Streamlit Web App
====================================================
Step 10: Deployment — Premium dark-themed dashboard
for fraud prediction and model analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="Billing Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container background */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e1a 0%, #111827 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }
    
    /* Card styling with glassmorphism */
    .metric-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-top: 8px;
    }
    
    .metric-subtitle {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 32px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 50%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #e2e8f0 0%, #818cf8 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 12px 0;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        position: relative;
        z-index: 1;
    }
    
    /* Status badges */
    .badge-fraud {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .badge-safe {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Result card */
    .result-card-fraud {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    
    .result-card-safe {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.05) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    
    .risk-score {
        font-size: 4rem;
        font-weight: 900;
        margin: 16px 0;
        line-height: 1;
    }
    
    .risk-score-fraud {
        color: #ef4444;
    }
    
    .risk-score-safe {
        color: #22c55e;
    }
    
    /* Info box */
    .info-box {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #c7d2fe;
        font-size: 0.9rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        padding: 8px 20px;
        color: #94a3b8;
        border: 1px solid rgba(99, 102, 241, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(139, 92, 246, 0.2) 100%);
        color: #e2e8f0;
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Table improvements */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Slider styling */
    .stSlider > div > div {
        color: #818cf8;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME
# ============================================================
PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(15, 23, 42, 0.5)',
        'font': {'color': '#94a3b8', 'family': 'Inter'},
        'xaxis': {
            'gridcolor': 'rgba(99, 102, 241, 0.1)',
            'zerolinecolor': 'rgba(99, 102, 241, 0.2)',
        },
        'yaxis': {
            'gridcolor': 'rgba(99, 102, 241, 0.1)',
            'zerolinecolor': 'rgba(99, 102, 241, 0.2)',
        },
        'colorway': ['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#818cf8',
                      '#22c55e', '#ef4444', '#f59e0b', '#06b6d4', '#ec4899'],
    }
}

GRADIENT_COLORS = ['#6366f1', '#8b5cf6', '#a78bfa', '#818cf8', '#c4b5fd']
FRAUD_COLORS = {'Fraud': '#ef4444', 'Non-Fraud': '#22c55e'}

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_artifacts():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    try:
        model_results = joblib.load(os.path.join(base, 'model_results.pkl'))
        eda_data = joblib.load(os.path.join(base, 'eda_data.pkl'))
        clean_data = joblib.load(os.path.join(base, 'clean_data.pkl'))
        feature_names = joblib.load(os.path.join(base, 'feature_names.pkl'))
        medians = joblib.load(os.path.join(base, 'medians.pkl'))
        best_model = joblib.load(os.path.join(base, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(base, 'scaler.pkl'))
        test_data = joblib.load(os.path.join(base, 'test_data.pkl'))
        return model_results, eda_data, clean_data, feature_names, medians, best_model, scaler, test_data
    except FileNotFoundError as e:
        st.error(f"⚠️ Model artifacts not found. Please run `python train_model.py` first.\n\nError: {e}")
        st.stop()

model_results, eda_data, clean_data, feature_names, medians, best_model, scaler, test_data = load_artifacts()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 3rem;">🛡️</div>
        <div style="font-size: 1.2rem; font-weight: 700; color: #e2e8f0; margin-top: 8px;">
            Fraud Detection
        </div>
        <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">
            ML-Powered Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 EDA & Analysis", "🤖 Model Performance", 
         "🔍 Predict Fraud", "⚙️ Threshold Tuning", "📈 Monitoring"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model info
    st.markdown(f"""
    <div class="info-box">
        <strong>📋 Model Info</strong><br>
        <span style="color: #64748b;">Best Model:</span> {eda_data['best_model_name']}<br>
        <span style="color: #64748b;">Trained:</span> {eda_data.get('training_date', 'N/A')}<br>
        <span style="color: #64748b;">Features:</span> {eda_data['n_features']}<br>
        <span style="color: #64748b;">Dataset:</span> {eda_data['total_rows_clean']} rows
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def metric_card(value, label, subtitle="", col=None):
    html = f"""
    <div class="metric-card">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
        <p class="metric-subtitle">{subtitle}</p>
    </div>
    """
    if col:
        col.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def create_plotly_figure():
    """Create a figure with the dark theme."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        font=dict(color='#94a3b8', family='Inter'),
        xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', zerolinecolor='rgba(99, 102, 241, 0.2)'),
        yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', zerolinecolor='rgba(99, 102, 241, 0.2)'),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ============================================================
# PAGE: DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    # Hero
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🛡️ Billing Fraud Detection System</h1>
        <p class="hero-subtitle">
            ML-powered fraud detection pipeline — analyzing billing patterns to catch anomalies
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    best = model_results[eda_data['best_model_name']]
    col1, col2, col3, col4 = st.columns(4)
    metric_card(f"{best['recall']*100:.1f}%", "Recall (Fraud Caught)", 
                f"Best: {eda_data['best_model_name']}", col1)
    metric_card(f"{best['precision']*100:.1f}%", "Precision", 
                "Of flagged bills, % actually fraud", col2)
    metric_card(f"{best['roc_auc']*100:.1f}%", "ROC-AUC Score", 
                "Ranking quality", col3)
    metric_card(f"{eda_data['total_rows_clean']}", "Records Analyzed",
                f"{eda_data['n_dropped_target']} dropped (missing target)", col4)
    
    st.markdown("")
    
    # Two-column layout
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown('<p class="section-header">📊 Model Comparison</p>', unsafe_allow_html=True)
        
        # Model comparison bar chart
        model_names = list(model_results.keys())
        metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig = go.Figure()
        colors = ['#6366f1', '#8b5cf6', '#22c55e', '#f59e0b', '#06b6d4']
        for i, metric in enumerate(metrics_list):
            values = [model_results[m][metric] * 100 for m in model_names]
            fig.update_layout(barmode='group')
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=model_names,
                y=values,
                marker_color=colors[i],
                text=[f"{v:.1f}%" for v in values],
                textposition='outside',
                textfont=dict(size=10),
            ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            font=dict(color='#94a3b8', family='Inter'),
            xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)'),
            yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Score (%)'),
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center',
                       font=dict(size=11)),
            height=420,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown('<p class="section-header">📈 Class Distribution</p>', unsafe_allow_html=True)
        
        fraud_n = eda_data['class_distribution']['fraud']
        non_fraud_n = eda_data['class_distribution']['non_fraud']
        
        fig = go.Figure(data=[go.Pie(
            labels=['Fraud', 'Non-Fraud'],
            values=[fraud_n, non_fraud_n],
            hole=0.65,
            marker=dict(colors=['#ef4444', '#22c55e'],
                       line=dict(color='rgba(15, 23, 42, 1)', width=3)),
            textinfo='label+percent',
            textfont=dict(size=13, color='#e2e8f0'),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
        )])
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'),
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            annotations=[dict(
                text=f"<b>{eda_data['fraud_rate']:.1f}%</b><br>Fraud Rate",
                x=0.5, y=0.5, font_size=18, showarrow=False,
                font=dict(color='#e2e8f0', family='Inter')
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown('<p class="section-header">🎯 Top Feature Importances</p>', unsafe_allow_html=True)
    
    best_importances = model_results[eda_data['best_model_name']]['feature_importances']
    top_features = dict(list(best_importances.items())[:10])
    
    fig = go.Figure(go.Bar(
        y=list(top_features.keys())[::-1],
        x=list(top_features.values())[::-1],
        orientation='h',
        marker=dict(
            color=list(top_features.values())[::-1],
            colorscale=[[0, '#6366f1'], [0.5, '#8b5cf6'], [1, '#a78bfa']],
        ),
        text=[f"{v:.4f}" for v in list(top_features.values())[::-1]],
        textposition='outside',
        textfont=dict(size=11, color='#c7d2fe'),
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        font=dict(color='#94a3b8', family='Inter'),
        xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Importance'),
        yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)'),
        height=400,
        margin=dict(l=20, r=80, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: EDA & ANALYSIS
# ============================================================
elif page == "📊 EDA & Analysis":
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">📊 Exploratory Data Analysis</h1>
        <p class="hero-subtitle">Deep dive into billing patterns, fraud signals, and statistical insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Distributions", "🔗 Correlations", "📊 Fraud Patterns", "🔬 Statistical Tests"])
    
    with tab1:
        st.markdown('<p class="section-header">Feature Distributions</p>', unsafe_allow_html=True)
        
        feature_to_plot = st.selectbox(
            "Select feature to explore",
            ['billing_amount', 'avg_last_6_months', 'payment_delay_days', 
             'meter_reading', 'previous_reading', 'location_risk_score',
             'num_complaints', 'billing_ratio', 'billing_deviation', 'usage_diff'],
            key='dist_feature'
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram by fraud status
            fig = go.Figure()
            for label, color in [('Non-Fraud', '#22c55e'), ('Fraud', '#ef4444')]:
                mask = clean_data['is_fraud'] == (1 if label == 'Fraud' else 0)
                fig.add_trace(go.Histogram(
                    x=clean_data[mask][feature_to_plot],
                    name=label,
                    marker_color=color,
                    opacity=0.7,
                    nbinsx=30,
                ))
            
            fig.update_layout(
                title=dict(text=f'{feature_to_plot} by Fraud Status', font=dict(size=14, color='#e2e8f0')),
                barmode='overlay',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.5)',
                font=dict(color='#94a3b8', family='Inter'),
                xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title=feature_to_plot),
                yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Count'),
                legend=dict(font=dict(size=11)),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            df_plot = clean_data.copy()
            df_plot['Fraud Status'] = df_plot['is_fraud'].map({0: 'Non-Fraud', 1: 'Fraud'})
            
            fig = go.Figure()
            for label, color in [('Non-Fraud', '#22c55e'), ('Fraud', '#ef4444')]:
                mask = df_plot['Fraud Status'] == label
                fig.add_trace(go.Box(
                    y=df_plot[mask][feature_to_plot],
                    name=label,
                    marker_color=color,
                    boxmean='sd',
                    line=dict(color=color),
                ))
            
            fig.update_layout(
                title=dict(text=f'{feature_to_plot} — Box Plot Comparison', font=dict(size=14, color='#e2e8f0')),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.5)',
                font=dict(color='#94a3b8', family='Inter'),
                yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title=feature_to_plot),
                xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)'),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<p class="section-header">Correlation Analysis</p>', unsafe_allow_html=True)
        
        # Correlation heatmap
        corr_cols = ['billing_amount', 'avg_last_6_months', 'payment_delay_days',
                     'meter_reading', 'previous_reading', 'location_risk_score',
                     'num_complaints', 'billing_ratio', 'billing_deviation', 'usage_diff', 'is_fraud']
        available_cols = [c for c in corr_cols if c in clean_data.columns]
        corr_matrix = clean_data[available_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=[[0, '#1e1b4b'], [0.25, '#312e81'], [0.5, '#4338ca'], 
                        [0.75, '#818cf8'], [1, '#c4b5fd']],
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=10, color='#e2e8f0'),
            hoverongaps=False,
            colorbar=dict(title='Corr', tickfont=dict(color='#94a3b8')),
        ))
        
        fig.update_layout(
            title=dict(text='Feature Correlation Matrix', font=dict(size=14, color='#e2e8f0')),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'),
            height=600,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(tickangle=45),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation with target bar chart
        st.markdown('<p class="section-header">Correlation with Fraud Label</p>', unsafe_allow_html=True)
        
        corr_with_fraud = {k: v for k, v in eda_data['correlations'].items()}
        corr_sorted = dict(sorted(corr_with_fraud.items(), key=lambda x: abs(x[1]), reverse=True))
        
        colors_bar = ['#ef4444' if v > 0 else '#22c55e' for v in corr_sorted.values()]
        
        fig = go.Figure(go.Bar(
            x=list(corr_sorted.keys()),
            y=list(corr_sorted.values()),
            marker_color=colors_bar,
            text=[f"{v:.3f}" for v in corr_sorted.values()],
            textposition='outside',
            textfont=dict(size=10, color='#c7d2fe'),
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            font=dict(color='#94a3b8', family='Inter'),
            xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', tickangle=45),
            yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Correlation Coefficient'),
            height=400,
            margin=dict(l=20, r=20, t=20, b=100),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<p class="section-header">Fraud Rate by Feature Buckets</p>', unsafe_allow_html=True)
        
        # Billing ratio bands
        col1, col2 = st.columns(2)
        
        with col1:
            if 'billing_ratio' in clean_data.columns:
                bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, float('inf')]
                labels_br = ['<0.5x', '0.5-1x', '1-1.5x', '1.5-2x', '2-3x', '3-5x', '>5x']
                clean_data_temp = clean_data.copy()
                clean_data_temp['billing_ratio_band'] = pd.cut(
                    clean_data_temp['billing_ratio'], bins=bins, labels=labels_br
                )
                fraud_by_band = clean_data_temp.groupby('billing_ratio_band', observed=False)['is_fraud'].agg(['mean', 'count'])
                fraud_by_band['mean'] = fraud_by_band['mean'] * 100
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=fraud_by_band.index.astype(str),
                    y=fraud_by_band['mean'],
                    marker=dict(
                        color=fraud_by_band['mean'],
                        colorscale=[[0, '#22c55e'], [0.5, '#f59e0b'], [1, '#ef4444']],
                    ),
                    text=[f"{v:.1f}%<br>(n={int(c)})" for v, c in zip(fraud_by_band['mean'], fraud_by_band['count'])],
                    textposition='outside',
                    textfont=dict(size=10, color='#c7d2fe'),
                ))
                
                fig.update_layout(
                    title=dict(text='Fraud Rate by Billing Ratio', font=dict(size=14, color='#e2e8f0')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15, 23, 42, 0.5)',
                    font=dict(color='#94a3b8', family='Inter'),
                    xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Billing Ratio Band'),
                    yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Fraud Rate (%)'),
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Payment delay bands
            bins_delay = [-1, 5, 10, 15, 20, 25, 30]
            labels_delay = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30']
            clean_data_temp2 = clean_data.copy()
            clean_data_temp2['delay_band'] = pd.cut(
                clean_data_temp2['payment_delay_days'], bins=bins_delay, labels=labels_delay
            )
            fraud_by_delay = clean_data_temp2.groupby('delay_band', observed=False)['is_fraud'].agg(['mean', 'count'])
            fraud_by_delay['mean'] = fraud_by_delay['mean'] * 100
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=fraud_by_delay.index.astype(str),
                y=fraud_by_delay['mean'],
                marker=dict(
                    color=fraud_by_delay['mean'],
                    colorscale=[[0, '#22c55e'], [0.5, '#f59e0b'], [1, '#ef4444']],
                ),
                text=[f"{v:.1f}%<br>(n={int(c)})" for v, c in zip(fraud_by_delay['mean'], fraud_by_delay['count'])],
                textposition='outside',
                textfont=dict(size=10, color='#c7d2fe'),
            ))
            
            fig.update_layout(
                title=dict(text='Fraud Rate by Payment Delay', font=dict(size=14, color='#e2e8f0')),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.5)',
                font=dict(color='#94a3b8', family='Inter'),
                xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Delay (Days)'),
                yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Fraud Rate (%)'),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Location risk score scatter
        st.markdown('<p class="section-header">Billing Amount vs Location Risk</p>', unsafe_allow_html=True)
        
        df_scatter = clean_data.sample(min(500, len(clean_data)), random_state=42)
        df_scatter['Fraud Status'] = df_scatter['is_fraud'].map({0: 'Non-Fraud', 1: 'Fraud'})
        
        fig = go.Figure()
        for label, color in [('Non-Fraud', '#22c55e'), ('Fraud', '#ef4444')]:
            mask = df_scatter['Fraud Status'] == label
            fig.add_trace(go.Scatter(
                x=df_scatter[mask]['location_risk_score'],
                y=df_scatter[mask]['billing_amount'],
                mode='markers',
                name=label,
                marker=dict(color=color, size=6, opacity=0.6,
                           line=dict(width=0.5, color='rgba(255,255,255,0.2)')),
            ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            font=dict(color='#94a3b8', family='Inter'),
            xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Location Risk Score'),
            yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Billing Amount'),
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(font=dict(size=12)),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown('<p class="section-header">Statistical Significance Tests</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>📖 How to read:</strong> A p-value < 0.05 means the difference between fraud and non-fraud groups 
            is statistically significant (unlikely due to random chance). This validates whether each feature 
            genuinely distinguishes fraud from legitimate bills.
        </div>
        """, unsafe_allow_html=True)
        
        # Build table
        test_rows = []
        for col, data in eda_data['eda_comparison'].items():
            test_rows.append({
                'Feature': col,
                'Fraud Mean': f"{data['fraud_mean']:.1f}",
                'Non-Fraud Mean': f"{data['non_fraud_mean']:.1f}",
                'Difference': f"{data['fraud_mean'] - data['non_fraud_mean']:.1f}",
                't-statistic': f"{data['t_statistic']:.3f}",
                'p-value': f"{data['p_value']:.6f}",
                'Significant (p<0.05)': '✅ YES' if data['significant'] == 'YES' else '❌ NO'
            })
        
        st.dataframe(
            pd.DataFrame(test_rows).set_index('Feature'),
            use_container_width=True,
            height=350
        )
        
        # Missing value analysis
        st.markdown('<p class="section-header">Missing Values Report</p>', unsafe_allow_html=True)
        
        missing_rows = []
        for col, info in eda_data['missing_report'].items():
            if info['count'] > 0:
                missing_rows.append({
                    'Column': col,
                    'Missing Count': info['count'],
                    'Missing %': f"{info['pct']}%"
                })
        
        if missing_rows:
            st.dataframe(pd.DataFrame(missing_rows).set_index('Column'), use_container_width=True)
        else:
            st.markdown("""
            <div class="info-box">
                ✅ <strong>No missing values</strong> — All columns were fully populated after cleaning and imputation.
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "🤖 Model Performance":
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🤖 Model Performance</h1>
        <p class="hero-subtitle">Detailed comparison of all trained models with evaluation metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selector
    selected_model = st.selectbox(
        "Select model to examine",
        list(model_results.keys()),
        index=list(model_results.keys()).index(eda_data['best_model_name'])
    )
    
    result = model_results[selected_model]
    is_best = selected_model == eda_data['best_model_name']
    
    if is_best:
        st.markdown("""
        <div class="info-box">
            ⭐ This is the <strong>best performing model</strong> selected by highest recall score.
        </div>
        """, unsafe_allow_html=True)
    
    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    metric_card(f"{result['accuracy']*100:.1f}%", "Accuracy", "", col1)
    metric_card(f"{result['precision']*100:.1f}%", "Precision", "", col2)
    metric_card(f"{result['recall']*100:.1f}%", "Recall", "", col3)
    metric_card(f"{result['f1_score']*100:.1f}%", "F1 Score", "", col4)
    metric_card(f"{result['roc_auc']*100:.1f}%", "ROC-AUC", "", col5)
    
    st.markdown("")
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Confusion Matrix
        st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
        
        cm = np.array(result['confusion_matrix'])
        labels = ['Non-Fraud', 'Fraud']
        
        # Annotated text
        annotations = [
            [f"TN<br>{cm[0][0]}", f"FP<br>{cm[0][1]}"],
            [f"FN<br>{cm[1][0]}", f"TP<br>{cm[1][1]}"]
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Non-Fraud', 'Predicted Fraud'],
            y=['Actual Non-Fraud', 'Actual Fraud'],
            colorscale=[[0, '#1e1b4b'], [0.5, '#4338ca'], [1, '#818cf8']],
            text=[[f"TN = {cm[0][0]}", f"FP = {cm[0][1]}"],
                  [f"FN = {cm[1][0]}", f"TP = {cm[1][1]}"]],
            texttemplate='%{text}',
            textfont=dict(size=14, color='#e2e8f0'),
            showscale=False,
            hoverongaps=False,
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'),
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(side='bottom'),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        # ROC Curve
        st.markdown('<p class="section-header">ROC Curve</p>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Plot all models' ROC curves
        colors_roc = ['#6366f1', '#8b5cf6', '#22c55e', '#f59e0b']
        for i, (name, res) in enumerate(model_results.items()):
            width = 3 if name == selected_model else 1.5
            dash = None if name == selected_model else 'dot'
            fig.add_trace(go.Scatter(
                x=res['roc_curve']['fpr'],
                y=res['roc_curve']['tpr'],
                mode='lines',
                name=f"{name} (AUC={res['roc_auc']:.3f})",
                line=dict(color=colors_roc[i % len(colors_roc)], width=width, dash=dash),
            ))
        
        # Diagonal
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(color='#475569', width=1, dash='dash'),
            name='Random', showlegend=False
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            font=dict(color='#94a3b8', family='Inter'),
            xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='False Positive Rate'),
            yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='True Positive Rate'),
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(font=dict(size=10), x=0.4, y=0.1),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Precision-Recall Curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="section-header">Precision-Recall Curve</p>', unsafe_allow_html=True)
        
        fig = go.Figure()
        for i, (name, res) in enumerate(model_results.items()):
            width = 3 if name == selected_model else 1.5
            dash = None if name == selected_model else 'dot'
            fig.add_trace(go.Scatter(
                x=res['pr_curve']['recall'],
                y=res['pr_curve']['precision'],
                mode='lines',
                name=name,
                line=dict(color=colors_roc[i % len(colors_roc)], width=width, dash=dash),
            ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            font=dict(color='#94a3b8', family='Inter'),
            xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Recall'),
            yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Precision'),
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature importances for selected model
        st.markdown('<p class="section-header">Feature Importances</p>', unsafe_allow_html=True)
        
        importances = result['feature_importances']
        top_n = dict(list(importances.items())[:10])
        
        fig = go.Figure(go.Bar(
            y=list(top_n.keys())[::-1],
            x=list(top_n.values())[::-1],
            orientation='h',
            marker=dict(
                color=list(top_n.values())[::-1],
                colorscale=[[0, '#6366f1'], [1, '#a78bfa']],
            ),
            text=[f"{v:.4f}" for v in list(top_n.values())[::-1]],
            textposition='outside',
            textfont=dict(size=10, color='#c7d2fe'),
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            font=dict(color='#94a3b8', family='Inter'),
            xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)'),
            yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)'),
            height=400,
            margin=dict(l=20, r=80, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # All models comparison table
    st.markdown('<p class="section-header">All Models — Detailed Comparison</p>', unsafe_allow_html=True)
    
    comparison_data = []
    for name, res in model_results.items():
        cm_temp = np.array(res['confusion_matrix'])
        comparison_data.append({
            'Model': f"{'⭐ ' if name == eda_data['best_model_name'] else ''}{name}",
            'Accuracy': f"{res['accuracy']*100:.2f}%",
            'Precision': f"{res['precision']*100:.2f}%",
            'Recall': f"{res['recall']*100:.2f}%",
            'F1 Score': f"{res['f1_score']*100:.2f}%",
            'ROC-AUC': f"{res['roc_auc']*100:.2f}%",
            'True Pos': cm_temp[1][1],
            'False Neg': cm_temp[1][0],
            'False Pos': cm_temp[0][1],
            'True Neg': cm_temp[0][0],
        })
    
    st.dataframe(pd.DataFrame(comparison_data).set_index('Model'), use_container_width=True)


# ============================================================
# PAGE: PREDICT FRAUD
# ============================================================
elif page == "🔍 Predict Fraud":
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🔍 Predict Billing Fraud</h1>
        <p class="hero-subtitle">Enter billing details to get a real-time fraud risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>💡 How it works:</strong> Enter the billing information below. The model will analyze the patterns 
        and return a risk score (0–100%) along with a fraud/safe classification based on the selected threshold.
    </div>
    """, unsafe_allow_html=True)
    
    # Threshold selector
    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Bills with risk score above this threshold will be flagged as fraud")
    
    st.markdown("")
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        billing_amount = st.number_input("💰 Billing Amount (₹)", min_value=0.0, max_value=20000.0, 
                                         value=5000.0, step=100.0, key='ba')
        avg_last_6 = st.number_input("📊 Avg Last 6 Months (₹)", min_value=0.0, max_value=10000.0,
                                      value=2500.0, step=100.0, key='avg6')
        payment_delay = st.number_input("⏰ Payment Delay (Days)", min_value=0, max_value=30,
                                         value=10, step=1, key='pd')
    
    with col2:
        meter_reading = st.number_input("🔢 Meter Reading", min_value=0.0, max_value=20000.0,
                                         value=10000.0, step=100.0, key='mr')
        prev_reading = st.number_input("📉 Previous Reading", min_value=0.0, max_value=20000.0,
                                        value=8000.0, step=100.0, key='pr')
    
    with col3:
        location_risk = st.slider("📍 Location Risk Score", 0.0, 1.0, 0.5, 0.01, key='lr')
        num_complaints = st.number_input("📝 Number of Complaints", min_value=0, max_value=10,
                                          value=2, step=1, key='nc')
    
    st.markdown("")
    
    if st.button("🔍  Analyze for Fraud", use_container_width=True, type="primary"):
        # Feature engineering for the input
        billing_ratio = billing_amount / avg_last_6 if avg_last_6 > 0 else billing_amount / 1
        billing_ratio = min(billing_ratio, 50)
        billing_deviation = billing_amount - avg_last_6
        usage_diff = meter_reading - prev_reading
        delay_risk_interaction = payment_delay * location_risk
        billing_per_unit = billing_amount / usage_diff if usage_diff != 0 else 0
        billing_per_unit = max(min(billing_per_unit, 100), -100)
        high_billing_flag = 1 if billing_ratio > 2 else 0
        
        if payment_delay <= 7:
            payment_delay_cat = 0
        elif payment_delay <= 14:
            payment_delay_cat = 1
        elif payment_delay <= 21:
            payment_delay_cat = 2
        else:
            payment_delay_cat = 3
        
        # Build feature vector (same order as training)
        feature_values = {
            'billing_amount': billing_amount,
            'avg_last_6_months': avg_last_6,
            'payment_delay_days': payment_delay,
            'meter_reading': meter_reading,
            'previous_reading': prev_reading,
            'location_risk_score': location_risk,
            'num_complaints': num_complaints,
            'billing_ratio': billing_ratio,
            'billing_deviation': billing_deviation,
            'usage_diff': usage_diff,
            'delay_risk_interaction': delay_risk_interaction,
            'billing_per_unit': billing_per_unit,
            'high_billing_flag': high_billing_flag,
            'payment_delay_category': payment_delay_cat,
        }
        
        # Add missing flags (all 0 since user provided all values)
        missing_flag_cols = [f'{col}_missing' for col in 
                            ['billing_amount', 'avg_last_6_months', 'payment_delay_days',
                             'meter_reading', 'previous_reading', 'location_risk_score', 'num_complaints']]
        for col in missing_flag_cols:
            feature_values[col] = 0
        
        # Create DataFrame in correct feature order
        input_df = pd.DataFrame([feature_values])[feature_names]
        
        # Determine if best model needs scaling
        best_model_name = eda_data['best_model_name']
        if best_model_name == 'Logistic Regression':
            input_scaled = scaler.transform(input_df)
            proba = best_model.predict_proba(input_scaled)[0][1]
        else:
            proba = best_model.predict_proba(input_df)[0][1]
        
        is_fraud_pred = proba >= threshold
        risk_pct = proba * 100
        
        st.markdown("")
        
        # Result display
        col_r1, col_r2 = st.columns([1, 1])
        
        with col_r1:
            if is_fraud_pred:
                st.markdown(f"""
                <div class="result-card-fraud">
                    <span class="badge-fraud">⚠️ FRAUD DETECTED</span>
                    <p class="risk-score risk-score-fraud">{risk_pct:.1f}%</p>
                    <p style="color: #fca5a5; font-size: 1rem;">Risk Score</p>
                    <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 12px;">
                        This bill exceeds the {threshold*100:.0f}% fraud threshold.<br>
                        Recommend investigation by the audit team.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-safe">
                    <span class="badge-safe">✅ APPEARS SAFE</span>
                    <p class="risk-score risk-score-safe">{risk_pct:.1f}%</p>
                    <p style="color: #86efac; font-size: 1rem;">Risk Score</p>
                    <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 12px;">
                        This bill is below the {threshold*100:.0f}% fraud threshold.<br>
                        No immediate investigation needed.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_r2:
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                title={'text': "Fraud Risk Score", 'font': {'size': 16, 'color': '#e2e8f0'}},
                number={'suffix': '%', 'font': {'size': 36, 'color': '#e2e8f0'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#64748b'},
                    'bar': {'color': '#6366f1'},
                    'bgcolor': 'rgba(15, 23, 42, 0.5)',
                    'bordercolor': 'rgba(99, 102, 241, 0.3)',
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.2)'},
                        {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.2)'},
                    ],
                    'threshold': {
                        'line': {'color': '#ef4444', 'width': 3},
                        'thickness': 0.8,
                        'value': threshold * 100
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', family='Inter'),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Key risk factors
        st.markdown('<p class="section-header">🔎 Key Risk Factors</p>', unsafe_allow_html=True)
        
        risk_factors = []
        if billing_ratio > 2:
            risk_factors.append(("🔴", f"Billing ratio is {billing_ratio:.1f}x — bill is {billing_ratio:.1f}× the 6-month average"))
        elif billing_ratio > 1.5:
            risk_factors.append(("🟡", f"Billing ratio is {billing_ratio:.1f}x — moderately above average"))
        else:
            risk_factors.append(("🟢", f"Billing ratio is {billing_ratio:.1f}x — within normal range"))
        
        if payment_delay > 20:
            risk_factors.append(("🔴", f"Payment delay is {payment_delay} days — significantly late"))
        elif payment_delay > 14:
            risk_factors.append(("🟡", f"Payment delay is {payment_delay} days — moderately late"))
        else:
            risk_factors.append(("🟢", f"Payment delay is {payment_delay} days — acceptable"))
        
        if location_risk > 0.7:
            risk_factors.append(("🔴", f"Location risk score is {location_risk:.2f} — high-risk area"))
        elif location_risk > 0.4:
            risk_factors.append(("🟡", f"Location risk score is {location_risk:.2f} — moderate-risk area"))
        else:
            risk_factors.append(("🟢", f"Location risk score is {location_risk:.2f} — low-risk area"))
        
        if billing_deviation > 3000:
            risk_factors.append(("🔴", f"Bill is ₹{billing_deviation:.0f} above the 6-month average"))
        elif billing_deviation > 1000:
            risk_factors.append(("🟡", f"Bill is ₹{billing_deviation:.0f} above the 6-month average"))
        else:
            risk_factors.append(("🟢", f"Bill difference from average: ₹{billing_deviation:.0f}"))
        
        for icon, text in risk_factors:
            st.markdown(f"{icon} {text}")


# ============================================================
# PAGE: THRESHOLD TUNING
# ============================================================
elif page == "⚙️ Threshold Tuning":
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">⚙️ Threshold Tuning</h1>
        <p class="hero-subtitle">Adjust the decision boundary to balance precision and recall for your use case</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>📖 Understanding the trade-off:</strong><br>
        • <strong>Lower threshold</strong> → catches more fraud (higher recall) but more false alarms (lower precision)<br>
        • <strong>Higher threshold</strong> → fewer false alarms (higher precision) but misses more fraud (lower recall)<br>
        • The <strong>optimal threshold</strong> depends on your team's capacity and the cost of missed fraud vs false alarms.
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive threshold slider
    thresh_val = st.slider("Adjust Threshold", 0.10, 0.90, 0.50, 0.05, key='thresh_tune')
    
    # Calculate metrics at this threshold using best model's probabilities
    y_test = test_data['y_test']
    best_proba = np.array(model_results[eda_data['best_model_name']]['y_proba'])
    y_pred_t = (best_proba >= thresh_val).astype(int)
    
    prec_t = precision_score(y_test, y_pred_t, zero_division=0)
    rec_t = recall_score(y_test, y_pred_t, zero_division=0)
    f1_t = f1_score(y_test, y_pred_t, zero_division=0)
    cm_t = confusion_matrix(y_test, y_pred_t)
    
    # Display metrics at current threshold
    col1, col2, col3, col4 = st.columns(4)
    metric_card(f"{prec_t*100:.1f}%", "Precision", f"At threshold {thresh_val:.2f}", col1)
    metric_card(f"{rec_t*100:.1f}%", "Recall", f"At threshold {thresh_val:.2f}", col2)
    metric_card(f"{f1_t*100:.1f}%", "F1 Score", f"At threshold {thresh_val:.2f}", col3)
    
    if len(cm_t) == 2:
        flagged = int(cm_t[0][1] + cm_t[1][1])
    else:
        flagged = int(y_pred_t.sum())
    metric_card(f"{flagged}", "Bills Flagged", f"Out of {len(y_test)} test bills", col4)
    
    st.markdown("")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Precision-Recall-F1 vs Threshold
        st.markdown('<p class="section-header">Metrics vs Threshold</p>', unsafe_allow_html=True)
        
        threshold_data = eda_data['threshold_analysis']
        thresholds = [t['threshold'] for t in threshold_data]
        precisions = [t['precision'] for t in threshold_data]
        recalls = [t['recall'] for t in threshold_data]
        f1s = [t['f1_score'] for t in threshold_data]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=precisions, name='Precision',
                                 line=dict(color='#6366f1', width=2.5)))
        fig.add_trace(go.Scatter(x=thresholds, y=recalls, name='Recall',
                                 line=dict(color='#ef4444', width=2.5)))
        fig.add_trace(go.Scatter(x=thresholds, y=f1s, name='F1 Score',
                                 line=dict(color='#22c55e', width=2.5)))
        
        # Vertical line at current threshold
        fig.add_vline(x=thresh_val, line_dash="dash", line_color="#f59e0b", line_width=2,
                     annotation_text=f"Current: {thresh_val}", annotation_font_color="#f59e0b")
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            font=dict(color='#94a3b8', family='Inter'),
            xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Threshold'),
            yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Score', range=[0, 1.05]),
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(font=dict(size=12)),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        # Confusion matrix at current threshold
        st.markdown('<p class="section-header">Confusion Matrix at Current Threshold</p>', unsafe_allow_html=True)
        
        if len(cm_t) == 2:
            fig = go.Figure(data=go.Heatmap(
                z=cm_t,
                x=['Predicted Safe', 'Predicted Fraud'],
                y=['Actually Safe', 'Actually Fraud'],
                colorscale=[[0, '#1e1b4b'], [0.5, '#4338ca'], [1, '#818cf8']],
                text=[[f"TN = {cm_t[0][0]}", f"FP = {cm_t[0][1]}"],
                      [f"FN = {cm_t[1][0]}", f"TP = {cm_t[1][1]}"]],
                texttemplate='%{text}',
                textfont=dict(size=14, color='#e2e8f0'),
                showscale=False,
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', family='Inter'),
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Business impact analysis
    st.markdown('<p class="section-header">📋 Business Impact Analysis</p>', unsafe_allow_html=True)
    
    if len(cm_t) == 2:
        tn, fp, fn, tp = cm_t[0][0], cm_t[0][1], cm_t[1][0], cm_t[1][1]
        total_fraud = tp + fn
        total_legit = tn + fp
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **At threshold = {thresh_val:.2f}:**
            - 🎯 **Fraud caught:** {tp} out of {total_fraud} ({tp/total_fraud*100:.1f}% if total_fraud > 0)
            - ❌ **Fraud missed:** {fn} cases (potential revenue loss)
            - 🚫 **False alarms:** {fp} legitimate bills wrongly flagged
            - ✅ **Correctly cleared:** {tn} legitimate bills
            """)
        
        with col2:
            st.markdown(f"""
            **Investigator workload:**
            - 📋 **Bills to review:** {tp + fp} (flagged bills)
            - 📈 **Hit rate:** {tp/(tp+fp)*100:.1f}% of flagged bills are actual fraud
            - ⏱️ **Efficiency:** For every 10 bills reviewed, ~{tp/(tp+fp)*10:.1f} are real fraud
            """) if (tp + fp) > 0 else st.markdown("No bills flagged at this threshold.")
    
    # Threshold recommendations
    st.markdown('<p class="section-header">💡 Threshold Recommendations</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p style="color: #22c55e; font-weight: 700;">Conservative (0.7–0.9)</p>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                High precision, fewer false alarms.<br>
                Best when: investigation capacity is limited, 
                or false accusations damage customer trust.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p style="color: #f59e0b; font-weight: 700;">Balanced (0.4–0.6)</p>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Good balance of precision and recall.<br>
                Best when: you want a reasonable default that 
                catches most fraud without too many false alarms.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <p style="color: #ef4444; font-weight: 700;">Aggressive (0.1–0.3)</p>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                High recall, catches almost all fraud.<br>
                Best when: missed fraud is very expensive, 
                and you have capacity for more investigations.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: MONITORING & MODEL INFO (Step 11)
# ============================================================
elif page == "📈 Monitoring":
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">📈 Model Monitoring & Info</h1>
        <p class="hero-subtitle">Model metadata, feature documentation, and production monitoring guidance</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Section 1: Model Metadata ----
    st.markdown('<p class="section-header">🗂️ Model Metadata</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metric_card(eda_data['best_model_name'], "Best Model", "Selected by highest recall", col1)
    metric_card(eda_data.get('training_date', 'N/A'), "Training Date", "Last trained", col2)
    metric_card(f"{eda_data['total_rows_clean']}", "Training Rows",
                f"{eda_data['total_rows_raw']} raw → {eda_data['total_rows_clean']} clean", col3)
    metric_card(f"{eda_data['n_features']}", "Features Used",
                f"{len([c for c in feature_names if '_missing' in c])} are missingness flags", col4)

    st.markdown("")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<p class="section-header">📋 Training Configuration</p>', unsafe_allow_html=True)
        config_data = {
            'Parameter': [
                'Train / Test Split', 'Stratified Split', 'Random Seed',
                'Class Weight Handling', 'Scaler (Logistic Reg.)',
                'Models Trained', 'Selection Criterion',
                'Train Fraud Rate', 'Test Fraud Rate',
                'Missing Imputation'
            ],
            'Value': [
                f"{100 - 20}% / {20}%",
                'Yes (preserves fraud ratio)',
                '42',
                'Balanced (all models)',
                'StandardScaler',
                ', '.join(eda_data.get('models_trained', list(model_results.keys()))),
                'Highest Recall → F1 tiebreaker',
                f"{eda_data.get('train_fraud_rate', 0):.1f}%",
                f"{eda_data.get('test_fraud_rate', 0):.1f}%",
                'Median imputation'
            ]
        }
        st.dataframe(pd.DataFrame(config_data).set_index('Parameter'), use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">📊 Dataset Summary</p>', unsafe_allow_html=True)
        dataset_info = {
            'Metric': [
                'Raw Rows', 'Rows Dropped (missing target)', 'Clean Rows',
                'Fraud Cases', 'Non-Fraud Cases', 'Fraud Rate',
                'Original Features', 'Engineered Features', 'Total Features'
            ],
            'Value': [
                str(eda_data['total_rows_raw']),
                str(eda_data['n_dropped_target']),
                str(eda_data['total_rows_clean']),
                str(eda_data['class_distribution']['fraud']),
                str(eda_data['class_distribution']['non_fraud']),
                f"{eda_data['fraud_rate']:.1f}%",
                '7 (billing, meter, location, complaints)',
                '7 (ratios, diffs, interactions, flags)',
                str(eda_data['n_features'])
            ]
        }
        st.dataframe(pd.DataFrame(dataset_info).set_index('Metric'), use_container_width=True)

    # ---- Section 2: Feature Documentation ----
    st.markdown('<p class="section-header">🧬 Feature Documentation</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>📖 Feature Categories:</strong> The model uses three types of features —
        <em>raw measurements</em> from the billing system, <em>engineered signals</em> computed during preprocessing,
        and <em>missingness flags</em> that capture whether original data was absent (which itself can signal fraud).
    </div>
    """, unsafe_allow_html=True)

    feature_docs = {
        'Feature': [
            'billing_amount', 'avg_last_6_months', 'payment_delay_days',
            'meter_reading', 'previous_reading', 'location_risk_score', 'num_complaints',
            'billing_ratio', 'billing_deviation', 'usage_diff',
            'delay_risk_interaction', 'billing_per_unit', 'high_billing_flag',
            'payment_delay_category', '*_missing flags (7)'
        ],
        'Type': [
            'Raw', 'Raw', 'Raw', 'Raw', 'Raw', 'Raw', 'Raw',
            'Engineered', 'Engineered', 'Engineered',
            'Engineered', 'Engineered', 'Engineered',
            'Engineered', 'Signal'
        ],
        'Description': [
            'Current billing amount in ₹',
            'Average bill over the last 6 months',
            'Days the payment was delayed',
            'Current meter reading value',
            'Previous period meter reading',
            'Risk score for the customer location (0–1)',
            'Number of complaints filed by customer',
            'billing_amount / avg_last_6_months (capped at 50)',
            'billing_amount − avg_last_6_months',
            'meter_reading − previous_reading',
            'payment_delay_days × location_risk_score',
            'billing_amount / meter_diff (capped ±100)',
            '1 if billing_ratio > 2, else 0',
            'Binned delay: 0=0-7d, 1=8-14d, 2=15-21d, 3=22-30d',
            'Binary flag: 1 if original value was missing, 0 otherwise'
        ],
        'Fraud Signal': [
            'Higher bills may indicate overbilling',
            'Context baseline for billing_ratio',
            'Late payments correlate with fraud',
            'Potential meter tampering signal',
            'Baseline for usage delta',
            'High-risk areas have more fraud',
            'Weak direct signal',
            '⭐ Strong — ratio > 2 is suspicious',
            'Large positive diffs flag overbilling',
            'Negative values may signal meter resets',
            'Combined risk amplifier',
            'Extreme values flag billing-usage mismatch',
            'Binary threshold trigger',
            'Categorical risk grouping',
            'Strategic missing data may indicate fraud'
        ]
    }
    st.dataframe(pd.DataFrame(feature_docs).set_index('Feature'), use_container_width=True, height=520)

    # ---- Section 3: Feature Importance Deep Dive ----
    st.markdown('<p class="section-header">🎯 Feature Importance — Deep Dive</p>', unsafe_allow_html=True)

    best_importances = model_results[eda_data['best_model_name']]['feature_importances']

    fig = go.Figure()
    feat_names = list(best_importances.keys())
    feat_vals = list(best_importances.values())
    cumulative = np.cumsum(feat_vals) / sum(feat_vals) * 100

    fig.add_trace(go.Bar(
        x=feat_names,
        y=feat_vals,
        name='Importance',
        marker=dict(
            color=feat_vals,
            colorscale=[[0, '#312e81'], [0.3, '#6366f1'], [0.6, '#8b5cf6'], [1, '#c4b5fd']],
        ),
        text=[f"{v:.4f}" for v in feat_vals],
        textposition='outside',
        textfont=dict(size=9, color='#c7d2fe'),
    ))

    fig.add_trace(go.Scatter(
        x=feat_names,
        y=cumulative,
        name='Cumulative %',
        yaxis='y2',
        line=dict(color='#f59e0b', width=2.5, dash='dot'),
        mode='lines+markers',
        marker=dict(size=5, color='#f59e0b'),
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        font=dict(color='#94a3b8', family='Inter'),
        xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', tickangle=45),
        yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)', title='Importance'),
        yaxis2=dict(
            title='Cumulative %', overlaying='y', side='right',
            range=[0, 105], gridcolor='rgba(245, 158, 11, 0.1)',
            ticksuffix='%'
        ),
        height=450,
        margin=dict(l=20, r=60, t=20, b=120),
        legend=dict(font=dict(size=11), orientation='h', y=1.08, x=0.5, xanchor='center'),
        barmode='group',
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Section 4: Model Performance Snapshot ----
    st.markdown('<p class="section-header">📊 Production Model Performance Snapshot</p>', unsafe_allow_html=True)

    best_res = model_results[eda_data['best_model_name']]
    col1, col2, col3, col4, col5 = st.columns(5)
    metric_card(f"{best_res['accuracy']*100:.1f}%", "Accuracy", "", col1)
    metric_card(f"{best_res['precision']*100:.1f}%", "Precision", "", col2)
    metric_card(f"{best_res['recall']*100:.1f}%", "Recall", "", col3)
    metric_card(f"{best_res['f1_score']*100:.1f}%", "F1 Score", "", col4)
    metric_card(f"{best_res['roc_auc']*100:.1f}%", "ROC-AUC", "", col5)

    st.markdown("")

    # ---- Section 5: Data Drift Detection (Placeholder) ----
    st.markdown('<p class="section-header">🔄 Data Drift Detection</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>📖 What is Data Drift?</strong><br>
        Data drift occurs when the statistical properties of the input data change over time,
        causing the model's predictions to become less reliable. For a billing fraud system, drift could occur when:
        <ul style="margin-top: 8px; color: #94a3b8;">
            <li>Customer billing patterns change seasonally (e.g., winter heating bills)</li>
            <li>Tariff structures are updated by the utility provider</li>
            <li>Fraud tactics evolve (adversarial drift)</li>
            <li>New customer segments are onboarded with different baselines</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <p style="color: #818cf8; font-weight: 700; font-size: 1.1rem;">🔍 Recommended Drift Checks</p>
            <ul style="color: #94a3b8; font-size: 0.88rem; line-height: 1.8;">
                <li><strong>Population Stability Index (PSI)</strong> — Measures distribution shift between training and live data</li>
                <li><strong>Kolmogorov-Smirnov Test</strong> — Statistical test for distribution changes per feature</li>
                <li><strong>Feature Mean / Std Tracking</strong> — Monitor rolling statistics against training baselines</li>
                <li><strong>Prediction Distribution</strong> — Alert if the model's predicted fraud rate deviates ±5% from training baseline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <p style="color: #f59e0b; font-weight: 700; font-size: 1.1rem;">⚡ Implementation Roadmap</p>
            <ul style="color: #94a3b8; font-size: 0.88rem; line-height: 1.8;">
                <li><strong>Phase 1:</strong> Log all predictions with timestamps and input features</li>
                <li><strong>Phase 2:</strong> Weekly PSI computation against training distribution</li>
                <li><strong>Phase 3:</strong> Automated alerts when PSI > 0.2 (significant drift)</li>
                <li><strong>Phase 4:</strong> Scheduled retraining pipeline with fresh labeled data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Training vs expected ranges table
    st.markdown('<p class="section-header">📐 Training Data Baseline Ranges</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Monitor incoming data against these ranges. Significant deviations may indicate data drift or upstream data quality issues.
    </div>
    """, unsafe_allow_html=True)

    baseline_rows = []
    desc = eda_data.get('desc_stats', {})
    for col_name in ['billing_amount', 'avg_last_6_months', 'payment_delay_days',
                     'meter_reading', 'previous_reading', 'location_risk_score', 'num_complaints']:
        stats_col = desc.get(col_name, {})
        if stats_col:
            baseline_rows.append({
                'Feature': col_name,
                'Mean': f"{stats_col.get('mean', 0):.2f}",
                'Std': f"{stats_col.get('std', 0):.2f}",
                'Min': f"{stats_col.get('min', 0):.2f}",
                '25%': f"{stats_col.get('25%', 0):.2f}",
                'Median': f"{stats_col.get('50%', 0):.2f}",
                '75%': f"{stats_col.get('75%', 0):.2f}",
                'Max': f"{stats_col.get('max', 0):.2f}",
            })

    if baseline_rows:
        st.dataframe(pd.DataFrame(baseline_rows).set_index('Feature'), use_container_width=True)

    # ---- Section 6: Retraining Recommendations ----
    st.markdown('<p class="section-header">🔁 Model Retraining Recommendations</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p style="color: #22c55e; font-weight: 700;">📅 Scheduled Retraining</p>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Retrain the model on a fixed schedule (e.g., monthly or quarterly)
                to incorporate new labeled data and adapt to evolving patterns.<br><br>
                <strong>Recommended:</strong> Every 30–90 days, or when ≥500 new labeled samples are available.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <p style="color: #f59e0b; font-weight: 700;">📉 Performance-Triggered</p>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Retrain when live performance metrics drop below acceptable thresholds.<br><br>
                <strong>Triggers:</strong>
                <ul style="font-size: 0.82rem; margin-top: 4px;">
                    <li>Recall drops below 90%</li>
                    <li>Precision drops below 85%</li>
                    <li>ROC-AUC drops below 0.90</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <p style="color: #ef4444; font-weight: 700;">🚨 Drift-Triggered</p>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Retrain immediately when significant data drift is detected.<br><br>
                <strong>Indicators:</strong>
                <ul style="font-size: 0.82rem; margin-top: 4px;">
                    <li>PSI > 0.2 for any key feature</li>
                    <li>Fraud rate shifts ±10% from baseline</li>
                    <li>New feature distributions detected</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ---- Section 7: Saved Artifacts ----
    st.markdown('<p class="section-header">💾 Saved Model Artifacts</p>', unsafe_allow_html=True)

    artifacts_data = {
        'File': [
            'best_model.pkl', 'scaler.pkl', 'model_results.pkl',
            'eda_data.pkl', 'clean_data.pkl', 'feature_names.pkl',
            'medians.pkl', 'test_data.pkl',
            'logistic_regression.pkl', 'decision_tree.pkl',
            'random_forest.pkl', 'xgboost.pkl'
        ],
        'Purpose': [
            'Production model for predictions',
            'StandardScaler (for Logistic Regression inputs)',
            'All model evaluation metrics, curves, and probabilities',
            'EDA statistics, thresholds, correlations, and metadata',
            'Cleaned dataset for app visualizations',
            'Ordered feature column names',
            'Median values for missing data imputation',
            'Test set (X, y, scaled X) for threshold analysis',
            'Trained Logistic Regression model',
            'Trained Decision Tree model',
            'Trained Random Forest model',
            'Trained XGBoost model'
        ],
        'Required For': [
            'Predict Fraud page', 'Predict Fraud page (if LR)',
            'Dashboard, Model Performance, Threshold pages',
            'All pages (metadata + charts)',
            'EDA & Analysis page',
            'Predict Fraud page',
            'Future imputation reference',
            'Threshold Tuning page',
            'Model comparison only',
            'Model comparison only',
            'Model comparison only',
            'Model comparison only'
        ]
    }
    st.dataframe(pd.DataFrame(artifacts_data).set_index('File'), use_container_width=True, height=450)

    # ---- Footer ----
    st.markdown("")
    st.markdown("""
    <div class="info-box" style="text-align: center;">
        <strong>🛡️ Billing Fraud Detection System</strong><br>
        <span style="color: #64748b;">ML Pipeline v1.0 — Built with Scikit-learn, XGBoost, and Streamlit</span><br>
        <span style="color: #64748b; font-size: 0.8rem;">For production deployment, integrate with your billing database and set up scheduled retraining.</span>
    </div>
    """, unsafe_allow_html=True)
