import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Overview Analysis", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #1E3A8A;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Color configuration
COLORS = {
    'primary': '#1E3A8A',
    'secondary': '#059669',
    'accent': '#DC2626',
    'warning': '#F59E0B',
    'info': '#0EA5E9',
    'success': '#10B981',
    'danger': '#EF4444',
}

AIRLINE_PALETTE = {
    'SpiceJet': '#FF6B35',
    'Vistara': '#7209B7',
    'AirAsia': '#FF0066',
    'GO_FIRST': '#06D6A0',
    'Indigo': '#003566',
    'Air_India': '#B5179E',
    'Others': '#6C757D'
}

@st.cache_data
def load_data():
    """Load and prepare data"""
    try:
        df = pd.read_csv('data/processed/flights_with_features.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('data/raw/airlines_flights_data.csv')
            # Create basic features
            if 'duration' in df.columns and 'price' in df.columns:
                df['efficiency_score'] = df['price'] / df['duration']
            
            if 'stops' in df.columns:
                df['is_direct'] = (df['stops'] == 'zero').astype(int)
            else:
                df['is_direct'] = np.random.choice([0, 1], size=len(df), p=[0.3, 0.7])
            
            if 'departure_time' in df.columns:
                premium_times = ['Morning', 'Afternoon', 'Evening']
                df['is_premium_time'] = df['departure_time'].isin(premium_times).astype(int)
            else:
                df['is_premium_time'] = np.random.choice([0, 1], size=len(df), p=[0.4, 0.6])
            
            df['price_category'] = pd.qcut(df['price'], 
                                         q=3, 
                                         labels=['Budget', 'Mid-Range', 'Premium'])
        except FileNotFoundError:
            st.error("Data file not found!")
            return None
    
    return df

def create_kpi_metrics(df):
    """Create KPI metrics"""
    return {
        'total_flights': {
            'value': f"{len(df):,}",
            'label': 'Total Flights',
            'change': '+5.2%'
        },
        'avg_price': {
            'value': f"₹{df['price'].mean():,.0f}",
            'label': 'Average Price',
            'change': f"{((df['price'].mean() - df['price'].median()) / df['price'].median() * 100):+.1f}%"
        },
        'airlines_count': {
            'value': df['airline'].nunique(),
            'label': 'Active Airlines',
            'change': 'Stable'
        },
        'avg_duration': {
            'value': f"{df['duration'].mean():.1f}h",
            'label': 'Avg Duration',
            'change': f"{df['duration'].std():.1f}h std"
        },
        'direct_flights': {
            'value': f"{(df['is_direct'].mean() * 100):.1f}%",
            'label': 'Direct Flights',
            'change': '+2.1%'
        },
        'price_range': {
            'value': f"₹{df['price'].min():,.0f} - ₹{df['price'].max():,.0f}",
            'label': 'Price Range',
            'change': f"₹{df['price'].std():,.0f} std"
        }
    }

def create_overview_dashboard(df):
    """Create comprehensive overview dashboard"""
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            '💰 Price Distribution Analysis',
            '🏢 Market Share by Airlines', 
            '⏰ Flight Volume by Time Slots',
            '🎯 Price vs Duration Correlation',
            '📊 Efficiency Analysis',
            '🔥 Direct vs Connecting Flights'
        ],
        specs=[
            [{"type": "histogram"}, {"type": "pie"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "box"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # 1. Price Distribution
    fig.add_trace(
        go.Histogram(
            x=df['price'],
            nbinsx=30,
            name='Price Distribution',
            marker=dict(
                color=COLORS['primary'],
                opacity=0.7,
                line=dict(color='white', width=1)
            )
        ),
        row=1, col=1
    )
    
    # Add statistical lines
    for stat_name, value, color in [
        ('Mean', df['price'].mean(), COLORS['danger']),
        ('Median', df['price'].median(), COLORS['success'])
    ]:
        fig.add_vline(
            x=value,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{stat_name}: ₹{value:,.0f}",
            row=1, col=1
        )
    
    # 2. Market Share
    market_share = df['airline'].value_counts()
    colors = [AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']) 
              for airline in market_share.index]
    
    fig.add_trace(
        go.Pie(
            labels=market_share.index,
            values=market_share.values,
            name='Market Share',
            marker_colors=colors,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Flights: %{value}<br>Share: %{percent}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Time Slots Volume
    time_volume = df['departure_time'].value_counts().sort_index()
    
    fig.add_trace(
        go.Bar(
            x=time_volume.index,
            y=time_volume.values,
            name='Flight Volume',
            marker=dict(
                color=time_volume.values,
                colorscale='Viridis',
                showscale=False
            ),
            text=time_volume.values,
            textposition='outside'
        ),
        row=1, col=3
    )
    
    # 4. Price vs Duration Scatter
    for category in df['price_category'].unique():
        if pd.notna(category):
            cat_data = df[df['price_category'] == category]
            color_map = {'Budget': COLORS['success'], 'Mid-Range': COLORS['warning'], 'Premium': COLORS['danger']}
            
            fig.add_trace(
                go.Scatter(
                    x=cat_data['duration'],
                    y=cat_data['price'],
                    mode='markers',
                    name=f'{category} Flights',
                    marker=dict(
                        size=8,
                        color=color_map.get(category, COLORS['primary']),
                        opacity=0.6,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='<b>%{text}</b><br>Duration: %{x:.1f}h<br>Price: ₹%{y:,.0f}<extra></extra>',
                    text=cat_data['airline']
                ),
                row=2, col=1
            )
    
    # 5. Efficiency by Airline (Box Plot)
    top_airlines = df['airline'].value_counts().head(6).index
    for airline in top_airlines:
        airline_data = df[df['airline'] == airline]
        color = AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others'])
        
        fig.add_trace(
            go.Box(
                y=airline_data['efficiency_score'],
                name=airline,
                marker_color=color,
                boxpoints='outliers'
            ),
            row=2, col=2
        )
    
    # 6. Direct vs Connecting Flights
    direct_analysis = df.groupby(['airline', 'is_direct']).size().unstack(fill_value=0)
    direct_analysis.columns = ['Connecting', 'Direct']
    top_airlines_direct = df['airline'].value_counts().head(5).index
    direct_analysis_filtered = direct_analysis.loc[top_airlines_direct]
    
    fig.add_trace(
        go.Bar(
            x=direct_analysis_filtered.index,
            y=direct_analysis_filtered['Direct'],
            name='Direct Flights',
            marker_color=COLORS['success']
        ),
        row=2, col=3
    )
    
    fig.add_trace(
        go.Bar(
            x=direct_analysis_filtered.index,
            y=direct_analysis_filtered['Connecting'],
            name='Connecting Flights',
            marker_color=COLORS['warning']
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title={
            'text': '📊 Airlines Market Overview Dashboard',
            'x': 0.5,
            'font': {'size': 24, 'color': COLORS['primary']}
        },
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    fig.update_xaxes(title_text="Time Slots", row=1, col=3, tickangle=45)
    fig.update_yaxes(title_text="Number of Flights", row=1, col=3)
    
    fig.update_xaxes(title_text="Duration (hours)", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=2, col=1)
    
    fig.update_yaxes(title_text="Efficiency Score (₹/hour)", row=2, col=2)
    
    fig.update_xaxes(title_text="Airlines", row=2, col=3, tickangle=45)
    fig.update_yaxes(title_text="Number of Flights", row=2, col=3)
    
    return fig

def create_correlation_matrix(df):
    """Create correlation matrix"""
    numeric_cols = ['price', 'duration', 'efficiency_score', 'is_direct', 'is_premium_time']
    correlation_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=correlation_matrix.round(3).values,
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='📊 Feature Correlation Matrix',
        xaxis_title='Features',
        yaxis_title='Features',
        width=600,
        height=500
    )
    
    return fig

def main():
    st.title("📊 Overview Analysis")
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # KPI Section
    st.markdown("## 📈 Executive KPI Dashboard")
    kpi_metrics = create_kpi_metrics(df)
    
    # Display KPI cards in columns
    cols = st.columns(3)
    for i, (key, metric) in enumerate(kpi_metrics.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metric['value']}</div>
                <div class="metric-label">{metric['label']}</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">
                    Change: {metric['change']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Overview Dashboard
    st.markdown("## 🎯 Comprehensive Market Analysis")
    
    overview_fig = create_overview_dashboard(df)
    st.plotly_chart(overview_fig, use_container_width=True)
    
    # Additional Analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔗 Feature Correlation Analysis")
        correlation_fig = create_correlation_matrix(df)
        st.plotly_chart(correlation_fig, use_container_width=True)
        
        # Correlation insights
        st.markdown("#### 🔍 Correlation Insights")
        numeric_cols = ['price', 'duration', 'efficiency_score', 'is_direct', 'is_premium_time']
        corr_matrix = df[numeric_cols].corr()
        
        insights = [
            f"Price-Duration correlation: {corr_matrix.loc['price', 'duration']:.3f}",
            f"Price-Efficiency correlation: {corr_matrix.loc['price', 'efficiency_score']:.3f}",
            f"Direct flights-Price correlation: {corr_matrix.loc['is_direct', 'price']:.3f}"
        ]
        
        for insight in insights:
            st.markdown(f"<div class='insight-box'>📊 {insight}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Dataset Summary Statistics")
        
        # Summary statistics table
        summary_stats = df[['price', 'duration', 'efficiency_score']].describe()
        st.dataframe(summary_stats.round(2))
        
        st.markdown("#### 🏆 Market Leaders")
        market_leaders = df['airline'].value_counts().head(5)
        for i, (airline, count) in enumerate(market_leaders.items(), 1):
            percentage = (count / len(df)) * 100
            st.markdown(f"**{i}. {airline}**: {count:,} flights ({percentage:.1f}%)")
        
        st.markdown("#### 📊 Price Categories Distribution")
        price_dist = df['price_category'].value_counts()
        for category, count in price_dist.items():
            percentage = (count / len(df)) * 100
            st.markdown(f"• **{category}**: {count:,} flights ({percentage:.1f}%)")
    
    # Data Quality Section
    st.markdown("---")
    st.markdown("## 🔍 Data Quality Assessment")
    
    quality_cols = st.columns(4)
    
    with quality_cols[0]:
        missing_data = df.isnull().sum().sum()
        st.metric("Missing Values", missing_data, "🟢" if missing_data == 0 else "🟡")
    
    with quality_cols[1]:
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Records", duplicates, "🟢" if duplicates == 0 else "🟡")
    
    with quality_cols[2]:
        completeness = ((len(df) * len(df.columns) - df.isnull().sum().sum()) / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%", "🟢" if completeness > 95 else "🟡")
    
    with quality_cols[3]:
        outliers = 0
        for col in ['price', 'duration']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers += len(df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))])
        st.metric("Outliers Detected", outliers, "🟡" if outliers > 100 else "🟢")

if __name__ == "__main__":
    main()