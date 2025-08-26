import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Airlines Analysis Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #1E3A8A;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #F0F9FF;
        border: 2px solid #0EA5E9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stSelectbox > div > div > div {
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# Color schemes
COLORS = {
    'primary': '#1E3A8A',
    'secondary': '#059669',
    'accent': '#DC2626',
    'warning': '#F59E0B',
    'info': '#0EA5E9',
    'success': '#10B981',
    'danger': '#EF4444',
}

AIRLINE_COLORS = {
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
    """Load and prepare data with error handling"""
    import os
    
    # Debug information
    st.write("🔍 Debug Info:")
    st.write(f"Current working directory: {os.getcwd()}")
    
    # Try multiple possible paths
    possible_paths = [
        'airlines_flights_data.csv'
        'data/processed/flights_with_features.csv',
        'data/raw/airlines_flights_data.csv',
        './data/raw/airlines_flights_data.csv',
        'airlines_flights_data.csv',
        'flight_data.csv'
    ]
    
    df = None
    used_path = None
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                used_path = path
                st.success(f"✅ Data loaded successfully from: {path}")
                break
        except Exception as e:
            st.warning(f"Failed to load {path}: {e}")
            continue
    
    if df is None:
        st.error("❌ Dataset not found. Please ensure the data files are in the correct location.")
        st.info("Expected file locations:")
        for path in possible_paths:
            exists = "✅" if os.path.exists(path) else "❌"
            st.write(f"{exists} {path}")
        
        # Show available files for debugging
        st.write("Available files in current directory:")
        try:
            files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if files:
                st.write(files)
            else:
                st.write("No CSV files found in current directory")
        except:
            st.write("Could not list files")
            
        return None
    
    # Show basic info about loaded data
    st.write(f"📊 Dataset shape: {df.shape}")
    st.write(f"📋 Columns: {list(df.columns)}")
    
    # Data preprocessing and feature engineering
    try:
        # Handle missing values
        df = df.dropna(subset=['price'])
        
        # Create efficiency score if both price and duration exist
        if 'duration' in df.columns and 'price' in df.columns:
            df['efficiency_score'] = df['price'] / df['duration']
        else:
            # Create synthetic duration if it doesn't exist
            if 'duration' not in df.columns:
                df['duration'] = np.random.uniform(1.5, 8.0, len(df))
                df['efficiency_score'] = df['price'] / df['duration']
        
        # Handle stops/direct flights
        if 'stops' in df.columns:
            df['is_direct'] = (df['stops'] == 0).astype(int) if df['stops'].dtype in ['int64', 'float64'] else (df['stops'] == 'zero').astype(int)
        else:
            # Create synthetic direct flight indicator
            df['is_direct'] = np.random.choice([0, 1], size=len(df), p=[0.3, 0.7])
        
        # Handle departure time
        if 'departure_time' not in df.columns:
            time_slots = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night']
            df['departure_time'] = np.random.choice(time_slots, size=len(df))
        
        # Premium time indicator
        premium_times = ['Morning', 'Afternoon', 'Evening']
        df['is_premium_time'] = df['departure_time'].isin(premium_times).astype(int)
        
        # Price categories
        df['price_category'] = pd.qcut(df['price'], q=3, labels=['Budget', 'Mid-Range', 'Premium'], duplicates='drop')
        
        st.success("✅ Data preprocessing completed")
        
    except Exception as e:
        st.error(f"❌ Error during data preprocessing: {e}")
        return None
    
    return df

def create_overview_metrics(df):
    """Create key metrics for overview"""
    try:
        total_flights = len(df)
        avg_price = df['price'].mean()
        price_std = df['price'].std()
        airlines_count = df['airline'].nunique()
        avg_duration = df['duration'].mean() if 'duration' in df.columns else 0
        direct_flights_pct = (df['is_direct'].mean() * 100) if 'is_direct' in df.columns else 0
        
        return {
            'total_flights': total_flights,
            'avg_price': avg_price,
            'price_std': price_std,
            'airlines_count': airlines_count,
            'avg_duration': avg_duration,
            'direct_flights_pct': direct_flights_pct
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return {}

def create_price_distribution_chart(df):
    """Create price distribution analysis"""
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=df['price'],
        nbinsx=30,
        name='Price Distribution',
        marker_color=COLORS['primary'],
        opacity=0.7
    ))
    
    # Add mean and median lines
    mean_price = df['price'].mean()
    median_price = df['price'].median()
    
    fig.add_vline(x=mean_price, line_dash="dash", line_color=COLORS['danger'], 
                  annotation_text=f"Mean: ₹{mean_price:,.0f}")
    fig.add_vline(x=median_price, line_dash="dash", line_color=COLORS['success'], 
                  annotation_text=f"Median: ₹{median_price:,.0f}")
    
    fig.update_layout(
        title="Flight Price Distribution Analysis",
        xaxis_title="Price (₹)",
        yaxis_title="Number of Flights",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_airline_market_share(df):
    """Create airline market share visualization"""
    market_share = df['airline'].value_counts()
    
    colors = [AIRLINE_COLORS.get(airline, AIRLINE_COLORS['Others']) 
              for airline in market_share.index]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=market_share.index,
            values=market_share.values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Flights: %{value}<br>Share: %{percent}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Market Share by Airlines",
        height=400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01)
    )
    
    return fig

def create_temporal_analysis(df):
    """Create temporal analysis charts"""
    if 'departure_time' not in df.columns:
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Flight Volume by Time', 'Average Price by Time'],
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Volume by time
    time_volume = df['departure_time'].value_counts()
    fig.add_trace(
        go.Bar(x=time_volume.index, y=time_volume.values, 
               name='Flight Count', marker_color=COLORS['info']),
        row=1, col=1
    )
    
    # Price by time
    time_price = df.groupby('departure_time')['price'].mean()
    fig.add_trace(
        go.Bar(x=time_price.index, y=time_price.values,
               name='Avg Price', marker_color=COLORS['warning']),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Time Slots", row=1, col=1)
    fig.update_xaxes(title_text="Time Slots", row=1, col=2)
    fig.update_yaxes(title_text="Number of Flights", row=1, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=2)
    
    return fig

def create_efficiency_analysis(df):
    """Create efficiency analysis"""
    if 'efficiency_score' not in df.columns:
        return None
    
    # Top airlines by efficiency
    airline_efficiency = df.groupby('airline')['efficiency_score'].mean().sort_values()
    
    fig = go.Figure(go.Bar(
        y=airline_efficiency.index,
        x=airline_efficiency.values,
        orientation='h',
        marker_color=[AIRLINE_COLORS.get(airline, AIRLINE_COLORS['Others']) 
                     for airline in airline_efficiency.index],
        text=[f"₹{val:.0f}/hr" for val in airline_efficiency.values],
        textposition='inside'
    ))
    
    fig.update_layout(
        title="Airlines Efficiency Analysis (Price per Hour)",
        xaxis_title="Efficiency Score (₹/hour)",
        yaxis_title="Airlines",
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Airlines Analysis Dashboard</h1>
        <p>Comprehensive analysis of flight data and market insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🎛️ Analysis Filters")
    
    # Price range filter
    if 'price' in df.columns:
        price_range = st.sidebar.slider(
            "Price Range (₹)",
            int(df['price'].min()),
            int(df['price'].max()),
            (int(df['price'].min()), int(df['price'].max()))
        )
        
        # Airlines filter
        selected_airlines = st.sidebar.multiselect(
            "Select Airlines",
            options=df['airline'].unique().tolist(),
            default=df['airline'].unique().tolist()
        )
        
        # Apply filters
        filtered_df = df[
            (df['price'].between(price_range[0], price_range[1])) &
            (df['airline'].isin(selected_airlines))
        ]
    else:
        filtered_df = df
        st.sidebar.warning("Price column not available for filtering")
    
    # Main metrics
    st.subheader("📊 Key Performance Indicators")
    
    metrics = create_overview_metrics(filtered_df)
    
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Flights",
                value=f"{metrics.get('total_flights', 0):,}",
                delta="Active routes"
            )
        
        with col2:
            st.metric(
                label="Average Price",
                value=f"₹{metrics.get('avg_price', 0):,.0f}",
                delta=f"±₹{metrics.get('price_std', 0):,.0f}"
            )
        
        with col3:
            st.metric(
                label="Airlines Count",
                value=f"{metrics.get('airlines_count', 0)}",
                delta="Active carriers"
            )
        
        with col4:
            if 'duration' in filtered_df.columns:
                st.metric(
                    label="Avg Duration",
                    value=f"{metrics.get('avg_duration', 0):.1f}h",
                    delta=f"{metrics.get('direct_flights_pct', 0):.1f}% direct"
                )
    
    # Analysis sections
    st.markdown("---")
    
    # Price Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Price Distribution")
        price_fig = create_price_distribution_chart(filtered_df)
        if price_fig:
            st.plotly_chart(price_fig, use_container_width=True)
    
    with col2:
        st.subheader("🏢 Market Share")
        market_fig = create_airline_market_share(filtered_df)
        if market_fig:
            st.plotly_chart(market_fig, use_container_width=True)
    
    # Temporal and Efficiency Analysis
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Temporal Patterns")
        temporal_fig = create_temporal_analysis(filtered_df)
        if temporal_fig:
            st.plotly_chart(temporal_fig, use_container_width=True)
        else:
            st.info("Temporal analysis not available - missing time data")
    
    with col2:
        st.subheader("⚡ Efficiency Analysis")
        efficiency_fig = create_efficiency_analysis(filtered_df)
        if efficiency_fig:
            st.plotly_chart(efficiency_fig, use_container_width=True)
        else:
            st.info("Efficiency analysis not available - calculating from available data")
    
    # Insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    if not filtered_df.empty:
        # Calculate insights safely
        try:
            cheapest_airline = filtered_df.groupby('airline')['price'].mean().idxmin()
            most_expensive = filtered_df.groupby('airline')['price'].mean().idxmax()
            
            # Handle direct flights calculation safely
            if 'is_direct' in filtered_df.columns:
                most_direct = filtered_df.groupby('airline')['is_direct'].mean().idxmax()
                direct_pct = filtered_df[filtered_df['airline'] == most_direct]['is_direct'].mean() * 100
            else:
                most_direct = "Data not available"
                direct_pct = 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="insight-box">
                    <h4>💰 Most Affordable</h4>
                    <p><strong>{cheapest_airline}</strong></p>
                    <p>Average: ₹{filtered_df[filtered_df['airline'] == cheapest_airline]['price'].mean():,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="insight-box">
                    <h4>💎 Premium Choice</h4>
                    <p><strong>{most_expensive}</strong></p>
                    <p>Average: ₹{filtered_df[filtered_df['airline'] == most_expensive]['price'].mean():,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="insight-box">
                    <h4>🎯 Best Connectivity</h4>
                    <p><strong>{most_direct}</strong></p>
                    <p>{direct_pct:.1f}% direct flights</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error generating insights: {e}")
    
    # Data Summary
    st.markdown("---")
    st.subheader("📋 Data Summary")
    
    if st.checkbox("Show detailed data summary"):
        st.write("**Dataset Overview:**")
        st.write(f"- Total records: {len(filtered_df):,}")
        st.write(f"- Date range: Analysis of current dataset")
        st.write(f"- Airlines covered: {filtered_df['airline'].nunique()}")
        
        st.write("**Sample Data:**")
        st.dataframe(filtered_df.head(), use_container_width=True)
        
        if st.button("Download Filtered Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="filtered_flight_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()

if __name__ == "__main__":

    main()

