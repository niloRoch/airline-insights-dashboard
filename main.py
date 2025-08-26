import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Airlines Price Analysis Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/airlines-analysis',
        'Report a bug': 'https://github.com/yourusername/airlines-analysis/issues',
        'About': 'Comprehensive analysis of airline pricing patterns Delhi-Mumbai route'
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    .stMetric > div > div > div > div {
        font-size: 2rem;
    }
    
    .time-slot-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .segment-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Airline color palette
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
    """Load and cache the dataset"""
    try:
        # Try to load processed data first
        df = pd.read_csv('data/processed/cleaned_flights_data.csv')
    except FileNotFoundError:
        try:
            # Fallback to raw data
            df = pd.read_csv('data/raw/airlines_flights_data.csv')
            
            # Create additional features like in other pages
            df['efficiency_score'] = df['price'] / df['duration']
            df['is_direct'] = (df['stops'] == 'zero').astype(int)
            
            premium_times = ['Morning', 'Afternoon', 'Evening']
            df['is_premium_time'] = df['departure_time'].isin(premium_times).astype(int)
            
        except FileNotFoundError:
            st.error("Dataset not found. Please ensure the data files are in the correct location.")
            st.stop()
    
    return df

def create_airline_comparison_chart(df):
    """Create airline comparison visualization"""
    airline_metrics = df.groupby('airline').agg({
        'price': ['mean', 'count'],
        'duration': 'mean',
        'efficiency_score': 'mean',
        'is_direct': 'mean'
    }).round(2)
    
    airline_metrics.columns = ['avg_price', 'flight_count', 'avg_duration', 'efficiency', 'direct_rate']
    airline_metrics = airline_metrics.reset_index()
    airline_metrics['market_share'] = (airline_metrics['flight_count'] / airline_metrics['flight_count'].sum() * 100).round(1)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Market Share by Airlines',
            'Average Price Comparison',
            'Flight Efficiency (Price/Hour)',
            'Direct Flight Percentage'
        ]
    )
    
    # Market Share
    fig.add_trace(
        go.Bar(
            x=airline_metrics['airline'],
            y=airline_metrics['market_share'],
            name='Market Share',
            marker_color=[AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']) 
                         for airline in airline_metrics['airline']],
            text=[f'{x}%' for x in airline_metrics['market_share']],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Average Price
    fig.add_trace(
        go.Bar(
            x=airline_metrics['airline'],
            y=airline_metrics['avg_price'],
            name='Avg Price',
            marker_color=[AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']) 
                         for airline in airline_metrics['airline']],
            text=[f'₹{x:,.0f}' for x in airline_metrics['avg_price']],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Efficiency
    fig.add_trace(
        go.Bar(
            x=airline_metrics['airline'],
            y=airline_metrics['efficiency'],
            name='Efficiency',
            marker_color=[AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']) 
                         for airline in airline_metrics['airline']],
            text=[f'₹{x:.0f}/h' for x in airline_metrics['efficiency']],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # Direct Flight Rate
    fig.add_trace(
        go.Bar(
            x=airline_metrics['airline'],
            y=airline_metrics['direct_rate'] * 100,
            name='Direct Rate',
            marker_color=[AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']) 
                         for airline in airline_metrics['airline']],
            text=[f'{x:.1f}%' for x in airline_metrics['direct_rate'] * 100],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_temporal_analysis_chart(df):
    """Create temporal analysis visualization"""
    time_analysis = df.groupby('departure_time').agg({
        'price': ['mean', 'count'],
        'duration': 'mean',
        'efficiency_score': 'mean',
        'is_direct': 'mean'
    }).round(2)
    
    time_analysis.columns = ['avg_price', 'flight_count', 'avg_duration', 'efficiency', 'direct_rate']
    time_analysis = time_analysis.reset_index()
    
    # Order time slots logically
    time_order = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night']
    time_analysis['departure_time'] = pd.Categorical(time_analysis['departure_time'], categories=time_order, ordered=True)
    time_analysis = time_analysis.sort_values('departure_time')
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Flights by Time of Day',
            'Average Price by Time',
            'Efficiency by Time of Day',
            'Direct Flights by Time'
        ]
    )
    
    # Flight Count
    fig.add_trace(
        go.Bar(
            x=time_analysis['departure_time'],
            y=time_analysis['flight_count'],
            name='Flight Count',
            marker_color='lightblue',
            text=time_analysis['flight_count'],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Average Price
    fig.add_trace(
        go.Bar(
            x=time_analysis['departure_time'],
            y=time_analysis['avg_price'],
            name='Avg Price',
            marker_color='lightcoral',
            text=[f'₹{x:,.0f}' for x in time_analysis['avg_price']],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Efficiency
    fig.add_trace(
        go.Bar(
            x=time_analysis['departure_time'],
            y=time_analysis['efficiency'],
            name='Efficiency',
            marker_color='lightgreen',
            text=[f'₹{x:.0f}/h' for x in time_analysis['efficiency']],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # Direct Flight Rate
    fig.add_trace(
        go.Bar(
            x=time_analysis['departure_time'],
            y=time_analysis['direct_rate'] * 100,
            name='Direct Rate',
            marker_color='lightsalmon',
            text=[f'{x:.1f}%' for x in time_analysis['direct_rate'] * 100],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=False)
    
    return fig

def main():
    """Main application function"""
    
    # Load data
    df = load_data()
    
    # Header
    st.markdown('<h1 class="main-header">✈️ Airlines Price Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 3rem;">Comprehensive Statistical Analysis of Delhi-Mumbai Flight Pricing Patterns</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f4e79/ffffff?text=Airlines+Analytics", use_column_width=True)
        
        st.markdown("## 📊 Dashboard Navigation")
        page = st.radio(
            "Select Analysis Section:",
            ["Overview", "Airlines Analysis", "Temporal Analysis", "Market Segmentation", "Recommendations"]
        )
        
        st.markdown("---")
        st.markdown("## 🎯 Data Filters")
        
        # Quick filters
        selected_airlines = st.multiselect(
            "Select Airlines",
            options=df['airline'].unique(),
            default=df['airline'].unique()
        )
        
        price_range = st.slider(
            "Price Range (₹)",
            min_value=int(df['price'].min()),
            max_value=int(df['price'].max()),
            value=(int(df['price'].min()), int(df['price'].max()))
        )
        
        time_slots = st.multiselect(
            "Departure Time",
            options=df['departure_time'].unique(),
            default=df['departure_time'].unique()
        )
        
        flight_type = st.selectbox(
            "Flight Type",
            options=["All", "Direct Only", "With Stops"]
        )
        
        # Apply filters
        filtered_df = df[
            (df['airline'].isin(selected_airlines)) &
            (df['price'] >= price_range[0]) &
            (df['price'] <= price_range[1]) &
            (df['departure_time'].isin(time_slots))
        ]
        
        if flight_type == "Direct Only":
            filtered_df = filtered_df[filtered_df['stops'] == 'zero']
        elif flight_type == "With Stops":
            filtered_df = filtered_df[filtered_df['stops'] != 'zero']
        
        st.markdown(f"**Filtered Records:** {len(filtered_df):,} / {len(df):,}")
    
    # Display different pages based on selection
    if page == "Overview":
        display_overview(filtered_df, df)
    elif page == "Airlines Analysis":
        display_airlines_analysis(filtered_df)
    elif page == "Temporal Analysis":
        display_temporal_analysis(filtered_df)
    elif page == "Market Segmentation":
        display_market_segmentation(filtered_df)
    elif page == "Recommendations":
        display_recommendations(filtered_df)

def display_overview(filtered_df, original_df):
    """Display overview dashboard"""
    
    # Key Metrics Row
    st.markdown("## 📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_price = filtered_df['price'].mean()
        st.metric(
            label="Average Price",
            value=f"₹{avg_price:,.0f}",
            delta=f"{((avg_price - original_df['price'].mean()) / original_df['price'].mean() * 100):+.1f}%"
        )
    
    with col2:
        total_flights = len(filtered_df)
        st.metric(
            label="Total Flights",
            value=f"{total_flights:,}",
            delta=f"{total_flights - len(original_df)}"
        )
    
    with col3:
        direct_flights_pct = (filtered_df['stops'] == 'zero').mean() * 100
        st.metric(
            label="Direct Flights",
            value=f"{direct_flights_pct:.1f}%",
            delta=f"{direct_flights_pct - (original_df['stops'] == 'zero').mean() * 100:+.1f}%"
        )
    
    with col4:
        avg_duration = filtered_df['duration'].mean()
        st.metric(
            label="Avg Duration",
            value=f"{avg_duration:.1f}h",
            delta=f"{avg_duration - original_df['duration'].mean():+.1f}h"
        )
    
    with col5:
        unique_airlines = filtered_df['airline'].nunique()
        st.metric(
            label="Airlines",
            value=f"{unique_airlines}",
            delta=f"{unique_airlines - original_df['airline'].nunique()}"
        )
    
    st.markdown("---")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Price Analysis", "🏢 Airlines Overview", "⏰ Time Analysis"])
    
    with tab1:
        st.markdown("### Price Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution histogram
            fig_hist = px.histogram(
                filtered_df, 
                x='price', 
                nbins=30, 
                title='Flight Price Distribution',
                labels={'price': 'Price (₹)', 'count': 'Number of Flights'},
                color_discrete_sequence=['#636EFA']
            )
            fig_hist.add_vline(x=filtered_df['price'].mean(), line_dash="dash", line_color="red", 
                              annotation_text=f"Mean: ₹{filtered_df['price'].mean():,.0f}")
            fig_hist.add_vline(x=filtered_df['price'].median(), line_dash="dash", line_color="green", 
                              annotation_text=f"Median: ₹{filtered_df['price'].median():,.0f}")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot by airline
            fig_box = px.box(
                filtered_df, 
                x='airline', 
                y='price',
                title='Price Distribution by Airline',
                labels={'price': 'Price (₹)', 'airline': 'Airline'},
                color='airline',
                color_discrete_map=AIRLINE_PALETTE
            )
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Statistical summary
        st.markdown("### Statistical Summary")
        stats_df = filtered_df.groupby('airline')['price'].agg(['count', 'mean', 'std', 'min', 'max', 'median']).round(0)
        stats_df.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Median']
        st.dataframe(stats_df, use_container_width=True)
    
    with tab2:
        st.markdown("### Airlines Performance Overview")
        
        # Airline market share
        airline_counts = filtered_df['airline'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                values=airline_counts.values,
                names=airline_counts.index,
                title='Market Share by Number of Flights',
                color=airline_counts.index,
                color_discrete_map=AIRLINE_PALETTE
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Average price by airline
            avg_prices = filtered_df.groupby('airline')['price'].mean().sort_values(ascending=True)
            fig_bar = px.bar(
                x=avg_prices.values,
                y=avg_prices.index,
                orientation='h',
                title='Average Price by Airline',
                labels={'x': 'Average Price (₹)', 'y': 'Airline'},
                color=avg_prices.index,
                color_discrete_map=AIRLINE_PALETTE
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Price efficiency analysis
        st.markdown("### Price Efficiency Analysis")
        efficiency_df = filtered_df.groupby('airline').apply(
            lambda x: pd.Series({
                'avg_price': x['price'].mean(),
                'avg_duration': x['duration'].mean(),
                'price_per_hour': (x['price'] / x['duration']).mean(),
                'direct_flights_pct': (x['stops'] == 'zero').mean() * 100
            })
        ).round(2)
        
        st.dataframe(efficiency_df, use_container_width=True)
    
    with tab3:
        st.markdown("### Temporal Analysis Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price by departure time
            time_prices = filtered_df.groupby('departure_time')['price'].mean()
            fig_time = px.bar(
                x=time_prices.index,
                y=time_prices.values,
                title='Average Price by Departure Time',
                labels={'x': 'Departure Time', 'y': 'Average Price (₹)'},
                color=time_prices.index
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Days left vs price scatter
            fig_scatter = px.scatter(
                filtered_df,
                x='days_left',
                y='price',
                color='airline',
                title='Price vs Days Until Departure',
                labels={'days_left': 'Days Left', 'price': 'Price (₹)'},
                color_discrete_map=AIRLINE_PALETTE
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlation analysis
        st.markdown("### Correlation Analysis")
        corr_data = filtered_df[['price', 'duration', 'days_left']].corr()
        
        fig_corr = px.imshow(
            corr_data,
            title='Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

def display_airlines_analysis(filtered_df):
    """Display airlines analysis"""
    st.markdown("## 🏢 Airlines Performance Analysis")
    
    # Create comprehensive airline comparison chart
    airline_fig = create_airline_comparison_chart(filtered_df)
    st.plotly_chart(airline_fig, use_container_width=True)
    
    # Top performers section
    st.markdown("## 🏆 Top Performers")
    
    col1, col2, col3 = st.columns(3)
    
    # Market leader
    market_share = filtered_df['airline'].value_counts(normalize=True)
    top_airline_market = market_share.index[0]
    top_market_share = market_share.iloc[0] * 100
    
    with col1:
        st.markdown(f"""
        <div class="segment-card">
            <h4>🥇 Market Leader</h4>
            <h3>{top_airline_market}</h3>
            <p>{top_market_share:.1f}% market share</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Most efficient
    efficiency = filtered_df.groupby('airline')['efficiency_score'].mean()
    top_airline_efficiency = efficiency.idxmin()
    top_efficiency = efficiency.min()
    
    with col2:
        st.markdown(f"""
        <div class="segment-card">
            <h4>⚡ Most Efficient</h4>
            <h3>{top_airline_efficiency}</h3>
            <p>₹{top_efficiency:.0f}/hour</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Best direct flight provider
    direct_rate = filtered_df.groupby('airline')['is_direct'].mean()
    top_airline_direct = direct_rate.idxmax()
    top_direct_rate = direct_rate.max() * 100
    
    with col3:
        st.markdown(f"""
        <div class="segment-card">
            <h4>🎯 Best Service</h4>
            <h3>{top_airline_direct}</h3>
            <p>{top_direct_rate:.1f}% direct flights</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed airline metrics
    st.markdown("## 📊 Detailed Airline Metrics")
    
    airline_metrics = filtered_df.groupby('airline').agg({
        'price': ['mean', 'std', 'count'],
        'duration': 'mean',
        'efficiency_score': 'mean',
        'is_direct': 'mean'
    }).round(2)
    
    airline_metrics.columns = ['Avg Price', 'Price Std Dev', 'Flight Count', 'Avg Duration', 'Efficiency', 'Direct Rate']
    airline_metrics['Market Share'] = (airline_metrics['Flight Count'] / airline_metrics['Flight Count'].sum() * 100).round(1)
    airline_metrics['Direct Rate'] = (airline_metrics['Direct Rate'] * 100).round(1)
    
    st.dataframe(airline_metrics, use_container_width=True)

def display_temporal_analysis(filtered_df):
    """Display temporal analysis"""
    st.markdown("## ⏰ Temporal Analysis")
    
    # Create temporal analysis chart
    temporal_fig = create_temporal_analysis_chart(filtered_df)
    st.plotly_chart(temporal_fig, use_container_width=True)
    
    # Peak hours analysis
    st.markdown("## 🔥 Peak Hours Analysis")
    
    time_volume = filtered_df['departure_time'].value_counts()
    peak_threshold = time_volume.quantile(0.75)
    peak_hours = time_volume[time_volume > peak_threshold].index.tolist()
    
    # Display peak hours
    peak_cols = st.columns(len(peak_hours) if len(peak_hours) <= 4 else 4)
    
    for i, time_slot in enumerate(peak_hours[:4]):
        with peak_cols[i]:
            volume = time_volume[time_slot]
            avg_price = filtered_df[filtered_df['departure_time'] == time_slot]['price'].mean()
            direct_rate = filtered_df[filtered_df['departure_time'] == time_slot]['is_direct'].mean() * 100
            
            st.markdown(f"""
            <div class="time-slot-card">
                <h4>{time_slot}</h4>
                <div style="background: #EF4444; color: white; padding: 0.2rem 0.5rem; border-radius: 20px; font-size: 0.8rem; font-weight: bold;">PEAK HOUR</div>
                <p><strong>{volume:,}</strong> flights</p>
                <p>Avg Price: <strong>₹{avg_price:,.0f}</strong></p>
                <p>Direct: <strong>{direct_rate:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Best value time slots
    st.markdown("## 💎 Best Value Time Slots")
    
    value_by_time = filtered_df.groupby('departure_time').agg({
        'price': 'mean',
        'is_direct': 'mean',
        'efficiency_score': 'mean'
    })
    
    # Calculate value score (lower price, higher direct rate = better value)
    value_by_time['value_score'] = (
        (1 - (value_by_time['price'] - value_by_time['price'].min()) / 
         (value_by_time['price'].max() - value_by_time['price'].min())) * 0.6 +
        value_by_time['is_direct'] * 0.4
    )
    
    best_value_times = value_by_time.nlargest(3, 'value_score')
    
    value_cols = st.columns(3)
    
    for i, (time_slot, data) in enumerate(best_value_times.iterrows()):
        with value_cols[i]:
            st.markdown(f"""
            <div class="time-slot-card" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%);">
                <h4>{time_slot}</h4>
                <div style="background: #F59E0B; color: white; padding: 0.2rem 0.5rem; border-radius: 20px; font-size: 0.8rem; font-weight: bold;">BEST VALUE</div>
                <p>Value Score: <strong>{data['value_score']:.3f}</strong></p>
                <p>Avg Price: <strong>₹{data['price']:,.0f}</strong></p>
                <p>Direct: <strong>{data['is_direct']*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

def display_market_segmentation(filtered_df):
    """Display market segmentation analysis"""
    st.markdown("## 🎯 Market Segmentation Analysis")
    
    # Simple segmentation based on price quantiles
    price_segments = pd.qcut(filtered_df['price'], q=3, labels=['Budget', 'Mid-Range', 'Premium'])
    filtered_df['price_segment'] = price_segments
    
    # Segment analysis
    segment_analysis = filtered_df.groupby('price_segment').agg({
        'price': ['mean', 'count'],
        'duration': 'mean',
        'is_direct': 'mean',
        'airline': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Mixed'
    }).round(2)
    
    segment_analysis.columns = ['Avg Price', 'Count', 'Avg Duration', 'Direct Rate', 'Top Airline']
    segment_analysis['Direct Rate'] = (segment_analysis['Direct Rate'] * 100).round(1)
    
    # Display segments
    st.markdown("### 📊 Price-based Market Segments")
    
    seg_cols = st.columns(3)
    
    segments = segment_analysis.reset_index()
    
    for i, (_, segment) in enumerate(segments.iterrows()):
        with seg_cols[i]:
            st.markdown(f"""
            <div class="segment-card">
                <h4>{segment['price_segment']}</h4>
                <p><strong>{segment['Count']:,}</strong> flights</p>
                <p>Avg Price: <strong>₹{segment['Avg Price']:,.0f}</strong></p>
                <p>Avg Duration: <strong>{segment['Avg Duration']:.1f}h</strong></p>
                <p>Direct: <strong>{segment['Direct Rate']:.1f}%</strong></p>
                <p>Top Airline: <strong>{segment['Top Airline']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Segment distribution visualization
    st.markdown("### 📈 Segment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_seg_pie = px.pie(
            values=segments['Count'],
            names=segments['price_segment'],
            title='Flight Distribution by Price Segment'
        )
        st.plotly_chart(fig_seg_pie, use_container_width=True)
    
    with col2:
        # Airline distribution within segments
        airline_by_segment = filtered_df.groupby(['price_segment', 'airline']).size().unstack(fill_value=0)
        fig_seg_bar = px.bar(
            airline_by_segment,
            title='Airline Distribution by Price Segment',
            barmode='stack'
        )
        st.plotly_chart(fig_seg_bar, use_container_width=True)
    
    # Detailed segment metrics
    st.markdown("### 📋 Segment Metrics Table")
    st.dataframe(segment_analysis, use_container_width=True)

def display_recommendations(filtered_df):
    """Display recommendations"""
    st.markdown("## 💡 Flight Recommendations")
    
    # Simple recommendation system based on score
    filtered_df['recommendation_score'] = (
        (1 - (filtered_df['price'] - filtered_df['price'].min()) / 
         (filtered_df['price'].max() - filtered_df['price'].min())) * 0.4 +
        (1 - (filtered_df['duration'] - filtered_df['duration'].min()) / 
         (filtered_df['duration'].max() - filtered_df['duration'].min())) * 0.3 +
        filtered_df['is_direct'] * 0.3
    )
    
    # Get top recommendations
    top_recommendations = filtered_df.nlargest(5, 'recommendation_score')
    
    st.markdown("### 🏆 Top Recommended Flights")
    
    for i, (_, flight) in enumerate(top_recommendations.iterrows(), 1):
        score_percent = flight['recommendation_score'] * 100
        
        if score_percent >= 80:
            tier = "Excellent"
            color = "#10B981"
        elif score_percent >= 60:
            tier = "Good"
            color = "#0EA5E9"
        elif score_percent >= 40:
            tier = "Fair"
            color = "#F59E0B"
        else:
            tier = "Poor"
            color = "#EF4444"
        
        st.markdown(f"""
        <div style="background: {color}; padding: 1.5rem; border-radius: 10px; color: white; margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4>#{i}. {flight['airline']} - {tier}</h4>
                    <p><strong>Time:</strong> {flight['departure_time']} | <strong>Duration:</strong> {flight['duration']:.1f}h</p>
                    <p><strong>Price:</strong> ₹{flight['price']:,.0f} | <strong>Type:</strong> {'Direct' if flight['is_direct'] else 'Connecting'}</p>
                </div>
                <div style="text-align: center;">
                    <div style="background: white; color: {color}; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; font-size: 1.2rem;">
                        {score_percent:.1f}%
                    </div>
                    <p style="margin: 0.5rem 0; font-size: 0.9rem;">Recommendation Score</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendation by travel profile
    st.markdown("### 👥 Recommendations by Travel Profile")
    
    profile_tabs = st.tabs(["💰 Budget", "⏱️ Time", "🎯 Convenience"])
    
    with profile_tabs[0]:
        budget_recs = filtered_df.nsmallest(5, 'price')
        for i, (_, flight) in enumerate(budget_recs.iterrows(), 1):
            st.markdown(f"""
            <div class="profile-card">
                <h5>#{i} {flight['airline']}</h5>
                <p><strong>₹{flight['price']:,.0f}</strong></p>
                <p>{flight['duration']:.1f}h | {flight['departure_time']}</p>
                <p>{'Direct' if flight['is_direct'] else 'Connecting'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with profile_tabs[1]:
        time_recs = filtered_df.nsmallest(5, 'duration')
        for i, (_, flight) in enumerate(time_recs.iterrows(), 1):
            st.markdown(f"""
            <div class="profile-card">
                <h5>#{i} {flight['airline']}</h5>
                <p><strong>{flight['duration']:.1f}h</strong></p>
                <p>₹{flight['price']:,.0f} | {flight['departure_time']}</p>
                <p>{'Direct' if flight['is_direct'] else 'Connecting'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with profile_tabs[2]:
        convenience_recs = filtered_df[filtered_df['is_direct'] == 1].nsmallest(5, 'price')
        for i, (_, flight) in enumerate(convenience_recs.iterrows(), 1):
            st.markdown(f"""
            <div class="profile-card">
                <h5>#{i} {flight['airline']}</h5>
                <p><strong>Direct Flight</strong></p>
                <p>₹{flight['price']:,.0f} | {flight['duration']:.1f}h</p>
                <p>{flight['departure_time']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Travel tips
    st.markdown("### 💡 Smart Travel Tips")
    
    tip_cols = st.columns(3)
    
    with tip_cols[0]:
        cheapest_time = filtered_df.groupby('departure_time')['price'].mean().idxmin()
        st.info(f"""
        **💰 Budget Tip:**\n
        Cheapest time to fly: **{cheapest_time}**
        """)
    
    with tip_cols[1]:
        fastest_airline = filtered_df.groupby('airline')['duration'].mean().idxmin()
        st.info(f"""
        **⏱️ Time Tip:**\n
        Fastest airline: **{fastest_airline}**
        """)
    
    with tip_cols[2]:
        best_value_airline = filtered_df.groupby('airline')['efficiency_score'].mean().idxmin()
        st.info(f"""
        **🎯 Value Tip:**\n
        Best value airline: **{best_value_airline}**
        """)

if __name__ == "__main__":
    main()
