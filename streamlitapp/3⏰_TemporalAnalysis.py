import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Temporal Analysis", page_icon="⏰", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .time-slot-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .peak-indicator {
        background: #EF4444;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .trend-card {
        background: #f8f9fa;
        border-left: 4px solid #059669;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
    }
</style>
""", unsafe_allow_html=True)

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
            
        except FileNotFoundError:
            st.error("Data file not found!")
            return None
    
    return df

def prepare_temporal_analysis(df):
    """Prepare detailed temporal analysis"""
    
    temporal_analysis = df.groupby(['departure_time', 'airline']).agg({
        'price': ['mean', 'count', 'std'],
        'duration': 'mean',
        'efficiency_score': 'mean',
        'is_direct': 'sum'
    }).round(2)
    
    temporal_analysis.columns = [
        'avg_price', 'flight_count', 'price_volatility',
        'avg_duration', 'efficiency', 'direct_flights'
    ]
    temporal_analysis = temporal_analysis.reset_index()
    
    # Create price heatmap data
    price_heatmap_data = df.pivot_table(
        values='price',
        index='airline',
        columns='departure_time',
        aggfunc='mean',
        fill_value=0
    )
    
    # Logical time ordering
    time_order = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night']
    available_times = [t for t in time_order if t in price_heatmap_data.columns]
    if available_times:
        price_heatmap_data = price_heatmap_data[available_times]
    
    return temporal_analysis, price_heatmap_data

def create_temporal_intelligence_dashboard(df, temporal_analysis, price_heatmap_data):
    """Create intelligent temporal analysis dashboard"""
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Price Heatmap: Airlines vs Time Slots',
            'Temporal Demand Patterns',
            'Efficiency Trends Throughout Day',
            'Direct Flights Availability',
            'Peak Hours Analysis',
            'Time-based Value Score'
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # 1. Price heatmap
    fig.add_trace(
        go.Heatmap(
            z=price_heatmap_data.values,
            x=price_heatmap_data.columns,
            y=price_heatmap_data.index,
            colorscale='RdYlBu_r',
            text=price_heatmap_data.round(0).values.astype(int),
            texttemplate='₹%{text}',
            textfont={"size": 10},
            hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Avg Price: ₹%{z:,.0f}<extra></extra>',
            colorbar=dict(title="Average Price (₹)", x=0.48)
        ),
        row=1, col=1
    )
    
    # 2. Temporal demand patterns
    time_demand = df.groupby('departure_time').agg({
        'price': ['count', 'mean']
    }).round(2)
    time_demand.columns = ['flight_count', 'avg_price']
    time_demand = time_demand.reset_index()
    
    fig.add_trace(
        go.Scatter(
            x=time_demand['flight_count'],
            y=time_demand['avg_price'],
            mode='markers+lines+text',
            text=time_demand['departure_time'],
            textposition='top center',
            marker=dict(
                size=time_demand['flight_count'] / 10,
                color=time_demand['avg_price'],
                colorscale='Viridis',
                showscale=False,
                line=dict(width=2, color='white')
            ),
            line=dict(color=COLORS['primary'], width=2),
            name='Demand Pattern'
        ),
        row=1, col=2
    )
    
    # 3. Efficiency trends
    time_efficiency = df.groupby('departure_time')['efficiency_score'].mean().reset_index()
    
    fig.add_trace(
        go.Bar(
            x=time_efficiency['departure_time'],
            y=time_efficiency['efficiency_score'],
            name='Efficiency Score',
            marker=dict(
                color=time_efficiency['efficiency_score'],
                colorscale='RdYlGn_r',
                showscale=False
            ),
            text=[f'₹{x:.0f}/hr' for x in time_efficiency['efficiency_score']],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # 4. Direct flights availability
    direct_availability = df.groupby('departure_time').agg({
        'is_direct': ['sum', 'count']
    })
    direct_availability.columns = ['direct_count', 'total_count']
    direct_availability['direct_rate'] = (
        direct_availability['direct_count'] / direct_availability['total_count'] * 100
    ).round(1)
    direct_availability = direct_availability.reset_index()
    
    fig.add_trace(
        go.Bar(
            x=direct_availability['departure_time'],
            y=direct_availability['direct_rate'],
            name='Direct Flights %',
            marker_color=COLORS['success'],
            text=[f'{x:.1f}%' for x in direct_availability['direct_rate']],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    # 5. Peak hours analysis
    peak_analysis = df.groupby('departure_time').agg({
        'price': 'count'
    }).reset_index()
    peak_analysis.columns = ['departure_time', 'volume']
    
    # Calculate value score (inverse of price, high direct flights)
    value_score = df.groupby('departure_time').agg({
        'price': 'mean',
        'is_direct': 'mean'
    })
    value_score['value_score'] = (
        (1 - (value_score['price'] - value_score['price'].min()) / 
         (value_score['price'].max() - value_score['price'].min())) * 0.6 +
        value_score['is_direct'] * 0.4
    )
    peak_analysis = peak_analysis.merge(value_score.reset_index(), on='departure_time')
    
    # Identify peaks
    peak_threshold = peak_analysis['volume'].quantile(0.75)
    peak_analysis['is_peak'] = peak_analysis['volume'] > peak_threshold
    
    colors_peak = [COLORS['danger'] if is_peak else COLORS['info'] 
                   for is_peak in peak_analysis['is_peak']]
    
    fig.add_trace(
        go.Scatter(
            x=peak_analysis['volume'],
            y=peak_analysis['value_score'],
            mode='markers+text',
            text=peak_analysis['departure_time'],
            textposition='top center',
            marker=dict(
                size=15,
                color=colors_peak,
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            name='Peak Analysis'
        ),
        row=3, col=1
    )
    
    # 6. Time-based value score
    value_score_sorted = value_score.reset_index().sort_values('value_score', ascending=False)
    
    fig.add_trace(
        go.Bar(
            x=value_score_sorted['departure_time'],
            y=value_score_sorted['value_score'],
            name='Value Score',
            marker=dict(
                color=value_score_sorted['value_score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Score", x=1.02)
            ),
            text=[f'{x:.3f}' for x in value_score_sorted['value_score']],
            textposition='outside'
        ),
        row=3, col=2
    )
    
    # Layout configuration
    fig.update_layout(
        height=1200,
        title={
            'text': 'Temporal Intelligence Dashboard - Advanced Time Analytics',
            'x': 0.5,
            'font': {'size': 26, 'color': COLORS['primary']}
        },
        template='plotly_white',
        showlegend=False
    )
    
    # Update axes
    fig.update_yaxes(title_text="Airlines", row=1, col=1)
    fig.update_xaxes(title_text="Time Slots", row=1, col=1)
    
    fig.update_xaxes(title_text="Flight Volume", row=1, col=2)
    fig.update_yaxes(title_text="Average Price (₹)", row=1, col=2)
    
    fig.update_xaxes(title_text="Time Slots", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="Efficiency (₹/hour)", row=2, col=1)
    
    fig.update_xaxes(title_text="Time Slots", row=2, col=2, tickangle=45)
    fig.update_yaxes(title_text="Direct Flights (%)", row=2, col=2)
    
    fig.update_xaxes(title_text="Flight Volume", row=3, col=1)
    fig.update_yaxes(title_text="Value Score", row=3, col=1)
    
    fig.update_xaxes(title_text="Time Slots", row=3, col=2, tickangle=45)
    fig.update_yaxes(title_text="Value Score", row=3, col=2)
    
    return fig, peak_analysis

def create_time_series_analysis(df):
    """Create time series analysis charts"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Price Trends by Time Slot',
            'Flight Volume Distribution',
            'Efficiency Patterns',
            'Service Quality by Time'
        ]
    )
    
    # 1. Price trends by airline and time
    for airline in df['airline'].unique()[:5]:  # Top 5 airlines
        airline_data = df[df['airline'] == airline]
        time_prices = airline_data.groupby('departure_time')['price'].mean()
        
        fig.add_trace(
            go.Scatter(
                x=time_prices.index,
                y=time_prices.values,
                mode='lines+markers',
                name=airline,
                line=dict(color=AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others'])),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    
    # 2. Flight volume distribution
    volume_dist = df['departure_time'].value_counts()
    
    fig.add_trace(
        go.Bar(
            x=volume_dist.index,
            y=volume_dist.values,
            name='Flight Volume',
            marker=dict(
                color=volume_dist.values,
                colorscale='Blues',
                showscale=False
            ),
            text=volume_dist.values,
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # 3. Efficiency patterns
    efficiency_by_time = df.groupby('departure_time')['efficiency_score'].mean()
    
    fig.add_trace(
        go.Scatter(
            x=efficiency_by_time.index,
            y=efficiency_by_time.values,
            mode='lines+markers',
            name='Efficiency Trend',
            line=dict(color=COLORS['warning'], width=3),
            marker=dict(size=10),
            fill='tonexty'
        ),
        row=2, col=1
    )
    
    # 4. Service quality (direct flights percentage)
    service_quality = df.groupby('departure_time')['is_direct'].mean() * 100
    
    fig.add_trace(
        go.Bar(
            x=service_quality.index,
            y=service_quality.values,
            name='Direct Flights %',
            marker=dict(
                color=service_quality.values,
                colorscale='RdYlGn',
                showscale=False
            ),
            text=[f'{x:.1f}%' for x in service_quality.values],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title="Time Series Analysis",
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time Slots", row=1, col=1, tickangle=45)
    fig.update_yaxes(title_text="Average Price (₹)", row=1, col=1)
    
    fig.update_xaxes(title_text="Time Slots", row=1, col=2, tickangle=45)
    fig.update_yaxes(title_text="Number of Flights", row=1, col=2)
    
    fig.update_xaxes(title_text="Time Slots", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="Efficiency Score (₹/hour)", row=2, col=1)
    
    fig.update_xaxes(title_text="Time Slots", row=2, col=2, tickangle=45)
    fig.update_yaxes(title_text="Direct Flights (%)", row=2, col=2)
    
    return fig

def main():
    st.title("⏰ Temporal Analysis")
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Prepare temporal analysis
    temporal_analysis, price_heatmap_data = prepare_temporal_analysis(df)
    
    # Sidebar filters
    st.sidebar.markdown("## 🎛️ Time Filters")
    
    # Time slot selection
    all_time_slots = df['departure_time'].unique().tolist()
    selected_time_slots = st.sidebar.multiselect(
        "Select Time Slots",
        all_time_slots,
        default=all_time_slots,
        help="Choose time slots for detailed analysis"
    )
    
    # Airline selection for temporal comparison
    selected_airlines_temporal = st.sidebar.multiselect(
        "Select Airlines for Comparison",
        df['airline'].unique().tolist(),
        default=df['airline'].unique().tolist()[:3],
        help="Choose airlines to compare temporal patterns"
    )
    
    # Filter data
    filtered_df = df[
        (df['departure_time'].isin(selected_time_slots)) &
        (df['airline'].isin(selected_airlines_temporal))
    ]
    
    # Peak Hours Summary
    st.markdown("## 🔥 Peak Hours Analysis")
    
    # Calculate peak metrics
    time_volume = df['departure_time'].value_counts()
    peak_threshold = time_volume.quantile(0.75)
    peak_hours = time_volume[time_volume > peak_threshold].index.tolist()
    
    # Display peak hours
    peak_cols = st.columns(len(peak_hours) if len(peak_hours) <= 4 else 4)
    
    for i, time_slot in enumerate(peak_hours[:4]):
        with peak_cols[i]:
            volume = time_volume[time_slot]
            avg_price = df[df['departure_time'] == time_slot]['price'].mean()
            direct_rate = df[df['departure_time'] == time_slot]['is_direct'].mean() * 100
            
            st.markdown(f"""
            <div class="time-slot-card">
                <h4>{time_slot}</h4>
                <div class="peak-indicator">PEAK HOUR</div>
                <p><strong>{volume:,}</strong> flights</p>
                <p>Avg Price: <strong>₹{avg_price:,.0f}</strong></p>
                <p>Direct: <strong>{direct_rate:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main temporal dashboard
    st.markdown("## 📊 Comprehensive Temporal Intelligence")
    
    temporal_fig, peak_analysis = create_temporal_intelligence_dashboard(
        filtered_df, temporal_analysis, price_heatmap_data
    )
    st.plotly_chart(temporal_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Time series analysis
    st.markdown("## 📈 Time Series Patterns")
    
    time_series_fig = create_time_series_analysis(filtered_df)
    st.plotly_chart(time_series_fig, use_container_width=True)
    
    # Insights and recommendations
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 💡 Temporal Insights")
        
        # Best time slots
        best_value = peak_analysis.nlargest(3, 'value_score')
        st.markdown("### 🏆 Best Value Time Slots")
        
        for i, (_, slot_data) in enumerate(best_value.iterrows(), 1):
            st.markdown(f"""
            <div class="trend-card">
                <strong>{i}. {slot_data['departure_time']}</strong><br>
                Value Score: {slot_data['value_score']:.3f}<br>
                Volume: {slot_data['volume']:,} flights<br>
                Avg Price: ₹{slot_data['price']:,.0f}
            </div>
            """, unsafe_allow_html=True)
        
        # Efficiency insights
        st.markdown("### ⚡ Efficiency Patterns")
        efficiency_by_time = df.groupby('departure_time')['efficiency_score'].mean().sort_values()
        
        most_efficient = efficiency_by_time.index[0]
        least_efficient = efficiency_by_time.index[-1]
        
        st.markdown(f"""
        <div class="trend-card">
            <strong>Most Efficient:</strong> {most_efficient}<br>
            Score: ₹{efficiency_by_time[most_efficient]:.0f}/hour
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="trend-card">
            <strong>Least Efficient:</strong> {least_efficient}<br>
            Score: ₹{efficiency_by_time[least_efficient]:.0f}/hour
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## 📋 Time Slot Performance Table")
        
        # Create performance summary table
        performance_summary = df.groupby('departure_time').agg({
            'price': ['count', 'mean', 'std'],
            'duration': 'mean',
            'efficiency_score': 'mean',
            'is_direct': 'mean'
        }).round(2)
        
        performance_summary.columns = [
            'Flights', 'Avg_Price', 'Price_Std', 
            'Avg_Duration', 'Efficiency', 'Direct_Rate'
        ]
        
        # Add value score
        performance_summary['Value_Score'] = (
            (1 - (performance_summary['Avg_Price'] - performance_summary['Avg_Price'].min()) / 
             (performance_summary['Avg_Price'].max() - performance_summary['Avg_Price'].min())) * 0.6 +
            performance_summary['Direct_Rate'] * 0.4
        ).round(3)
        
        performance_summary = performance_summary.sort_values('Value_Score', ascending=False)
        
        st.dataframe(
            performance_summary,
            use_container_width=True
        )
        
        # Recommendations
        st.markdown("### 🎯 Recommendations")
        
        st.markdown("""
        <div class="trend-card">
            <strong>For Budget Travelers:</strong><br>
            Choose off-peak hours for lower prices
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="trend-card">
            <strong>For Business Travelers:</strong><br>
            Morning slots offer best direct flight availability
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="trend-card">
            <strong>For Flexible Travelers:</strong><br>
            Evening slots provide optimal value-price balance
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()