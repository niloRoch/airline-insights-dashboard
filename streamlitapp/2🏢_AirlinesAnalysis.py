import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Airlines Analysis", page_icon="🏢", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .airline-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .performance-metric {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1E3A8A;
    }
    .top-performer {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
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
            
            df['price_category'] = pd.qcut(df['price'], 
                                         q=3, 
                                         labels=['Budget', 'Mid-Range', 'Premium'])
        except FileNotFoundError:
            st.error("Data file not found!")
            return None
    
    return df

def prepare_airline_metrics(df):
    """Prepare detailed metrics by airline"""
    airline_metrics = df.groupby('airline').agg({
        'price': ['mean', 'std', 'min', 'max', 'count'],
        'duration': ['mean', 'std'],
        'efficiency_score': ['mean', 'std'],
        'is_direct': ['sum', 'mean'],
        'is_premium_time': 'mean'
    }).round(2)
    
    airline_metrics.columns = [
        'price_mean', 'price_std', 'price_min', 'price_max', 'flight_count',
        'duration_mean', 'duration_std', 'efficiency_mean', 'efficiency_std',
        'direct_count', 'direct_rate', 'premium_rate'
    ]
    airline_metrics = airline_metrics.reset_index()
    
    # Market share
    airline_metrics['market_share'] = (
        airline_metrics['flight_count'] / airline_metrics['flight_count'].sum() * 100
    ).round(1)
    
    # Performance rankings
    airline_metrics['efficiency_rank'] = airline_metrics['efficiency_mean'].rank()
    airline_metrics['market_rank'] = airline_metrics['market_share'].rank(ascending=False)
    
    return airline_metrics

def create_comprehensive_airline_dashboard(df, airline_metrics):
    """Create comprehensive airline performance analysis"""
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Price Performance Matrix',
            'Market Share & Volume',
            'Operational Efficiency',
            'Service Quality Index',
            'Competitive Positioning',
            'Performance Radar'
        ],
        specs=[
            [{"type": "scatter"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatterpolar"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # 1. Price Performance Matrix
    for i, airline in enumerate(airline_metrics['airline']):
        color = AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others'])
        
        fig.add_trace(
            go.Scatter(
                x=[airline_metrics[airline_metrics['airline'] == airline]['price_mean'].iloc[0]],
                y=[airline_metrics[airline_metrics['airline'] == airline]['price_std'].iloc[0]],
                mode='markers+text',
                name=airline,
                text=[airline],
                textposition='top center',
                marker=dict(
                    size=airline_metrics[airline_metrics['airline'] == airline]['market_share'].iloc[0] * 2,
                    color=color,
                    opacity=0.7,
                    line=dict(width=2, color='white')
                ),
                hovertemplate=f'<b>{airline}</b><br>' +
                             'Avg Price: ₹%{x:,.0f}<br>' +
                             'Price Volatility: ₹%{y:.0f}<br>' +
                             f'Market Share: {airline_metrics[airline_metrics["airline"] == airline]["market_share"].iloc[0]:.1f}%' +
                             '<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. Market Share Pie Chart
    fig.add_trace(
        go.Pie(
            labels=airline_metrics['airline'],
            values=airline_metrics['market_share'],
            name='Market Share',
            marker_colors=[AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']) 
                          for airline in airline_metrics['airline']],
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Share: %{percent}<br>Flights: %{value:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Operational Efficiency
    fig.add_trace(
        go.Bar(
            x=airline_metrics['airline'],
            y=airline_metrics['efficiency_mean'],
            name='Efficiency Score',
            marker=dict(
                color=[AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']) 
                      for airline in airline_metrics['airline']],
                opacity=0.8
            ),
            text=[f'₹{x:.0f}/hr' for x in airline_metrics['efficiency_mean']],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # 4. Service Quality (Direct Flights Rate)
    fig.add_trace(
        go.Bar(
            x=airline_metrics['airline'],
            y=airline_metrics['direct_rate'] * 100,
            name='Direct Flights %',
            marker=dict(
                color=airline_metrics['direct_rate'] * 100,
                colorscale='RdYlGn',
                showscale=False
            ),
            text=[f'{x*100:.1f}%' for x in airline_metrics['direct_rate']],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    # 5. Competitive Positioning
    # Create attractiveness score for positioning
    attractiveness_score = (
        (1 - (airline_metrics['price_mean'] - airline_metrics['price_mean'].min()) / 
         (airline_metrics['price_mean'].max() - airline_metrics['price_mean'].min())) * 0.4 +
        airline_metrics['direct_rate'] * 0.3 +
        (1 - (airline_metrics['efficiency_mean'] - airline_metrics['efficiency_mean'].min()) /
         (airline_metrics['efficiency_mean'].max() - airline_metrics['efficiency_mean'].min())) * 0.3
    )
    
    fig.add_trace(
        go.Scatter(
            x=attractiveness_score,
            y=airline_metrics['market_share'],
            mode='markers+text',
            name='Positioning',
            text=airline_metrics['airline'],
            textposition='top center',
            marker=dict(
                size=airline_metrics['flight_count'] / 50,
                color=[AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']) 
                      for airline in airline_metrics['airline']],
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Attractiveness: %{x:.3f}<br>' +
                         'Market Share: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. Performance Radar for top 3 airlines
    top_3_airlines = airline_metrics.nlargest(3, 'flight_count')['airline'].tolist()
    
    categories = ['Price Score', 'Efficiency', 'Direct Flights', 'Market Share', 'Service']
    
    for airline in top_3_airlines:
        airline_data = airline_metrics[airline_metrics['airline'] == airline].iloc[0]
        
        # Normalize scores for radar chart (0-100 scale)
        values = [
            100 - ((airline_data['price_mean'] - airline_metrics['price_mean'].min()) / 
                   (airline_metrics['price_mean'].max() - airline_metrics['price_mean'].min()) * 100),
            100 - ((airline_data['efficiency_mean'] - airline_metrics['efficiency_mean'].min()) / 
                   (airline_metrics['efficiency_mean'].max() - airline_metrics['efficiency_mean'].min()) * 100),
            airline_data['direct_rate'] * 100,
            airline_data['market_share'] * 2,
            airline_data['premium_rate'] * 100
        ]
        
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=airline,
                line_color=AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']),
                opacity=0.6
            ),
            row=3, col=2
        )
    
    # Layout configuration
    fig.update_layout(
        height=1200,
        title={
            'text': '🏢 Comprehensive Airline Performance Dashboard',
            'x': 0.5,
            'font': {'size': 26, 'color': COLORS['primary']}
        },
        template='plotly_white',
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Average Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Price Volatility (₹)", row=1, col=1)
    
    fig.update_xaxes(title_text="Airlines", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="Efficiency Score (₹/hour)", row=2, col=1)
    
    fig.update_xaxes(title_text="Airlines", row=2, col=2, tickangle=45)
    fig.update_yaxes(title_text="Direct Flights (%)", row=2, col=2)
    
    fig.update_xaxes(title_text="Attractiveness Score", row=3, col=1)
    fig.update_yaxes(title_text="Market Share (%)", row=3, col=1)
    
    fig.update_polars(radialaxis=dict(visible=True, range=[0, 100]), row=3, col=2)
    
    return fig

def create_airline_comparison_chart(df, selected_airlines):
    """Create comparison chart for selected airlines"""
    
    filtered_df = df[df['airline'].isin(selected_airlines)]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Price Distribution Comparison',
            'Duration vs Price Scatter',
            'Flight Volume by Time',
            'Efficiency Comparison'
        ]
    )
    
    # 1. Price distribution by airline
    for airline in selected_airlines:
        airline_data = filtered_df[filtered_df['airline'] == airline]
        fig.add_trace(
            go.Histogram(
                x=airline_data['price'],
                name=airline,
                opacity=0.7,
                marker_color=AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']),
                nbinsx=20
            ),
            row=1, col=1
        )
    
    # 2. Duration vs Price scatter
    for airline in selected_airlines:
        airline_data = filtered_df[filtered_df['airline'] == airline]
        fig.add_trace(
            go.Scatter(
                x=airline_data['duration'],
                y=airline_data['price'],
                mode='markers',
                name=airline,
                marker=dict(
                    color=AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']),
                    opacity=0.6,
                    size=8
                )
            ),
            row=1, col=2
        )
    
    # 3. Flight volume by time
    time_volume = filtered_df.groupby(['departure_time', 'airline']).size().unstack(fill_value=0)
    
    for airline in selected_airlines:
        if airline in time_volume.columns:
            fig.add_trace(
                go.Bar(
                    x=time_volume.index,
                    y=time_volume[airline],
                    name=airline,
                    marker_color=AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others'])
                ),
                row=2, col=1
            )
    
    # 4. Efficiency comparison
    efficiency_data = filtered_df.groupby('airline')['efficiency_score'].mean().reset_index()
    
    fig.add_trace(
        go.Bar(
            x=efficiency_data['airline'],
            y=efficiency_data['efficiency_score'],
            name='Efficiency',
            marker=dict(
                color=[AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']) 
                      for airline in efficiency_data['airline']]
            ),
            text=[f'₹{x:.0f}/hr' for x in efficiency_data['efficiency_score']],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title="📊 Airline Comparison Analysis",
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    fig.update_xaxes(title_text="Duration (hours)", row=1, col=2)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=2)
    
    fig.update_xaxes(title_text="Time Slots", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="Number of Flights", row=2, col=1)
    
    fig.update_xaxes(title_text="Airlines", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency Score (₹/hour)", row=2, col=2)
    
    return fig

def main():
    st.title("🏢 Airlines Performance Analysis")
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Prepare airline metrics
    airline_metrics = prepare_airline_metrics(df)
    
    # Sidebar filters
    st.sidebar.markdown("## 🎛️ Analysis Filters")
    
    # Airline selection
    all_airlines = df['airline'].unique().tolist()
    selected_airlines = st.sidebar.multiselect(
        "Select Airlines for Comparison",
        all_airlines,
        default=all_airlines[:3],
        help="Choose airlines to compare in detailed analysis"
    )
    
    # Price range filter
    price_range = st.sidebar.slider(
        "Price Range (₹)",
        int(df['price'].min()),
        int(df['price'].max()),
        (int(df['price'].min()), int(df['price'].max()))
    )
    
    # Filter data based on selections
    filtered_df = df[
        (df['airline'].isin(selected_airlines if selected_airlines else all_airlines)) &
        (df['price'].between(price_range[0], price_range[1]))
    ]
    
    # Performance Summary Cards
    st.markdown("## 🏆 Performance Summary")
    
    # Top performers
    top_market = airline_metrics.nlargest(1, 'market_share').iloc[0]
    top_efficiency = airline_metrics.nsmallest(1, 'efficiency_mean').iloc[0]
    top_direct = airline_metrics.nlargest(1, 'direct_rate').iloc[0]
    
    perf_cols = st.columns(3)
    
    with perf_cols[0]:
        st.markdown(f"""
        <div class="top-performer">
            <h4>🥇 Market Leader</h4>
            <h3>{top_market['airline']}</h3>
            <p>{top_market['market_share']:.1f}% market share</p>
            <p>{top_market['flight_count']:,} flights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_cols[1]:
        st.markdown(f"""
        <div class="top-performer">
            <h4>⚡ Most Efficient</h4>
            <h3>{top_efficiency['airline']}</h3>
            <p>₹{top_efficiency['efficiency_mean']:.0f}/hour</p>
            <p>Lowest cost per hour</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_cols[2]:
        st.markdown(f"""
        <div class="top-performer">
            <h4>🎯 Best Service</h4>
            <h3>{top_direct['airline']}</h3>
            <p>{top_direct['direct_rate']*100:.1f}% direct flights</p>
            <p>Highest convenience score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Dashboard
    st.markdown("## 📊 Comprehensive Performance Analysis")
    
    airline_dashboard = create_comprehensive_airline_dashboard(df, airline_metrics)
    st.plotly_chart(airline_dashboard, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed Comparison
    if selected_airlines:
        st.markdown("## 🔍 Detailed Airline Comparison")
        
        comparison_chart = create_airline_comparison_chart(df, selected_airlines)
        st.plotly_chart(comparison_chart, use_container_width=True)
        
        # Comparison table
        st.markdown("### 📋 Performance Metrics Table")
        
        comparison_metrics = airline_metrics[airline_metrics['airline'].isin(selected_airlines)]
        comparison_metrics_display = comparison_metrics[[
            'airline', 'flight_count', 'market_share', 'price_mean', 
            'efficiency_mean', 'direct_rate', 'duration_mean'
        ]].round(2)
        
        st.dataframe(
            comparison_metrics_display,
            use_container_width=True,
            hide_index=True
        )
    
    # Detailed Performance Metrics
    st.markdown("---")
    st.markdown("## 📈 All Airlines Performance Metrics")
    
    # Performance rankings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏅 Market Share Rankings")
        market_ranking = airline_metrics.sort_values('market_share', ascending=False)
        for i, (_, airline) in enumerate(market_ranking.iterrows(), 1):
            st.markdown(f"""
            <div class="performance-metric">
                <strong>{i}. {airline['airline']}</strong><br>
                Market Share: {airline['market_share']:.1f}%<br>
                Flights: {airline['flight_count']:,}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚡ Efficiency Rankings")
        efficiency_ranking = airline_metrics.sort_values('efficiency_mean')
        for i, (_, airline) in enumerate(efficiency_ranking.iterrows(), 1):
            st.markdown(f"""
            <div class="performance-metric">
                <strong>{i}. {airline['airline']}</strong><br>
                Efficiency: ₹{airline['efficiency_mean']:.0f}/hour<br>
                Avg Price: ₹{airline['price_mean']:,.0f}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()