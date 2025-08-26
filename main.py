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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        # Try to load processed data first
        df = pd.read_csv('data/processed/cleaned_flights_data.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure the data files are in the correct location.")
        st.stop()

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
        st.markdown("Use the sidebar to navigate through different analysis sections:")
        st.markdown("""
        - **Overview**: Key metrics and summary statistics
        - **Airlines Analysis**: Compare performance across carriers
        - **Temporal Analysis**: Time-based pricing patterns
        - **Market Segmentation**: Customer clustering insights
        - **Recommendations**: Actionable business insights
        """)
        
        st.markdown("---")
        st.markdown("## 🎯 Quick Filters")
        
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
        
        # Apply filters
        filtered_df = df[
            (df['airline'].isin(selected_airlines)) &
            (df['price'] >= price_range[0]) &
            (df['price'] <= price_range[1])
        ]
        
        st.markdown(f"**Filtered Records:** {len(filtered_df):,} / {len(df):,}")
    
    # Key Metrics Row
    st.markdown("## 📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_price = filtered_df['price'].mean()
        st.metric(
            label="Average Price",
            value=f"₹{avg_price:,.0f}",
            delta=f"{((avg_price - df['price'].mean()) / df['price'].mean() * 100):+.1f}%"
        )
    
    with col2:
        total_flights = len(filtered_df)
        st.metric(
            label="Total Flights",
            value=f"{total_flights:,}",
            delta=f"{total_flights - len(df)}"
        )
    
    with col3:
        direct_flights_pct = (filtered_df['stops'] == 'zero').mean() * 100
        st.metric(
            label="Direct Flights",
            value=f"{direct_flights_pct:.1f}%",
            delta=f"{direct_flights_pct - (df['stops'] == 'zero').mean() * 100:+.1f}%"
        )
    
    with col4:
        avg_duration = filtered_df['duration'].mean()
        st.metric(
            label="Avg Duration",
            value=f"{avg_duration:.1f}h",
            delta=f"{avg_duration - df['duration'].mean():+.1f}h"
        )
    
    with col5:
        unique_airlines = filtered_df['airline'].nunique()
        st.metric(
            label="Airlines",
            value=f"{unique_airlines}",
            delta=f"{unique_airlines - df['airline'].nunique()}"
        )
    
    st.markdown("---")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Distribution", "🏢 Airlines Comparison", "⏰ Temporal Patterns", "🎯 Advanced Analytics"])
    
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
            fig_hist.add_vline(x=filtered_df['price'].mean(), line_dash="dash", line_color="red", annotation_text=f"Mean: ₹{filtered_df['price'].mean():,.0f}")
            fig_hist.add_vline(x=filtered_df['price'].median(), line_dash="dash", line_color="green", annotation_text=f"Median: ₹{filtered_df['price'].median():,.0f}")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot by airline
            fig_box = px.box(
                filtered_df, 
                x='airline', 
                y='price',
                title='Price Distribution by Airline',
                labels={'price': 'Price (₹)', 'airline': 'Airline'}
            )
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Statistical summary
        st.markdown("### Statistical Summary")
        stats_df = filtered_df.groupby('airline')['price'].agg(['count', 'mean', 'std', 'min', 'max', 'median']).round(0)
        stats_df.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Median']
        st.dataframe(stats_df, use_container_width=True)
    
    with tab2:
        st.markdown("### Airlines Performance Comparison")
        
        # Airline market share
        airline_counts = filtered_df['airline'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                values=airline_counts.values,
                names=airline_counts.index,
                title='Market Share by Number of Flights'
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
                labels={'x': 'Average Price (₹)', 'y': 'Airline'}
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
        st.markdown("### Temporal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price by departure time
            time_prices = filtered_df.groupby('departure_time')['price'].mean()
            fig_time = px.bar(
                x=time_prices.index,
                y=time_prices.values,
                title='Average Price by Departure Time',
                labels={'x': 'Departure Time', 'y': 'Average Price (₹)'}
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
                labels={'days_left': 'Days Left', 'price': 'Price (₹)'}
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
    
    with tab4:
        st.markdown("### Advanced Statistical Analysis")
        
        # ANOVA results
        from scipy.stats import f_oneway
        
        airline_groups = [filtered_df[filtered_df['airline'] == airline]['price'].values 
                         for airline in filtered_df['airline'].unique()]
        
        if len(airline_groups) > 1:
            f_stat, p_value = f_oneway(*airline_groups)
            
            st.markdown("#### ANOVA Test Results")
            st.markdown(f"**F-statistic:** {f_stat:.4f}")
            st.markdown(f"**P-value:** {p_value:.6f}")
            
            if p_value < 0.05:
                st.success("✅ Significant differences between airlines detected (p < 0.05)")
            else:
                st.info("ℹ️ No significant differences between airlines detected (p ≥ 0.05)")
        
        # Business insights
        st.markdown("#### Key Business Insights")
        
        insights = []
        
        # Price insights
        cheapest_airline = filtered_df.groupby('airline')['price'].mean().idxmin()
        most_expensive_airline = filtered_df.groupby('airline')['price'].mean().idxmax()
        
        insights.append(f"🏆 **Most Affordable:** {cheapest_airline} (₹{filtered_df[filtered_df['airline'] == cheapest_airline]['price'].mean():,.0f} avg)")
        insights.append(f"💎 **Premium Carrier:** {most_expensive_airline} (₹{filtered_df[filtered_df['airline'] == most_expensive_airline]['price'].mean():,.0f} avg)")
        
        # Duration insights
        fastest_avg = filtered_df.groupby('airline')['duration'].mean().idxmin()
        insights.append(f"⚡ **Fastest Average:** {fastest_avg} ({filtered_df[filtered_df['airline'] == fastest_avg]['duration'].mean():.1f}h avg)")
        
        # Direct flights insight
        most_direct = filtered_df.groupby('airline').apply(lambda x: (x['stops'] == 'zero').mean()).idxmax()
        direct_pct = filtered_df[filtered_df['airline'] == most_direct].apply(lambda x: (x['stops'] == 'zero').mean() * 100).iloc[0]
        insights.append(f"🎯 **Most Direct Routes:** {most_direct} ({direct_pct:.1f}% direct flights)")
        
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("## 📋 Dataset Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Data Quality")
        st.markdown(f"- **Total Records:** {len(df):,}")
        st.markdown(f"- **Missing Values:** {df.isnull().sum().sum()}")
        st.markdown(f"- **Duplicate Records:** {df.duplicated().sum()}")
        st.markdown(f"- **Data Completeness:** {((len(df) - df.isnull().sum().sum()) / (len(df) * len(df.columns)) * 100):.1f}%")
    
    with col2:
        st.markdown("### Business Metrics")
        st.markdown(f"- **Airlines Covered:** {df['airline'].nunique()}")
        st.markdown(f"- **Route:** Delhi → Mumbai")
        st.markdown(f"- **Price Range:** ₹{df['price'].min():,} - ₹{df['price'].max():,}")
        st.markdown(f"- **Duration Range:** {df['duration'].min():.1f}h - {df['duration'].max():.1f}h")
    
    with col3:
        st.markdown("### Analysis Scope")
        st.markdown("- ✅ Descriptive Statistics")
        st.markdown("- ✅ Inferential Testing")
        st.markdown("- ✅ Correlation Analysis")
        st.markdown("- ✅ Market Segmentation")
        st.markdown("- ✅ Predictive Modeling")

if __name__ == "__main__":

    main()


