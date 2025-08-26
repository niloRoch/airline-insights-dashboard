import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Overview - Airlines Analysis", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .insight-card {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    .warning-card {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the cleaned dataset"""
    try:
        df = pd.read_csv('data/processed/cleaned_flights_data.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'cleaned_flights_data.csv' is in the data/processed/ folder.")
        st.stop()

def calculate_statistical_summary(df):
    """Calculate comprehensive statistical summary"""
    summary = {
        'basic_stats': {
            'total_flights': len(df),
            'unique_airlines': df['airline'].nunique(),
            'price_mean': df['price'].mean(),
            'price_median': df['price'].median(),
            'price_std': df['price'].std(),
            'duration_mean': df['duration'].mean(),
            'days_left_mean': df['days_left'].mean()
        },
        'price_distribution': {
            'min_price': df['price'].min(),
            'max_price': df['price'].max(),
            'q1_price': df['price'].quantile(0.25),
            'q3_price': df['price'].quantile(0.75),
            'iqr_price': df['price'].quantile(0.75) - df['price'].quantile(0.25)
        },
        'categorical_distribution': {
            'direct_flights_pct': (df['stops'] == 'zero').mean() * 100,
            'morning_flights_pct': (df['departure_time'] == 'morning').mean() * 100,
            'evening_flights_pct': (df['departure_time'] == 'evening').mean() * 100
        }
    }
    return summary

def perform_normality_tests(data):
    """Perform multiple normality tests"""
    from scipy.stats import shapiro, normaltest, jarque_bera
    
    results = {}
    
    # Shapiro-Wilk (for smaller samples)
    if len(data) <= 5000:
        shapiro_stat, shapiro_p = shapiro(data)
        results['shapiro'] = {'stat': shapiro_stat, 'p_value': shapiro_p}
    
    # D'Agostino-Pearson
    dag_stat, dag_p = normaltest(data)
    results['dagostino'] = {'stat': dag_stat, 'p_value': dag_p}
    
    # Jarque-Bera
    jb_stat, jb_p = jarque_bera(data)
    results['jarque_bera'] = {'stat': jb_stat, 'p_value': jb_p}
    
    return results

def main():
    st.title("📊 Flight Data Overview")
    st.markdown("Comprehensive analysis of Delhi-Mumbai flight pricing data")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔧 Filters")
        
        # Airline filter
        selected_airlines = st.multiselect(
            "Select Airlines",
            options=df['airline'].unique(),
            default=df['airline'].unique()
        )
        
        # Price range filter
        price_range = st.slider(
            "Price Range (₹)",
            min_value=int(df['price'].min()),
            max_value=int(df['price'].max()),
            value=(int(df['price'].min()), int(df['price'].max())),
            step=100
        )
        
        # Duration filter
        duration_range = st.slider(
            "Duration Range (hours)",
            min_value=float(df['duration'].min()),
            max_value=float(df['duration'].max()),
            value=(float(df['duration'].min()), float(df['duration'].max())),
            step=0.1
        )
        
        # Stops filter
        selected_stops = st.selectbox(
            "Flight Type",
            options=['All', 'Direct Only', 'With Stops'],
            index=0
        )
    
    # Apply filters
    filtered_df = df[
        (df['airline'].isin(selected_airlines)) &
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1]) &
        (df['duration'] >= duration_range[0]) &
        (df['duration'] <= duration_range[1])
    ]
    
    if selected_stops == 'Direct Only':
        filtered_df = filtered_df[filtered_df['stops'] == 'zero']
    elif selected_stops == 'With Stops':
        filtered_df = filtered_df[filtered_df['stops'] != 'zero']
    
    # Calculate statistics
    stats_summary = calculate_statistical_summary(filtered_df)
    
    # Key Metrics Row
    st.markdown("## 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Flights",
            f"{stats_summary['basic_stats']['total_flights']:,}",
            delta=f"{len(filtered_df) - len(df)}" if len(filtered_df) != len(df) else None
        )
    
    with col2:
        st.metric(
            "Average Price",
            f"₹{stats_summary['basic_stats']['price_mean']:,.0f}",
            delta=f"₹{stats_summary['basic_stats']['price_mean'] - df['price'].mean():+,.0f}"
        )
    
    with col3:
        st.metric(
            "Median Price",
            f"₹{stats_summary['basic_stats']['price_median']:,.0f}",
            delta=f"₹{stats_summary['basic_stats']['price_median'] - df['price'].median():+,.0f}"
        )
    
    with col4:
        st.metric(
            "Price Std Dev",
            f"₹{stats_summary['basic_stats']['price_std']:,.0f}",
            delta=f"₹{stats_summary['basic_stats']['price_std'] - df['price'].std():+,.0f}"
        )
    
    with col5:
        st.metric(
            "Airlines Count",
            f"{stats_summary['basic_stats']['unique_airlines']}",
            delta=f"{filtered_df['airline'].nunique() - df['airline'].nunique()}"
        )
    
    # Main analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistical Summary", "📊 Distribution Analysis", "🔬 Normality Tests", "🎯 Business Insights"])
    
    with tab1:
        st.markdown("### Descriptive Statistics")
        
        # Comprehensive statistics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Statistics")
            price_stats = pd.DataFrame({
                'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3', 'IQR', 'CV%'],
                'Value': [
                    f"{len(filtered_df):,}",
                    f"₹{filtered_df['price'].mean():,.0f}",
                    f"₹{filtered_df['price'].median():,.0f}",
                    f"₹{filtered_df['price'].std():,.0f}",
                    f"₹{filtered_df['price'].min():,.0f}",
                    f"₹{filtered_df['price'].max():,.0f}",
                    f"₹{filtered_df['price'].quantile(0.25):,.0f}",
                    f"₹{filtered_df['price'].quantile(0.75):,.0f}",
                    f"₹{stats_summary['price_distribution']['iqr_price']:,.0f}",
                    f"{(filtered_df['price'].std()/filtered_df['price'].mean()*100):.1f}%"
                ]
            })
            st.dataframe(price_stats, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Duration & Booking Statistics")
            other_stats = pd.DataFrame({
                'Metric': ['Avg Duration', 'Min Duration', 'Max Duration', 'Avg Days Left', 'Direct Flights', 'Morning Departures'],
                'Value': [
                    f"{filtered_df['duration'].mean():.1f} hours",
                    f"{filtered_df['duration'].min():.1f} hours",
                    f"{filtered_df['duration'].max():.1f} hours",
                    f"{filtered_df['days_left'].mean():.1f} days",
                    f"{(filtered_df['stops'] == 'zero').mean()*100:.1f}%",
                    f"{(filtered_df['departure_time'] == 'morning').mean()*100:.1f}%"
                ]
            })
            st.dataframe(other_stats, use_container_width=True, hide_index=True)
        
        # Five number summary visualization
        st.markdown("#### Five Number Summary - Price Distribution")
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=filtered_df['price'],
            name="Price Distribution",
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))
        fig_box.update_layout(
            title="Price Distribution Box Plot",
            yaxis_title="Price (₹)",
            height=400
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab2:
        st.markdown("### Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price histogram with normal curve
            fig_hist = go.Figure()
            
            # Histogram
            fig_hist.add_trace(go.Histogram(
                x=filtered_df['price'],
                nbinsx=30,
                name='Observed',
                opacity=0.7,
                marker_color='lightblue'
            ))
            
            # Normal distribution overlay
            x_norm = np.linspace(filtered_df['price'].min(), filtered_df['price'].max(), 100)
            y_norm = stats.norm.pdf(x_norm, filtered_df['price'].mean(), filtered_df['price'].std())
            
            # Scale normal curve to match histogram
            y_norm_scaled = y_norm * len(filtered_df) * (filtered_df['price'].max() - filtered_df['price'].min()) / 30
            
            fig_hist.add_trace(go.Scatter(
                x=x_norm,
                y=y_norm_scaled,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', width=3)
            ))
            
            fig_hist.update_layout(
                title="Price Distribution vs Normal Curve",
                xaxis_title="Price (₹)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Q-Q Plot
            from scipy import stats
            
            # Generate Q-Q plot data
            sorted_data = np.sort(filtered_df['price'])
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
            sample_quantiles = np.percentile(sorted_data, np.linspace(1, 99, len(sorted_data)))
            
            fig_qq = go.Figure()
            
            # Q-Q scatter plot
            fig_qq.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Sample vs Theoretical',
                marker=dict(size=4, opacity=0.6)
            ))
            
            # Reference line
            min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
            max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
            fig_qq.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Normal',
                line=dict(color='red', dash='dash')
            ))
            
            fig_qq.update_layout(
                title="Q-Q Plot: Sample vs Normal Distribution",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                height=400
            )
            st.plotly_chart(fig_qq, use_container_width=True)
        
        # Distribution by categorical variables
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution by airline
            fig_violin = px.violin(
                filtered_df,
                x='airline',
                y='price',
                title='Price Distribution by Airline',
                box=True
            )
            fig_violin.update_xaxes(tickangle=45)
            st.plotly_chart(fig_violin, use_container_width=True)
        
        with col2:
            # Price distribution by departure time
            fig_box_time = px.box(
                filtered_df,
                x='departure_time',
                y='price',
                title='Price Distribution by Departure Time'
            )
            st.plotly_chart(fig_box_time, use_container_width=True)
    
    with tab3:
        st.markdown("### Normality Tests")
        
        # Perform normality tests on price data
        normality_results = perform_normality_tests(filtered_df['price'].dropna())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Test Results")
            
            test_results = []
            for test_name, result in normality_results.items():
                is_normal = result['p_value'] > 0.05
                test_results.append({
                    'Test': test_name.replace('_', ' ').title(),
                    'Statistic': f"{result['stat']:.4f}",
                    'P-value': f"{result['p_value']:.6f}",
                    'Normal?': "Yes" if is_normal else "No",
                    'Status': "✅" if is_normal else "❌"
                })
            
            results_df = pd.DataFrame(test_results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Interpretation
            normal_count = sum([1 for _, result in normality_results.items() if result['p_value'] > 0.05])
            total_tests = len(normality_results)
            
            if normal_count == total_tests:
                st.success("✅ All tests indicate normal distribution")
            elif normal_count == 0:
                st.error("❌ No tests indicate normal distribution")
            else:
                st.warning(f"⚠️ Mixed results: {normal_count}/{total_tests} tests indicate normality")
        
        with col2:
            st.markdown("#### Skewness and Kurtosis")
            
            skewness = stats.skew(filtered_df['price'])
            kurtosis = stats.kurtosis(filtered_df['price'])
            
            skew_kurt_df = pd.DataFrame({
                'Metric': ['Skewness', 'Kurtosis', 'Skewness Interpretation', 'Kurtosis Interpretation'],
                'Value': [
                    f"{skewness:.4f}",
                    f"{kurtosis:.4f}",
                    "Right-skewed" if skewness > 0.5 else "Left-skewed" if skewness < -0.5 else "Approximately symmetric",
                    "Heavy-tailed" if kurtosis > 1 else "Light-tailed" if kurtosis < -1 else "Normal-tailed"
                ]
            })
            st.dataframe(skew_kurt_df, use_container_width=True, hide_index=True)
            
            # Visual interpretation
            if abs(skewness) > 1:
                st.warning("⚠️ Highly skewed distribution")
            elif abs(skewness) > 0.5:
                st.info("ℹ️ Moderately skewed distribution")
            else:
                st.success("✅ Approximately symmetric distribution")
    
    with tab4:
        st.markdown("### Business Insights")
        
        # Generate dynamic insights
        insights = []
        
        # Price insights
        cheapest_airline = filtered_df.groupby('airline')['price'].mean().idxmin()
        most_expensive_airline = filtered_df.groupby('airline')['price'].mean().idxmax()
        price_diff = filtered_df[filtered_df['airline'] == most_expensive_airline]['price'].mean() - filtered_df[filtered_df['airline'] == cheapest_airline]['price'].mean()
        
        insights.append({
            'title': 'Price Leadership Analysis',
            'content': f"**Most Affordable**: {cheapest_airline} (₹{filtered_df[filtered_df['airline'] == cheapest_airline]['price'].mean():,.0f} average)\n\n**Premium Carrier**: {most_expensive_airline} (₹{filtered_df[filtered_df['airline'] == most_expensive_airline]['price'].mean():,.0f} average)\n\n**Price Gap**: ₹{price_diff:,.0f} ({price_diff/filtered_df[filtered_df['airline'] == cheapest_airline]['price'].mean()*100:.1f}% premium)",
            'type': 'info'
        })
        
        # Market concentration
        market_share = filtered_df['airline'].value_counts(normalize=True)
        top_airline = market_share.index[0]
        top_share = market_share.iloc[0] * 100
        
        insights.append({
            'title': 'Market Concentration',
            'content': f"**Market Leader**: {top_airline} ({top_share:.1f}% of flights)\n\n**Market Structure**: {'Concentrated' if top_share > 40 else 'Moderately concentrated' if top_share > 25 else 'Fragmented'} market\n\n**Competition Level**: {'Low' if market_share.iloc[0] > 0.5 else 'Moderate' if market_share.iloc[0] > 0.3 else 'High'}",
            'type': 'success' if top_share < 40 else 'warning'
        })
        
        # Pricing efficiency
        efficiency_data = filtered_df.groupby('airline').apply(lambda x: (x['price'] / x['duration']).mean())
        most_efficient = efficiency_data.idxmin()
        least_efficient = efficiency_data.idxmax()
        
        insights.append({
            'title': 'Pricing Efficiency Analysis',
            'content': f"**Most Efficient**: {most_efficient} (₹{efficiency_data[most_efficient]:.0f}/hour)\n\n**Least Efficient**: {least_efficient} (₹{efficiency_data[least_efficient]:.0f}/hour)\n\n**Efficiency Gap**: {efficiency_data[least_efficient]/efficiency_data[most_efficient]:.1f}x difference",
            'type': 'info'
        })
        
        # Direct flight analysis
        direct_flight_stats = filtered_df.groupby('airline').apply(lambda x: (x['stops'] == 'zero').mean() * 100)
        best_direct = direct_flight_stats.idxmax()
        
        insights.append({
            'title': 'Route Efficiency',
            'content': f"**Best Direct Route Coverage**: {best_direct} ({direct_flight_stats[best_direct]:.1f}% direct flights)\n\n**Overall Direct Flight Rate**: {(filtered_df['stops'] == 'zero').mean() * 100:.1f}%\n\n**Customer Preference**: {'Strong preference for direct flights' if (filtered_df['stops'] == 'zero').mean() > 0.7 else 'Mixed preference for route types'}",
            'type': 'success'
        })
        
        # Display insights
        for i, insight in enumerate(insights):
            if insight['type'] == 'success':
                st.success(f"**{insight['title']}**\n\n{insight['content']}")
            elif insight['type'] == 'warning':
                st.warning(f"**{insight['title']}**\n\n{insight['content']}")
            else:
                st.info(f"**{insight['title']}**\n\n{insight['content']}")
        
        # Recommendations
        st.markdown("#### Strategic Recommendations")
        
        recommendations = []
        
        if price_diff > 5000:
            recommendations.append("Consider price positioning strategy - significant premium opportunity exists")
        
        if (filtered_df['stops'] == 'zero').mean() > 0.8:
            recommendations.append("Focus on direct route optimization - strong customer preference detected")
        
        if filtered_df['price'].std() / filtered_df['price'].mean() > 0.3:
            recommendations.append("High price volatility detected - implement dynamic pricing strategy")
        
        if len(filtered_df['airline'].unique()) > 5:
            recommendations.append("Highly competitive market - differentiation strategy crucial")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
    
    # Data quality assessment
    st.markdown("---")
    st.markdown("## 📋 Data Quality Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Completeness")
        completeness = (1 - filtered_df.isnull().sum() / len(filtered_df)) * 100
        for col in completeness.index:
            if completeness[col] == 100:
                st.success(f"{col}: {completeness[col]:.1f}%")
            elif completeness[col] >= 95:
                st.warning(f"{col}: {completeness[col]:.1f}%")
            else:
                st.error(f"{col}: {completeness[col]:.1f}%")
    
    with col2:
        st.markdown("#### Outlier Detection")
        Q1 = filtered_df['price'].quantile(0.25)
        Q3 = filtered_df['price'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = filtered_df[(filtered_df['price'] < (Q1 - 1.5 * IQR)) | (filtered_df['price'] > (Q3 + 1.5 * IQR))]
        
        outlier_pct = len(outliers) / len(filtered_df) * 100
        
        if outlier_pct < 5:
            st.success(f"Outliers: {len(outliers)} ({outlier_pct:.1f}%)")
        elif outlier_pct < 10:
            st.warning(f"Outliers: {len(outliers)} ({outlier_pct:.1f}%)")
        else:
            st.error(f"Outliers: {len(outliers)} ({outlier_pct:.1f}%)")
    
    with col3:
        st.markdown("#### Data Consistency")
        duplicates = filtered_df.duplicated().sum()
        
        if duplicates == 0:
            st.success(f"Duplicates: {duplicates}")
        else:
            st.warning(f"Duplicates: {duplicates}")
        
        # Check for logical inconsistencies
        inconsistent = 0
        if (filtered_df['duration'] <= 0).any():
            inconsistent += 1
        if (filtered_df['price'] <= 0).any():
            inconsistent += 1
        if (filtered_df['days_left'] < 0).any():
            inconsistent += 1
        
        if inconsistent == 0:
            st.success("No logical inconsistencies")
        else:
            st.error(f"{inconsistent} logical inconsistencies found")

if __name__ == "__main__":
    main()