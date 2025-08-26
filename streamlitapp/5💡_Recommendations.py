import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Flight Recommendations", page_icon="💡", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .recommendation-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .score-badge {
        background: #F59E0B;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
    }
    .profile-card {
        background: #f8f9fa;
        border: 2px solid #1E3A8A;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .tier-excellent { border-left: 5px solid #10B981; }
    .tier-good { border-left: 5px solid #0EA5E9; }
    .tier-fair { border-left: 5px solid #F59E0B; }
    .tier-poor { border-left: 5px solid #EF4444; }
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

@st.cache_data
def create_recommendation_system(df):
    """Create intelligent recommendation system"""
    
    df_rec = df.copy()
    
    # 1. Price Score (0-100, lower price = higher score)
    df_rec['price_score'] = ((df_rec['price'].max() - df_rec['price']) / 
                            (df_rec['price'].max() - df_rec['price'].min()) * 100)
    
    # 2. Duration Score (0-100, shorter duration = higher score)
    df_rec['duration_score'] = ((df_rec['duration'].max() - df_rec['duration']) / 
                               (df_rec['duration'].max() - df_rec['duration'].min()) * 100)
    
    # 3. Convenience Score
    df_rec['convenience_score'] = (
        df_rec['is_direct'] * 50 +  # Direct flight worth 50 points
        (1 - df_rec['is_premium_time']) * 30 +  # Non-premium time worth 30 points
        np.random.random(len(df_rec)) * 20  # Random factor for variability
    )
    
    # 4. Airline Reputation Score (based on market share and efficiency)
    airline_reputation = df_rec.groupby('airline').agg({
        'price': 'count',
        'efficiency_score': 'mean'
    })
    airline_reputation['market_share'] = (airline_reputation['price'] / 
                                         airline_reputation['price'].sum())
    airline_reputation['reputation_score'] = (
        airline_reputation['market_share'] * 50 +
        ((airline_reputation['efficiency_score'].max() - airline_reputation['efficiency_score']) /
         (airline_reputation['efficiency_score'].max() - airline_reputation['efficiency_score'].min())) * 50
    )
    
    df_rec = df_rec.merge(
        airline_reputation[['reputation_score']].reset_index(),
        on='airline'
    )
    
    # 5. Composite Recommendation Score
    weights = {
        'price': 0.35,
        'duration': 0.25, 
        'convenience': 0.20,
        'reputation': 0.20
    }
    
    df_rec['recommendation_score'] = (
        df_rec['price_score'] * weights['price'] +
        df_rec['duration_score'] * weights['duration'] +
        df_rec['convenience_score'] * weights['convenience'] +
        df_rec['reputation_score'] * weights['reputation']
    ).round(2)
    
    # Categorize recommendations
    df_rec['recommendation_tier'] = pd.cut(
        df_rec['recommendation_score'],
        bins=[0, 40, 70, 85, 100],
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )
    
    # Top recommendations by different profiles
    recommendations = {
        'budget_conscious': df_rec.nlargest(10, 'price_score'),
        'time_sensitive': df_rec.nlargest(10, 'duration_score'), 
        'convenience_seeker': df_rec.nlargest(10, 'convenience_score'),
        'balanced': df_rec.nlargest(10, 'recommendation_score')
    }
    
    return df_rec, recommendations

def create_recommendation_dashboard(df_rec, recommendations):
    """Create intelligent recommendation dashboard"""
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Top Recommendations Overview',
            'Recommendation Distribution',
            'Price vs Recommendation Score',
            'Multi-Criteria Analysis',
            'Profile-based Recommendations',
            'Recommendation Trends by Airline'
        ],
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "scatter"}, {"type": "scatterpolar"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.1
    )
    
    # 1. Top 10 Overall Recommendations
    top_10 = df_rec.nlargest(10, 'recommendation_score')
    
    fig.add_trace(
        go.Bar(
            x=top_10['recommendation_score'],
            y=[f"{row['airline']} ({row['departure_time']})" for _, row in top_10.iterrows()],
            orientation='h',
            name='Top Recommendations',
            marker=dict(
                color=top_10['recommendation_score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Score", x=0.48)
            ),
            text=[f"{score:.1f}" for score in top_10['recommendation_score']],
            textposition='inside'
        ),
        row=1, col=1
    )
    
    # 2. Distribution by tier
    tier_distribution = df_rec['recommendation_tier'].value_counts()
    tier_colors = {'Excellent': COLORS['success'], 'Good': COLORS['info'], 
                   'Fair': COLORS['warning'], 'Poor': COLORS['danger']}
    
    fig.add_trace(
        go.Pie(
            labels=tier_distribution.index,
            values=tier_distribution.values,
            name='Recommendation Tiers',
            marker_colors=[tier_colors.get(tier, COLORS['primary']) for tier in tier_distribution.index],
            textinfo='label+percent+value'
        ),
        row=1, col=2
    )
    
    # 3. Price vs Recommendation Score
    for tier in df_rec['recommendation_tier'].unique():
        if pd.notna(tier):
            tier_data = df_rec[df_rec['recommendation_tier'] == tier]
            
            fig.add_trace(
                go.Scatter(
                    x=tier_data['price'],
                    y=tier_data['recommendation_score'],
                    mode='markers',
                    name=tier,
                    marker=dict(
                        size=8,
                        color=tier_colors.get(tier, COLORS['primary']),
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Price: ₹%{x:,.0f}<br>' +
                                 'Score: %{y:.1f}<br>' +
                                 'Tier: ' + tier + '<br>' +
                                 '<extra></extra>',
                    text=tier_data['airline']
                ),
                row=2, col=1
            )
    
    # 4. Multi-Criteria Radar Chart (Top 3 Airlines)
    top_airlines_rec = df_rec.groupby('airline')['recommendation_score'].mean().nlargest(3)
    
    criteria = ['Price Score', 'Duration Score', 'Convenience Score', 'Reputation Score']
    
    for i, airline in enumerate(top_airlines_rec.index):
        airline_data = df_rec[df_rec['airline'] == airline]
        
        values = [
            airline_data['price_score'].mean(),
            airline_data['duration_score'].mean(), 
            airline_data['convenience_score'].mean(),
            airline_data['reputation_score'].mean()
        ]
        
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=criteria,
                fill='toself',
                name=airline,
                line_color=AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']),
                opacity=0.6
            ),
            row=2, col=2
        )
    
    # 5. Profile-based Recommendations
    profiles = ['budget_conscious', 'time_sensitive', 'convenience_seeker', 'balanced']
    profile_scores = []
    
    for profile in profiles:
        if profile == 'budget_conscious':
            score = df_rec['price_score'].mean()
        elif profile == 'time_sensitive':
            score = df_rec['duration_score'].mean()
        elif profile == 'convenience_seeker':
            score = df_rec['convenience_score'].mean()
        else:
            score = df_rec['recommendation_score'].mean()
        
        profile_scores.append(score)
    
    fig.add_trace(
        go.Bar(
            x=profiles,
            y=profile_scores,
            name='Profile Scores',
            marker=dict(
                color=profile_scores,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{score:.1f}' for score in profile_scores],
            textposition='outside'
        ),
        row=3, col=1
    )
    
    # 6. Recommendation Trends by Airline
    airline_rec_trends = df_rec.groupby('airline').agg({
        'recommendation_score': 'mean'
    }).round(1)
    
    fig.add_trace(
        go.Bar(
            x=airline_rec_trends.index,
            y=airline_rec_trends['recommendation_score'],
            name='Overall Score',
            marker_color=[AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others']) 
                         for airline in airline_rec_trends.index],
            text=airline_rec_trends['recommendation_score'].values,
            textposition='outside'
        ),
        row=3, col=2
    )
    
    # Layout configuration
    fig.update_layout(
        height=1400,
        title={
            'text': 'Intelligent Flight Recommendation System',
            'x': 0.5,
            'font': {'size': 28, 'color': COLORS['primary']}
        },
        template='plotly_white',
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Recommendation Score", row=1, col=1)
    fig.update_yaxes(title_text="Flight Options", row=1, col=1)
    
    fig.update_xaxes(title_text="Price (₹)", row=2, col=1)
    fig.update_yaxes(title_text="Recommendation Score", row=2, col=1)
    
    fig.update_polars(radialaxis=dict(visible=True, range=[0, 100]), row=2, col=2)
    
    fig.update_xaxes(title_text="User Profiles", row=3, col=1, tickangle=45)
    fig.update_yaxes(title_text="Average Score", row=3, col=1)
    
    fig.update_xaxes(title_text="Airlines", row=3, col=2, tickangle=45)
    fig.update_yaxes(title_text="Recommendation Score", row=3, col=2)
    
    return fig

def create_personalized_recommendations(df_rec, user_profile, filters):
    """Create personalized recommendations based on user preferences"""
    
    filtered_df = df_rec.copy()
    
    # Apply filters
    if filters['price_range']:
        filtered_df = filtered_df[
            (filtered_df['price'] >= filters['price_range'][0]) & 
            (filtered_df['price'] <= filters['price_range'][1])
        ]
    
    if filters['airlines']:
        filtered_df = filtered_df[filtered_df['airline'].isin(filters['airlines'])]
    
    if filters['time_slots']:
        filtered_df = filtered_df[filtered_df['departure_time'].isin(filters['time_slots'])]
    
    if filters['direct_only']:
        filtered_df = filtered_df[filtered_df['is_direct'] == 1]
    
    # Calculate personalized scores based on profile
    if user_profile == 'budget_conscious':
        filtered_df['personalized_score'] = (
            filtered_df['price_score'] * 0.6 +
            filtered_df['duration_score'] * 0.2 +
            filtered_df['convenience_score'] * 0.1 +
            filtered_df['reputation_score'] * 0.1
        )
    elif user_profile == 'time_sensitive':
        filtered_df['personalized_score'] = (
            filtered_df['price_score'] * 0.1 +
            filtered_df['duration_score'] * 0.5 +
            filtered_df['convenience_score'] * 0.3 +
            filtered_df['reputation_score'] * 0.1
        )
    elif user_profile == 'convenience_seeker':
        filtered_df['personalized_score'] = (
            filtered_df['price_score'] * 0.2 +
            filtered_df['duration_score'] * 0.2 +
            filtered_df['convenience_score'] * 0.5 +
            filtered_df['reputation_score'] * 0.1
        )
    elif user_profile == 'quality_focused':
        filtered_df['personalized_score'] = (
            filtered_df['price_score'] * 0.1 +
            filtered_df['duration_score'] * 0.2 +
            filtered_df['convenience_score'] * 0.2 +
            filtered_df['reputation_score'] * 0.5
        )
    else:  # balanced
        filtered_df['personalized_score'] = filtered_df['recommendation_score']
    
    # Get top recommendations
    top_recommendations = filtered_df.nlargest(10, 'personalized_score')
    
    return top_recommendations

def main():
    st.title("💡 Intelligent Flight Recommendations")
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Create recommendation system
    with st.spinner("Building intelligent recommendation system..."):
        df_rec, recommendations = create_recommendation_system(df)
    
    st.success("✅ Recommendation system ready!")
    
    # Sidebar - User Profile Selection
    st.sidebar.markdown("## 👤 Your Travel Profile")
    
    user_profile = st.sidebar.selectbox(
        "Select your travel profile:",
        ['balanced', 'budget_conscious', 'time_sensitive', 'convenience_seeker', 'quality_focused'],
        help="Choose the profile that best describes your travel preferences"
    )
    
    profile_descriptions = {
        'balanced': '⚖️ Balanced - Equal weight to all factors',
        'budget_conscious': '💰 Budget Conscious - Price is the priority',
        'time_sensitive': '⏱️ Time Sensitive - Shortest duration preferred',
        'convenience_seeker': '🎯 Convenience Seeker - Direct flights priority',
        'quality_focused': '⭐ Quality Focused - Airline reputation matters'
    }
    
    st.sidebar.info(profile_descriptions[user_profile])
    
    # Sidebar - Filters
    st.sidebar.markdown("## 🎛️ Recommendation Filters")
    
    # Price range filter
    price_min, price_max = int(df['price'].min()), int(df['price'].max())
    price_range = st.sidebar.slider(
        "Price Range (₹)",
        price_min, price_max,
        (price_min, price_max),
        help="Filter flights by price range"
    )
    
    # Airline filter
    selected_airlines = st.sidebar.multiselect(
        "Preferred Airlines",
        df['airline'].unique().tolist(),
        default=df['airline'].unique().tolist(),
        help="Select airlines to include in recommendations"
    )
    
    # Time slot filter
    selected_time_slots = st.sidebar.multiselect(
        "Preferred Time Slots",
        df['departure_time'].unique().tolist(),
        default=df['departure_time'].unique().tolist(),
        help="Select preferred departure times"
    )
    
    # Direct flights only
    direct_only = st.sidebar.checkbox(
        "Direct flights only",
        value=False,
        help="Show only direct flights"
    )
    
    filters = {
        'price_range': price_range,
        'airlines': selected_airlines,
        'time_slots': selected_time_slots,
        'direct_only': direct_only
    }
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎯 Your Personalized Recommendations")
        
        # Get personalized recommendations
        personalized_recs = create_personalized_recommendations(df_rec, user_profile, filters)
        
        if len(personalized_recs) == 0:
            st.warning("No flights found matching your criteria. Try adjusting your filters.")
        else:
            # Display top recommendations
            for i, (_, flight) in enumerate(personalized_recs.head(5).iterrows(), 1):
                tier_class = f"tier-{flight['recommendation_tier'].lower()}" if pd.notna(flight['recommendation_tier']) else "tier-fair"
                
                st.markdown(f"""
                <div class="recommendation-card {tier_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4>#{i}. {flight['airline']}</h4>
                            <p><strong>Time:</strong> {flight['departure_time']} | <strong>Duration:</strong> {flight['duration']:.1f}h</p>
                            <p><strong>Price:</strong> ₹{flight['price']:,.0f} | <strong>Type:</strong> {'Direct' if flight['is_direct'] else 'Connecting'}</p>
                        </div>
                        <div style="text-align: center;">
                            <div class="score-badge">{flight['personalized_score']:.1f}</div>
                            <p style="margin: 0.5rem 0; font-size: 0.9rem;">{flight['recommendation_tier']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## 📊 Recommendation Stats")
        
        # Overall stats
        total_options = len(personalized_recs)
        avg_score = personalized_recs['personalized_score'].mean() if len(personalized_recs) > 0 else 0
        avg_price = personalized_recs['price'].mean() if len(personalized_recs) > 0 else 0
        
        st.metric("Available Options", f"{total_options:,}")
        st.metric("Average Score", f"{avg_score:.1f}")
        st.metric("Average Price", f"₹{avg_price:,.0f}")
        
        # Tier distribution
        if len(personalized_recs) > 0:
            st.markdown("### Recommendation Tiers")
            tier_dist = personalized_recs['recommendation_tier'].value_counts()
            for tier, count in tier_dist.items():
                percentage = (count / len(personalized_recs)) * 100
                st.markdown(f"**{tier}:** {count} ({percentage:.1f}%)")
    
    st.markdown("---")
    
    # Comprehensive dashboard
    st.markdown("## 📊 Comprehensive Recommendation Analysis")
    
    recommendation_fig = create_recommendation_dashboard(df_rec, recommendations)
    st.plotly_chart(recommendation_fig, use_container_width=True)
    
    # Profile-based recommendations
    st.markdown("---")
    st.markdown("## 👥 Recommendations by Travel Profile")
    
    profile_tabs = st.tabs(["💰 Budget", "⏱️ Time", "🎯 Convenience", "⚖️ Balanced"])
    
    profiles = ['budget_conscious', 'time_sensitive', 'convenience_seeker', 'balanced']
    
    for tab, profile in zip(profile_tabs, profiles):
        with tab:
            profile_recs = recommendations[profile].head(5)
            
            cols = st.columns(5)
            
            for i, (col, (_, flight)) in enumerate(zip(cols, profile_recs.iterrows())):
                with col:
                    score_key = {
                        'budget_conscious': 'price_score',
                        'time_sensitive': 'duration_score',
                        'convenience_seeker': 'convenience_score',
                        'balanced': 'recommendation_score'
                    }[profile]
                    
                    st.markdown(f"""
                    <div class="profile-card">
                        <h5>#{i+1} {flight['airline']}</h5>
                        <p><strong>₹{flight['price']:,.0f}</strong></p>
                        <p>{flight['duration']:.1f}h | {flight['departure_time']}</p>
                        <div class="score-badge">{flight[score_key]:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Insights and tips
    st.markdown("---")
    st.markdown("## 💡 Smart Travel Tips")
    
    tip_cols = st.columns(3)
    
    with tip_cols[0]:
        st.markdown("### 💰 Budget Tips")
        cheapest_time = df.groupby('departure_time')['price'].mean().idxmin()
        cheapest_airline = df.groupby('airline')['price'].mean().idxmin()
        
        st.info(f"""
        **Best Time:** {cheapest_time}
        
        **Budget Airline:** {cheapest_airline}
        
        **Tip:** Book connecting flights for lower prices
        """)
    
    with tip_cols[1]:
        st.markdown("### ⏱️ Time Tips")
        fastest_time = df.groupby('departure_time')['duration'].mean().idxmin()
        direct_rate = df['is_direct'].mean() * 100
        
        st.info(f"""
        **Fastest Time:** {fastest_time}
        
        **Direct Flights:** {direct_rate:.1f}% available
        
        **Tip:** Morning flights often have fewer delays
        """)
    
    with tip_cols[2]:
        st.markdown("### 🎯 Quality Tips")
        top_airline = df['airline'].value_counts().index[0]
        best_service_time = df.groupby('departure_time')['is_direct'].mean().idxmax()
        
        st.info(f"""
        **Top Airline:** {top_airline}
        
        **Best Service:** {best_service_time}
        
        **Tip:** Premium times offer better service
        """)
    
    # Export recommendations
    st.markdown("---")
    
    if st.button("📥 Export My Recommendations"):
        export_data = personalized_recs[['airline', 'departure_time', 'price', 'duration', 
                                        'personalized_score', 'recommendation_tier']].copy()
        
        csv_data = export_data.to_csv(index=False)
        
        st.download_button(
            label="Download Personalized Recommendations",
            data=csv_data,
            file_name=f"flight_recommendations_{user_profile}.csv",
            mime="text/csv"
        )
        
        st.success("Your personalized recommendations are ready for download!")

if __name__ == "__main__":
    main()