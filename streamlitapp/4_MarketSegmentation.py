import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Market Segmentation", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .segment-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .cluster-insight {
        background: #f8f9fa;
        border-left: 4px solid #059669;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
    }
    .metric-highlight {
        background: #10B981;
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
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

@st.cache_data
def perform_clustering_analysis(df, n_clusters=4):
    """Perform advanced clustering analysis"""
    
    # Prepare clustering features
    clustering_features = ['price', 'duration', 'efficiency_score', 'is_direct', 'is_premium_time']
    available_features = [f for f in clustering_features if f in df.columns]
    
    if len(available_features) < 3:
        available_features = ['price', 'duration']
        if 'is_direct' in df.columns:
            available_features.append('is_direct')
    
    X = df[available_features].copy()
    X = X.fillna(X.mean())
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters
    inertias = []
    silhouette_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Choose optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    # Apply clustering with optimal k
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(X_scaled)
    
    # Add clusters to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    # Cluster analysis
    cluster_analysis = df_clustered.groupby('cluster').agg({
        'price': ['mean', 'count'],
        'duration': 'mean',
        'airline': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Mixed'
    }).round(2)
    
    cluster_analysis.columns = ['avg_price', 'size', 'avg_duration', 'dominant_airline']
    cluster_analysis = cluster_analysis.reset_index()
    
    # Name clusters based on characteristics
    cluster_names = {}
    for i, row in cluster_analysis.iterrows():
        cluster_id = row['cluster']
        price = row['avg_price']
        duration = row['avg_duration']
        
        if price > df['price'].quantile(0.75):
            if duration < df['duration'].median():
                cluster_names[cluster_id] = 'Premium Express'
            else:
                cluster_names[cluster_id] = 'Premium Comfort'
        elif price < df['price'].quantile(0.25):
            cluster_names[cluster_id] = 'Budget Conscious'
        else:
            if duration < df['duration'].median():
                cluster_names[cluster_id] = 'Value Seeker'
            else:
                cluster_names[cluster_id] = 'Standard'
    
    df_clustered['cluster_name'] = df_clustered['cluster'].map(cluster_names)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return df_clustered, cluster_analysis, cluster_names, X_pca, pca, (inertias, silhouette_scores, k_range, optimal_k)

def create_clustering_dashboard(df_clustered, cluster_analysis, cluster_names, X_pca, pca, optimization_data):
    """Create comprehensive clustering dashboard"""
    
    inertias, silhouette_scores, k_range, optimal_k = optimization_data
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Elbow Method & Silhouette Analysis',
            'Cluster Distribution',
            'PCA Cluster Projection', 
            'Cluster Characteristics',
            'Airlines Distribution by Cluster',
            'Performance Matrix'
        ],
        specs=[
            [{"secondary_y": True}, {"type": "pie"}],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "heatmap"}]
        ],
        vertical_spacing=0.08
    )
    
    # 1. Elbow Method
    fig.add_trace(
        go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(color=COLORS['primary']),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(k_range),
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color=COLORS['success']),
            marker=dict(size=8),
            yaxis='y2'
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Add optimal point marker
    fig.add_vline(
        x=optimal_k,
        line_dash="dash",
        line_color=COLORS['danger'],
        annotation_text=f"Optimal k={optimal_k}",
        row=1, col=1
    )
    
    # 2. Cluster distribution
    cluster_sizes = df_clustered['cluster_name'].value_counts()
    colors_clusters = px.colors.qualitative.Set3[:len(cluster_sizes)]
    
    fig.add_trace(
        go.Pie(
            labels=cluster_sizes.index,
            values=cluster_sizes.values,
            name='Cluster Distribution',
            marker_colors=colors_clusters,
            textinfo='label+percent+value',
            hovertemplate='<b>%{label}</b><br>Flights: %{value}<br>Percentage: %{percent}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. PCA Projection
    for i, cluster_name in enumerate(cluster_names.values()):
        cluster_mask = df_clustered['cluster_name'] == cluster_name
        
        fig.add_trace(
            go.Scatter(
                x=X_pca[cluster_mask, 0],
                y=X_pca[cluster_mask, 1],
                mode='markers',
                name=f'{cluster_name}',
                marker=dict(
                    size=8,
                    color=colors_clusters[i % len(colors_clusters)],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. Cluster characteristics
    fig.add_trace(
        go.Bar(
            x=cluster_analysis['cluster_name'],
            y=cluster_analysis['avg_price'],
            name='Avg Price',
            marker=dict(
                color=cluster_analysis['avg_price'],
                colorscale='RdYlGn_r',
                showscale=False
            ),
            text=[f'₹{x:,.0f}' for x in cluster_analysis['avg_price']],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    # 5. Airlines distribution by cluster
    airline_cluster_dist = df_clustered.groupby('cluster_name')['airline'].value_counts().unstack(fill_value=0)
    
    # Show top airlines only
    top_airlines = df_clustered['airline'].value_counts().head(4).index
    
    for airline in top_airlines:
        if airline in airline_cluster_dist.columns:
            fig.add_trace(
                go.Bar(
                    x=airline_cluster_dist.index,
                    y=airline_cluster_dist[airline],
                    name=airline,
                    marker_color=AIRLINE_PALETTE.get(airline, AIRLINE_PALETTE['Others'])
                ),
                row=3, col=1
            )
    
    # 6. Performance matrix
    performance_matrix = df_clustered.groupby(['cluster_name', 'airline']).size().unstack(fill_value=0)
    
    fig.add_trace(
        go.Heatmap(
            z=performance_matrix.values,
            x=performance_matrix.columns,
            y=performance_matrix.index,
            colorscale='Blues',
            showscale=False,
            text=performance_matrix.values,
            texttemplate='%{text}',
            hovertemplate='<b>%{y}</b><br>Airline: %{x}<br>Flights: %{z}<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Layout configuration
    fig.update_layout(
        height=1400,
        title={
            'text': 'Advanced Market Segmentation & Clustering Analysis',
            'x': 0.5,
            'font': {'size': 26, 'color': COLORS['primary']}
        },
        template='plotly_white',
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True, row=1, col=1)
    
    fig.update_xaxes(title_text=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", row=2, col=1)
    fig.update_yaxes(title_text=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", row=2, col=1)
    
    fig.update_xaxes(title_text="Cluster Segments", row=2, col=2)
    fig.update_yaxes(title_text="Average Price (₹)", row=2, col=2)
    
    fig.update_xaxes(title_text="Cluster Segments", row=3, col=1)
    fig.update_yaxes(title_text="Number of Flights", row=3, col=1)
    
    fig.update_xaxes(title_text="Airlines", row=3, col=2)
    fig.update_yaxes(title_text="Cluster Segments", row=3, col=2)
    
    return fig

def create_segment_profile_charts(df_clustered, cluster_names):
    """Create detailed segment profile charts"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Price Distribution by Segment',
            'Duration Patterns',
            'Service Quality by Segment',
            'Segment Size & Value'
        ]
    )
    
    colors_segments = px.colors.qualitative.Set3[:len(cluster_names)]
    
    # 1. Price distribution by segment
    for i, (cluster_id, segment_name) in enumerate(cluster_names.items()):
        segment_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        fig.add_trace(
            go.Histogram(
                x=segment_data['price'],
                name=segment_name,
                opacity=0.7,
                marker_color=colors_segments[i],
                nbinsx=20
            ),
            row=1, col=1
        )
    
    # 2. Duration patterns
    duration_by_segment = df_clustered.groupby('cluster_name')['duration'].mean().reset_index()
    
    fig.add_trace(
        go.Bar(
            x=duration_by_segment['cluster_name'],
            y=duration_by_segment['duration'],
            name='Average Duration',
            marker=dict(
                color=duration_by_segment['duration'],
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{x:.1f}h' for x in duration_by_segment['duration']],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # 3. Service quality (direct flights percentage)
    service_by_segment = df_clustered.groupby('cluster_name')['is_direct'].mean().reset_index()
    service_by_segment['direct_percentage'] = service_by_segment['is_direct'] * 100
    
    fig.add_trace(
        go.Bar(
            x=service_by_segment['cluster_name'],
            y=service_by_segment['direct_percentage'],
            name='Direct Flights %',
            marker=dict(
                color=service_by_segment['direct_percentage'],
                colorscale='RdYlGn',
                showscale=False
            ),
            text=[f'{x:.1f}%' for x in service_by_segment['direct_percentage']],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # 4. Segment size and average value
    segment_metrics = df_clustered.groupby('cluster_name').agg({
        'price': ['count', 'mean']
    })
    segment_metrics.columns = ['size', 'avg_price']
    segment_metrics = segment_metrics.reset_index()
    
    # Create bubble chart
    fig.add_trace(
        go.Scatter(
            x=segment_metrics['size'],
            y=segment_metrics['avg_price'],
            mode='markers+text',
            text=segment_metrics['cluster_name'],
            textposition='middle center',
            marker=dict(
                size=segment_metrics['size'] / 10,
                color=colors_segments[:len(segment_metrics)],
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            name='Segment Positioning'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title="Detailed Segment Profiles",
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    fig.update_xaxes(title_text="Segments", row=1, col=2, tickangle=45)
    fig.update_yaxes(title_text="Duration (hours)", row=1, col=2)
    
    fig.update_xaxes(title_text="Segments", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="Direct Flights (%)", row=2, col=1)
    
    fig.update_xaxes(title_text="Segment Size (# flights)", row=2, col=2)
    fig.update_yaxes(title_text="Average Price (₹)", row=2, col=2)
    
    return fig

def main():
    st.title("🎯 Market Segmentation Analysis")
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Segmentation Parameters")
    
    # Number of clusters
    n_clusters = st.sidebar.slider(
        "Number of Clusters",
        min_value=2,
        max_value=8,
        value=4,
        help="Choose the number of market segments to create"
    )
    
    # Feature selection for clustering
    st.sidebar.markdown("### Clustering Features")
    use_price = st.sidebar.checkbox("Price", value=True)
    use_duration = st.sidebar.checkbox("Duration", value=True)
    use_efficiency = st.sidebar.checkbox("Efficiency Score", value=True)
    use_direct = st.sidebar.checkbox("Direct Flights", value=True)
    use_premium = st.sidebar.checkbox("Premium Time", value=True)
    
    # Perform clustering analysis
    with st.spinner("Performing advanced clustering analysis..."):
        df_clustered, cluster_analysis, cluster_names, X_pca, pca, optimization_data = perform_clustering_analysis(df, n_clusters)
    
    # Display optimal clusters found
    optimal_k = optimization_data[3]
    st.success(f"✅ Clustering completed! Optimal number of clusters: {optimal_k}")
    
    # Segment overview cards
    st.markdown("## 🎯 Market Segments Overview")
    
    segment_cols = st.columns(len(cluster_names))
    
    for i, (cluster_id, segment_name) in enumerate(cluster_names.items()):
        with segment_cols[i]:
            segment_data = df_clustered[df_clustered['cluster'] == cluster_id]
            segment_size = len(segment_data)
            avg_price = segment_data['price'].mean()
            avg_duration = segment_data['duration'].mean()
            direct_rate = segment_data['is_direct'].mean() * 100
            
            st.markdown(f"""
            <div class="segment-card">
                <h4>{segment_name}</h4>
                <div class="metric-highlight">{segment_size:,} flights</div>
                <p>Avg Price: <strong>₹{avg_price:,.0f}</strong></p>
                <p>Avg Duration: <strong>{avg_duration:.1f}h</strong></p>
                <p>Direct: <strong>{direct_rate:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main clustering dashboard
    st.markdown("## 📊 Comprehensive Clustering Analysis")
    
    clustering_fig = create_clustering_dashboard(
        df_clustered, cluster_analysis, cluster_names, X_pca, pca, optimization_data
    )
    st.plotly_chart(clustering_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed segment profiles
    st.markdown("## 🔍 Detailed Segment Profiles")
    
    segment_profile_fig = create_segment_profile_charts(df_clustered, cluster_names)
    st.plotly_chart(segment_profile_fig, use_container_width=True)
    
    # Insights and analysis
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 💡 Segment Insights")
        
        # Segment characteristics
        for cluster_id, segment_name in cluster_names.items():
            segment_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            # Key characteristics
            avg_price = segment_data['price'].mean()
            size_pct = (len(segment_data) / len(df_clustered)) * 100
            top_airline = segment_data['airline'].mode()[0] if len(segment_data['airline'].mode()) > 0 else 'Mixed'
            
            st.markdown(f"""
            <div class="cluster-insight">
                <h4>{segment_name}</h4>
                <p><strong>Size:</strong> {len(segment_data):,} flights ({size_pct:.1f}%)</p>
                <p><strong>Price Profile:</strong> ₹{segment_data['price'].min():,.0f} - ₹{segment_data['price'].max():,.0f}</p>
                <p><strong>Top Airline:</strong> {top_airline}</p>
                <p><strong>Avg Duration:</strong> {segment_data['duration'].mean():.1f} hours</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## 📋 Segment Performance Table")
        
        # Create detailed segment table
        segment_details = []
        
        for cluster_id, segment_name in cluster_names.items():
            segment_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            segment_details.append({
                'Segment': segment_name,
                'Size': len(segment_data),
                'Size_%': f"{(len(segment_data) / len(df_clustered)) * 100:.1f}%",
                'Avg_Price': f"₹{segment_data['price'].mean():,.0f}",
                'Avg_Duration': f"{segment_data['duration'].mean():.1f}h",
                'Direct_Rate': f"{segment_data['is_direct'].mean() * 100:.1f}%",
                'Top_Airline': segment_data['airline'].mode()[0] if len(segment_data['airline'].mode()) > 0 else 'Mixed'
            })
        
        segment_df = pd.DataFrame(segment_details)
        st.dataframe(segment_df, use_container_width=True, hide_index=True)
        
        # Market recommendations
        st.markdown("## 🎯 Strategic Recommendations")
        
        # Find largest segment
        largest_segment = max(cluster_names.items(), 
                            key=lambda x: len(df_clustered[df_clustered['cluster'] == x[0]]))
        
        # Find most profitable segment (highest average price)
        most_profitable = max(cluster_names.items(),
                            key=lambda x: df_clustered[df_clustered['cluster'] == x[0]]['price'].mean())
        
        # Find most efficient segment (lowest price per hour)
        most_efficient = min(cluster_names.items(),
                           key=lambda x: df_clustered[df_clustered['cluster'] == x[0]]['efficiency_score'].mean())
        
        st.markdown(f"""
        <div class="cluster-insight">
            <strong>🎯 Target Segment:</strong> {largest_segment[1]}<br>
            Largest market segment with growth potential
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="cluster-insight">
            <strong>💰 Premium Segment:</strong> {most_profitable[1]}<br>
            Highest revenue per flight opportunity
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="cluster-insight">
            <strong>⚡ Value Segment:</strong> {most_efficient[1]}<br>
            Best efficiency and cost optimization
        </div>
        """, unsafe_allow_html=True)
    
    # Export segment data
    st.markdown("---")
    
    if st.button("📊 Export Segment Analysis"):
        # Prepare export data
        export_data = df_clustered[['airline', 'price', 'duration', 'cluster', 'cluster_name']].copy()
        
        # Convert to CSV
        csv_data = export_data.to_csv(index=False)
        
        st.download_button(
            label="Download Segmentation Results",
            data=csv_data,
            file_name="market_segmentation_results.csv",
            mime="text/csv"
        )
        
        st.success("Segment analysis ready for download!")

if __name__ == "__main__":
    main()