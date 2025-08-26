"""
Clustering Analysis Module

This module provides comprehensive clustering analysis for market segmentation
of the Airlines Dataset, including K-means clustering, optimal cluster selection,
and cluster profiling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class MarketSegmentation:
    """
    Main class for market segmentation analysis using clustering techniques.
    """
    
    def __init__(self, data: pd.DataFrame, features: List[str] = None):
        """
        Initialize the market segmentation analyzer.
        
        Args:
            data: DataFrame containing flight information
            features: List of features to use for clustering
        """
        self.data = data.copy()
        self.original_data = data.copy()
        
        if features is None:
            self.features = ['price', 'duration', 'days_left']
        else:
            self.features = features
            
        self.scaled_data = None
        self.scaler = None
        self.cluster_labels = None
        self.optimal_k = None
        self.clustering_results = {}
        
    def prepare_data(self, scaling_method: str = 'standard') -> np.ndarray:
        """
        Prepare and scale data for clustering.
        
        Args:
            scaling_method: 'standard', 'minmax', or 'none'
            
        Returns:
            Scaled feature array
        """
        # Select features and remove missing values
        feature_data = self.data[self.features].dropna()
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'none':
            self.scaler = None
        else:
            raise ValueError("scaling_method must be 'standard', 'minmax', or 'none'")
        
        if self.scaler is not None:
            self.scaled_data = self.scaler.fit_transform(feature_data)
        else:
            self.scaled_data = feature_data.values
            
        # Update data to match cleaned data
        self.data = self.data.loc[feature_data.index].copy()
        
        return self.scaled_data
    
    def find_optimal_k(self, k_range: range = None, methods: List[str] = None) -> Dict[str, Any]:
        """
        Find optimal number of clusters using multiple methods.
        
        Args:
            k_range: Range of k values to test
            methods: List of methods to use ['elbow', 'silhouette', 'calinski', 'davies_bouldin']
            
        Returns:
            Dictionary with optimization results
        """
        if self.scaled_data is None:
            self.prepare_data()
            
        if k_range is None:
            k_range = range(2, 11)
            
        if methods is None:
            methods = ['elbow', 'silhouette', 'calinski', 'davies_bouldin']
        
        results = {
            'k_values': list(k_range),
            'metrics': {}
        }
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        for k in k_range:
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)
            
            # Calculate metrics
            if 'elbow' in methods:
                inertias.append(kmeans.inertia_)
            
            if 'silhouette' in methods:
                silhouette_scores.append(silhouette_score(self.scaled_data, labels))
            
            if 'calinski' in methods:
                calinski_scores.append(calinski_harabasz_score(self.scaled_data, labels))
            
            if 'davies_bouldin' in methods:
                davies_bouldin_scores.append(davies_bouldin_score(self.scaled_data, labels))
        
        # Store results
        if 'elbow' in methods:
            results['metrics']['inertia'] = inertias
        if 'silhouette' in methods:
            results['metrics']['silhouette'] = silhouette_scores
        if 'calinski' in methods:
            results['metrics']['calinski_harabasz'] = calinski_scores
        if 'davies_bouldin' in methods:
            results['metrics']['davies_bouldin'] = davies_bouldin_scores
        
        # Find optimal k for each method
        optimal_k_by_method = {}
        
        if 'elbow' in methods:
            # Elbow method using rate of change
            elbow_k = self._find_elbow_point(list(k_range), inertias)
            optimal_k_by_method['elbow'] = elbow_k
        
        if 'silhouette' in methods:
            optimal_k_by_method['silhouette'] = k_range[np.argmax(silhouette_scores)]
        
        if 'calinski' in methods:
            optimal_k_by_method['calinski_harabasz'] = k_range[np.argmax(calinski_scores)]
        
        if 'davies_bouldin' in methods:
            optimal_k_by_method['davies_bouldin'] = k_range[np.argmin(davies_bouldin_scores)]
        
        results['optimal_k_by_method'] = optimal_k_by_method
        
        # Choose overall optimal k (using silhouette score as primary)
        if 'silhouette' in methods:
            self.optimal_k = optimal_k_by_method['silhouette']
        else:
            # Use most common optimal k
            k_values = list(optimal_k_by_method.values())
            self.optimal_k = max(set(k_values), key=k_values.count)
        
        results['recommended_k'] = self.optimal_k
        
        self.clustering_results['optimization'] = results
        return results
    
    def perform_clustering(self, k: int = None, algorithm: str = 'kmeans') -> Dict[str, Any]:
        """
        Perform clustering analysis.
        
        Args:
            k: Number of clusters (if None, uses optimal_k)
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
            
        Returns:
            Dictionary with clustering results
        """
        if self.scaled_data is None:
            self.prepare_data()
        
        if k is None:
            if self.optimal_k is None:
                self.find_optimal_k()
            k = self.optimal_k
        
        results = {'algorithm': algorithm, 'n_clusters': k}
        
        if algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = clusterer.fit_predict(self.scaled_data)
            results['cluster_centers'] = clusterer.cluster_centers_
            results['inertia'] = clusterer.inertia_
            
        elif algorithm == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            labels = clusterer.fit_predict(self.scaled_data)
            k = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise points
            results['n_clusters'] = k
            results['n_noise_points'] = list(labels).count(-1)
            
        elif algorithm == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=k)
            labels = clusterer.fit_predict(self.scaled_data)
            
        else:
            raise ValueError("Algorithm must be 'kmeans', 'dbscan', or 'hierarchical'")
        
        self.cluster_labels = labels
        
        # Calculate clustering metrics
        if k > 1:
            if len(set(labels)) > 1:  # Check if we have more than one cluster
                results['silhouette_score'] = silhouette_score(self.scaled_data, labels)
                results['calinski_harabasz_score'] = calinski_harabasz_score(self.scaled_data, labels)
                results['davies_bouldin_score'] = davies_bouldin_score(self.scaled_data, labels)
        
        results['labels'] = labels
        self.clustering_results['final_clustering'] = results
        
        # Add cluster labels to original data
        self.data['cluster'] = labels
        
        return results
    
    def analyze_clusters(self) -> Dict[str, Any]:
        """
        Analyze cluster characteristics and profiles.
        
        Returns:
            Dictionary with cluster analysis
        """
        if self.cluster_labels is None:
            raise ValueError("Must perform clustering first")
        
        analysis = {}
        
        # Basic cluster information
        unique_clusters = np.unique(self.cluster_labels)
        n_clusters = len(unique_clusters)
        
        analysis['n_clusters'] = n_clusters
        analysis['cluster_sizes'] = {
            int(cluster): int(np.sum(self.cluster_labels == cluster))
            for cluster in unique_clusters
        }
        
        # Cluster profiles
        cluster_profiles = {}
        
        for cluster in unique_clusters:
            cluster_data = self.data[self.cluster_labels == cluster]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.data) * 100,
                'feature_means': {},
                'feature_stds': {},
                'categorical_distributions': {}
            }
            
            # Analyze numeric features
            for feature in self.features:
                profile['feature_means'][feature] = cluster_data[feature].mean()
                profile['feature_stds'][feature] = cluster_data[feature].std()
            
            # Analyze categorical variables
            categorical_cols = ['airline', 'departure_time', 'stops', 'class']
            for col in categorical_cols:
                if col in self.data.columns:
                    profile['categorical_distributions'][col] = cluster_data[col].value_counts().to_dict()
            
            # Cluster characterization
            profile['characterization'] = self._characterize_cluster(cluster_data)
            
            cluster_profiles[int(cluster)] = profile
        
        analysis['cluster_profiles'] = cluster_profiles
        
        # Overall comparison
        analysis['cluster_comparison'] = self._compare_clusters()
        
        self.clustering_results['analysis'] = analysis
        return analysis
    
    def get_cluster_recommendations(self) -> Dict[str, List[str]]:
        """
        Generate business recommendations for each cluster.
        
        Returns:
            Dictionary with recommendations for each cluster
        """
        if 'analysis' not in self.clustering_results:
            self.analyze_clusters()
        
        recommendations = {}
        cluster_profiles = self.clustering_results['analysis']['cluster_profiles']
        
        for cluster_id, profile in cluster_profiles.items():
            cluster_recs = []
            
            # Price-based recommendations
            avg_price = profile['feature_means']['price']
            overall_avg = self.data['price'].mean()
            
            if avg_price < overall_avg * 0.8:
                cluster_recs.append("Budget segment - Focus on cost efficiency and value propositions")
                cluster_recs.append("Market to price-sensitive customers")
            elif avg_price > overall_avg * 1.2:
                cluster_recs.append("Premium segment - Emphasize quality and premium services")
                cluster_recs.append("Target high-value customers with premium offerings")
            else:
                cluster_recs.append("Mainstream segment - Balance price and service quality")
            
            # Duration-based recommendations
            avg_duration = profile['feature_means']['duration']
            if avg_duration > self.data['duration'].mean():
                cluster_recs.append("Longer flights - Ensure comfort amenities and entertainment")
            
            # Booking pattern recommendations
            avg_days_left = profile['feature_means']['days_left']
            if avg_days_left < 7:
                cluster_recs.append("Last-minute bookers - Implement dynamic pricing strategies")
            elif avg_days_left > 30:
                cluster_recs.append("Early planners - Offer early bird discounts")
            
            # Airline-specific recommendations
            if 'airline' in profile['categorical_distributions']:
                top_airline = max(profile['categorical_distributions']['airline'].items(), key=lambda x: x[1])
                cluster_recs.append(f"Dominated by {top_airline[0]} - Consider partnership opportunities")
            
            recommendations[f"Cluster_{cluster_id}"] = cluster_recs
        
        return recommendations
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """
        Find elbow point in inertia curve.
        
        Args:
            k_values: List of k values
            inertias: List of corresponding inertias
            
        Returns:
            Optimal k value
        """
        # Calculate rate of change
        deltas = np.diff(inertias)
        delta_deltas = np.diff(deltas)
        
        # Find point with maximum change in rate of change
        if len(delta_deltas) > 0:
            elbow_idx = np.argmax(delta_deltas) + 2  # +2 because of double diff
            return k_values[min(elbow_idx, len(k_values) - 1)]
        else:
            return k_values[0]
    
    def _characterize_cluster(self, cluster_data: pd.DataFrame) -> str:
        """
        Characterize a cluster based on its properties.
        
        Args:
            cluster_data: DataFrame with cluster data
            
        Returns:
            Cluster characterization string
        """
        avg_price = cluster_data['price'].mean()
        overall_avg = self.data['price'].mean()
        
        if avg_price < overall_avg * 0.7:
            return "Budget Segment"
        elif avg_price > overall_avg * 1.3:
            return "Premium Segment"
        elif cluster_data['duration'].mean() > self.data['duration'].mean() * 1.2:
            return "Long-haul Segment"
        elif cluster_data['days_left'].mean() < 7:
            return "Last-minute Bookers"
        elif cluster_data['days_left'].mean() > 30:
            return "Early Planners"
        else:
            return "Mainstream Segment"
    
    def _compare_clusters(self) -> Dict[str, Any]:
        """
        Compare clusters across different dimensions.
        
        Returns:
            Dictionary with cluster comparisons
        """
        if self.cluster_labels is None:
            return {}
        
        comparison = {}
        
        # Price comparison
        price_by_cluster = self.data.groupby('cluster')['price'].agg(['mean', 'std', 'median'])
        comparison['price_stats'] = price_by_cluster.to_dict()
        
        # Duration comparison
        duration_by_cluster = self.data.groupby('cluster')['duration'].agg(['mean', 'std'])
        comparison['duration_stats'] = duration_by_cluster.to_dict()
        
        # Booking advance comparison
        days_by_cluster = self.data.groupby('cluster')['days_left'].agg(['mean', 'std'])
        comparison['booking_advance_stats'] = days_by_cluster.to_dict()
        
        # Airline distribution by cluster
        airline_dist = pd.crosstab(self.data['cluster'], self.data['airline'], normalize='index')
        comparison['airline_distribution'] = airline_dist.to_dict()
        
        return comparison
    
    def reduce_dimensions_for_visualization(self, n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensions using PCA for visualization.
        
        Args:
            n_components: Number of components to keep
            
        Returns:
            Reduced dimensional data
        """
        if self.scaled_data is None:
            self.prepare_data()
        
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(self.scaled_data)
        
        # Store PCA info
        self.clustering_results['pca'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_
        }
        
        return reduced_data
    
    def export_results(self) -> Dict[str, Any]:
        """
        Export all clustering results.
        
        Returns:
            Dictionary with all results
        """
        # Ensure we have all results
        if 'optimization' not in self.clustering_results:
            self.find_optimal_k()
        
        if 'final_clustering' not in self.clustering_results:
            self.perform_clustering()
        
        if 'analysis' not in self.clustering_results:
            self.analyze_clusters()
        
        results = {
            'clustering_results': self.clustering_results,
            'recommendations': self.get_cluster_recommendations(),
            'clustered_data': self.data.to_dict('records'),
            'feature_importance': self._calculate_feature_importance()
        }
        
        return results
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance for clustering.
        
        Returns:
            Dictionary with feature importance scores
        """
        if self.cluster_labels is None:
            return {}
        
        importance = {}
        
        for feature in self.features:
            # Calculate variance between clusters vs within clusters
            between_var = 0
            within_var = 0
            overall_mean = self.data[feature].mean()
            
            for cluster in np.unique(self.cluster_labels):
                cluster_data = self.data[self.cluster_labels == cluster][feature]
                cluster_mean = cluster_data.mean()
                cluster_size = len(cluster_data)
                
                # Between-cluster variance
                between_var += cluster_size * (cluster_mean - overall_mean) ** 2
                
                # Within-cluster variance
                within_var += ((cluster_data - cluster_mean) ** 2).sum()
            
            # F-ratio as importance measure
            if within_var > 0:
                f_ratio = (between_var / (len(np.unique(self.cluster_labels)) - 1)) / (within_var / (len(self.data) - len(np.unique(self.cluster_labels))))
                importance[feature] = f_ratio
            else:
                importance[feature] = 0
        
        return importance


def find_optimal_clusters(data: pd.DataFrame, features: List[str] = None, k_range: range = None) -> int:
    """
    Find optimal number of clusters using silhouette analysis.
    
    Args:
        data: DataFrame containing flight data
        features: Features to use for clustering
        k_range: Range of k values to test
        
    Returns:
        Optimal number of clusters
    """
    segmentation = MarketSegmentation(data, features)
    results = segmentation.find_optimal_k(k_range)
    return results['recommended_k']


def analyze_clusters(data: pd.DataFrame, features: List[str] = None, k: int = None) -> Dict[str, Any]:
    """
    Perform complete cluster analysis.
    
    Args:
        data: DataFrame containing flight data
        features: Features to use for clustering
        k: Number of clusters (if None, finds optimal)
        
    Returns:
        Dictionary with complete cluster analysis
    """
    segmentation = MarketSegmentation(data, features)
    
    if k is None:
        segmentation.find_optimal_k()
    
    segmentation.perform_clustering(k)
    analysis = segmentation.analyze_clusters()
    
    return {
        'analysis': analysis,
        'recommendations': segmentation.get_cluster_recommendations(),
        'cluster_data': segmentation.data
    }


def create_cluster_profiles(data: pd.DataFrame, cluster_column: str = 'cluster') -> pd.DataFrame:
    """
    Create detailed cluster profiles.
    
    Args:
        data: DataFrame with cluster assignments
        cluster_column: Name of cluster column
        
    Returns:
        DataFrame with cluster profiles
    """
    numeric_cols = ['price', 'duration', 'days_left']
    categorical_cols = ['airline', 'departure_time', 'stops']
    
    profiles = []
    
    for cluster in data[cluster_column].unique():
        cluster_data = data[data[cluster_column] == cluster]
        
        profile = {'cluster': cluster, 'size': len(cluster_data)}
        
        # Numeric features
        for col in numeric_cols:
            if col in data.columns:
                profile[f'{col}_mean'] = cluster_data[col].mean()
                profile[f'{col}_std'] = cluster_data[col].std()
        
        # Most common categorical values
        for col in categorical_cols:
            if col in data.columns:
                most_common = cluster_data[col].value_counts().index[0]
                profile[f'{col}_most_common'] = most_common
                profile[f'{col}_diversity'] = cluster_data[col].nunique()
        
        profiles.append(profile)
    
    return pd.DataFrame(profiles)


def perform_market_segmentation(data: pd.DataFrame, 
                              features: List[str] = None,
                              scaling_method: str = 'standard',
                              algorithm: str = 'kmeans',
                              k: int = None) -> Dict[str, Any]:
    """
    Perform complete market segmentation analysis.
    
    Args:
        data: DataFrame containing flight data
        features: Features to use for clustering
        scaling_method: Data scaling method
        algorithm: Clustering algorithm to use
        k: Number of clusters (if None, finds optimal)
        
    Returns:
        Complete segmentation analysis results
    """
    segmentation = MarketSegmentation(data, features)
    segmentation.prepare_data(scaling_method)
    
    if k is None:
        segmentation.find_optimal_k()
    
    segmentation.perform_clustering(k, algorithm)
    analysis = segmentation.analyze_clusters()
    
    return segmentation.export_results()