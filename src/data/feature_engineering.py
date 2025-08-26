"""
Feature Engineering Module
==========================

Advanced feature creation and transformation for airline data analysis.
Creates predictive features, time-based features, and interaction variables.

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering class for airline data.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize FeatureEngineer.
        
        Args:
            verbose (bool): Whether to print detailed logging
        """
        self.verbose = verbose
        self.feature_log = []
        self.scalers = {}
        self.encoders = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Cleaned flights data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        self._log_step("Starting feature engineering", f"Input shape: {df.shape}")
        
        # Create a copy to avoid modifying original data
        df_features = df.copy()
        
        # Core feature groups
        df_features = self._create_price_features(df_features)
        df_features = self._create_time_features(df_features)
        df_features = self._create_airline_features(df_features)
        df_features = self._create_route_features(df_features)
        df_features = self._create_booking_features(df_features)
        df_features = self._create_interaction_features(df_features)
        df_features = self._create_derived_features(df_features)
        
        # Encoding categorical variables
        df_features = self._encode_categorical_features(df_features)
        
        self._log_step("Feature engineering completed", f"Final shape: {df_features.shape}")
        
        return df_features
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-related features.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with price features
        """
        if 'price' not in df.columns:
            return df
        
        features_created = []
        
        # Price statistics by airline
        airline_price_stats = df.groupby('airline')['price'].agg(['mean', 'median', 'std']).add_prefix('airline_price_')
        df = df.merge(airline_price_stats, left_on='airline', right_index=True, how='left')
        features_created.extend(airline_price_stats.columns)
        
        # Price relative to airline average
        df['price_vs_airline_avg'] = df['price'] / df['airline_price_mean']
        features_created.append('price_vs_airline_avg')
        
        # Price percentile within airline
        df['price_percentile_airline'] = df.groupby('airline')['price'].rank(pct=True)
        features_created.append('price_percentile_airline')
        
        # Overall price percentile
        df['price_percentile_overall'] = df['price'].rank(pct=True)
        features_created.append('price_percentile_overall')
        
        # Price categories
        df['price_category'] = pd.qcut(df['price'], q=3, labels=['Budget', 'Mid-Range', 'Premium'])
        features_created.append('price_category')
        
        # Price per hour
        if 'duration' in df.columns:
            df['price_per_hour'] = df['price'] / df['duration']
            features_created.append('price_per_hour')
        
        # Deviation from median price
        overall_median_price = df['price'].median()
        df['price_deviation_from_median'] = df['price'] - overall_median_price
        features_created.append('price_deviation_from_median')
        
        self._log_step("Price features created", f"Added {len(features_created)} features: {features_created}")
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-related features.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with time features
        """
        features_created = []
        
        # Departure time features
        if 'departure_time' in df.columns:
            # Create binary features for each time period
            time_periods = df['departure_time'].unique()
            for period in time_periods:
                feature_name = f'is_{period.lower()}'
                df[feature_name] = (df['departure_time'] == period).astype(int)
                features_created.append(feature_name)
            
            # Time period priority (business travelers preference)
            time_priority_map = {
                'Morning': 4,    # High priority for business
                'Afternoon': 3,  # Moderate priority
                'Evening': 2,    # Lower priority
                'Night': 1       # Lowest priority
            }
            df['time_priority'] = df['departure_time'].map(time_priority_map)
            features_created.append('time_priority')
        
        # Days left features
        if 'days_left' in df.columns:
            # Booking urgency categories
            df['booking_urgency'] = pd.cut(
                df['days_left'], 
                bins=[0, 7, 21, 45, 365], 
                labels=['Last_Minute', 'Short_Term', 'Medium_Term', 'Long_Term'],
                include_lowest=True
            )
            features_created.append('booking_urgency')
            
            # Binary features for booking timing
            df['is_last_minute'] = (df['days_left'] <= 7).astype(int)
            df['is_advance_booking'] = (df['days_left'] > 21).astype(int)
            features_created.extend(['is_last_minute', 'is_advance_booking'])
            
            # Days left squared (non-linear relationship)
            df['days_left_squared'] = df['days_left'] ** 2
            features_created.append('days_left_squared')
            
            # Log transformation for days left
            df['days_left_log'] = np.log1p(df['days_left'])
            features_created.append('days_left_log')
        
        self._log_step("Time features created", f"Added {len(features_created)} features: {features_created}")
        
        return df
    
    def _create_airline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create airline-specific features.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with airline features
        """
        if 'airline' not in df.columns:
            return df
        
        features_created = []
        
        # Airline market share
        airline_counts = df['airline'].value_counts()
        total_flights = len(df)
        airline_market_share = (airline_counts / total_flights).to_dict()
        df['airline_market_share'] = df['airline'].map(airline_market_share)
        features_created.append('airline_market_share')
        
        # Airline category (based on typical positioning)
        airline_category_map = {
            'Vistara': 'Premium',
            'Air India': 'Premium',
            'IndiGo': 'Low-Cost',
            'SpiceJet': 'Low-Cost',
            'GoAir': 'Low-Cost',
            'AirAsia': 'Low-Cost'
        }
        df['airline_category'] = df['airline'].map(airline_category_map).fillna('Other')
        features_created.append('airline_category')
        
        # Binary features for major airlines
        major_airlines = df['airline'].value_counts().head(4).index
        for airline in major_airlines:
            feature_name = f'is_{airline.lower().replace(" ", "_")}'
            df[feature_name] = (df['airline'] == airline).astype(int)
            features_created.append(feature_name)
        
        # Airline performance metrics
        if 'duration' in df.columns:
            airline_duration_stats = df.groupby('airline')['duration'].mean().to_dict()
            df['airline_avg_duration'] = df['airline'].map(airline_duration_stats)
            features_created.append('airline_avg_duration')
        
        self._log_step("Airline features created", f"Added {len(features_created)} features: {features_created}")
        
        return df
    
    def _create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create route and stops-related features.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with route features
        """
        features_created = []
        
        # Stops features
        if 'stops' in df.columns:
            # Binary features for stop types
            stop_types = df['stops'].unique()
            for stop_type in stop_types:
                feature_name = f'has_{stop_type}_stops'
                df[feature_name] = (df['stops'] == stop_type).astype(int)
                features_created.append(feature_name)
            
            # Numeric stops count
            stops_map = {'zero': 0, 'one': 1, 'two_or_more': 2}
            df['stops_count'] = df['stops'].map(stops_map)
            features_created.append('stops_count')
            
            # Direct flight premium (additional cost for non-stop)
            if 'price' in df.columns:
                direct_avg_price = df[df['stops'] == 'zero']['price'].mean()
                connecting_avg_price = df[df['stops'] != 'zero']['price'].mean()
                df['direct_flight_premium'] = direct_avg_price - connecting_avg_price
                features_created.append('direct_flight_premium')
        
        # Duration-related route features
        if 'duration' in df.columns:
            # Duration categories
            df['duration_category'] = pd.cut(
                df['duration'], 
                bins=[0, 2, 3, 4, 24], 
                labels=['Short', 'Medium', 'Long', 'Very_Long'],
                include_lowest=True
            )
            features_created.append('duration_category')
            
            # Speed estimate (assuming ~1000km distance for Delhi-Mumbai)
            estimated_distance = 1000  # km
            df['estimated_speed'] = estimated_distance / df['duration']
            features_created.append('estimated_speed')
            
            # Duration efficiency (compared to minimum possible)
            min_duration = df['duration'].min()
            df['duration_efficiency'] = min_duration / df['duration']
            features_created.append('duration_efficiency')
        
        self._log_step("Route features created", f"Added {len(features_created)} features: {features_created}")
        
        return df
    
    def _create_booking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create booking behavior features.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with booking features
        """
        features_created = []
        
        if 'days_left' in df.columns and 'price' in df.columns:
            # Price trend based on booking time
            price_by_days_left = df.groupby('days_left')['price'].mean().sort_index()
            
            # Create booking time vs price relationship
            df['booking_time_price_trend'] = df['days_left'].map(price_by_days_left)
            features_created.append('booking_time_price_trend')
            
            # Optimal booking window (typically 3-8 weeks before)
            df['is_optimal_booking_window'] = ((df['days_left'] >= 21) & (df['days_left'] <= 56)).astype(int)
            features_created.append('is_optimal_booking_window')
            
            # Early bird vs last minute
            df['booking_strategy'] = np.where(
                df['days_left'] <= 7, 'Last_Minute',
                np.where(df['days_left'] >= 30, 'Early_Bird', 'Normal')
            )
            features_created.append('booking_strategy')
        
        # Weekend vs weekday patterns (assuming some flights are weekend-heavy)
        if 'departure_time' in df.columns:
            # Evening flights might be preferred for weekend travel
            df['likely_weekend_flight'] = ((df['departure_time'] == 'Evening') | 
                                         (df['departure_time'] == 'Night')).astype(int)
            features_created.append('likely_weekend_flight')
        
        self._log_step("Booking features created", f"Added {len(features_created)} features: {features_created}")
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with interaction features
        """
        features_created = []
        
        # Airline × Departure Time interactions
        if 'airline' in df.columns and 'departure_time' in df.columns:
            df['airline_time_combo'] = df['airline'] + '_' + df['departure_time']
            features_created.append('airline_time_combo')
        
        # Airline × Stops interactions
        if 'airline' in df.columns and 'stops' in df.columns:
            df['airline_stops_combo'] = df['airline'] + '_' + df['stops']
            features_created.append('airline_stops_combo')
        
        # Time × Booking urgency
        if 'departure_time' in df.columns and 'booking_urgency' in df.columns:
            df['time_urgency_combo'] = df['departure_time'] + '_' + df['booking_urgency'].astype(str)
            features_created.append('time_urgency_combo')
        
        # Price × Duration efficiency
        if 'price' in df.columns and 'duration' in df.columns:
            df['price_duration_ratio'] = df['price'] / df['duration']
            features_created.append('price_duration_ratio')
        
        # Airline category × Booking strategy
        if 'airline_category' in df.columns and 'booking_strategy' in df.columns:
            df['category_booking_combo'] = df['airline_category'] + '_' + df['booking_strategy']
            features_created.append('category_booking_combo')
        
        self._log_step("Interaction features created", f"Added {len(features_created)} features: {features_created}")
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced derived features using statistical techniques.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with derived features
        """
        features_created = []
        
        # Statistical features
        if 'price' in df.columns:
            # Z-score for price (how many standard deviations from mean)
            price_mean = df['price'].mean()
            price_std = df['price'].std()
            df['price_zscore'] = (df['price'] - price_mean) / price_std
            features_created.append('price_zscore')
            
            # Price rank within similar flights (same airline and stops)
            if 'airline' in df.columns and 'stops' in df.columns:
                df['price_rank_similar_flights'] = df.groupby(['airline', 'stops'])['price'].rank(pct=True)
                features_created.append('price_rank_similar_flights')
        
        # Competitive features
        if all(col in df.columns for col in ['airline', 'departure_time', 'price']):
            # Price competitiveness in same time slot
            avg_price_by_time = df.groupby('departure_time')['price'].mean().to_dict()
            df['price_vs_time_avg'] = df['price'] / df['departure_time'].map(avg_price_by_time)
            features_created.append('price_vs_time_avg')
        
        # Complexity score (combination of multiple factors)
        complexity_factors = []
        if 'stops_count' in df.columns:
            complexity_factors.append('stops_count')
        if 'duration' in df.columns:
            complexity_factors.append('duration')
        
        if complexity_factors:
            # Normalize factors and create complexity score
            for factor in complexity_factors:
                factor_normalized = f'{factor}_normalized'
                df[factor_normalized] = (df[factor] - df[factor].min()) / (df[factor].max() - df[factor].min())
            
            normalized_cols = [f'{factor}_normalized' for factor in complexity_factors]
            df['route_complexity_score'] = df[normalized_cols].mean(axis=1)
            features_created.append('route_complexity_score')
            
            # Clean up temporary normalized columns
            df.drop(columns=normalized_cols, inplace=True)
        
        # Value score (price vs features ratio)
        if all(col in df.columns for col in ['price', 'duration']):
            # Simple value score: lower price and shorter duration = better value
            duration_normalized = 1 - (df['duration'] - df['duration'].min()) / (df['duration'].max() - df['duration'].min())
            price_normalized = 1 - (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())
            df['value_score'] = (duration_normalized + price_normalized) / 2
            features_created.append('value_score')
        
        self._log_step("Derived features created", f"Added {len(features_created)} features: {features_created}")
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables for machine learning.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features
        """
        features_created = []
        
        # Get categorical columns (excluding target if exists)
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove price from categorical if it somehow got there
        if 'price' in categorical_columns:
            categorical_columns.remove('price')
        
        for col in categorical_columns:
            unique_values = df[col].nunique()