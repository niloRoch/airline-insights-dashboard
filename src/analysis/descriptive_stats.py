"""
Descriptive Statistics Analysis Module

This module provides comprehensive descriptive statistical analysis
for the Airlines Dataset, including summary statistics, distributions,
and comparative analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class DescriptiveAnalyzer:
    """
    Main class for descriptive statistical analysis of flight data.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with flight data.
        
        Args:
            data: DataFrame containing flight information
        """
        self.data = data.copy()
        self.numeric_columns = ['price', 'duration', 'days_left']
        self.categorical_columns = ['airline', 'departure_time', 'stops', 'class']
        
    def calculate_basic_stats(self, column: str) -> Dict[str, float]:
        """
        Calculate basic descriptive statistics for a numeric column.
        
        Args:
            column: Name of the column to analyze
            
        Returns:
            Dictionary with statistical measures
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
            
        data_col = self.data[column].dropna()
        
        return {
            'count': len(data_col),
            'mean': data_col.mean(),
            'median': data_col.median(),
            'std': data_col.std(),
            'var': data_col.var(),
            'min': data_col.min(),
            'max': data_col.max(),
            'q1': data_col.quantile(0.25),
            'q3': data_col.quantile(0.75),
            'iqr': data_col.quantile(0.75) - data_col.quantile(0.25),
            'skewness': data_col.skew(),
            'kurtosis': data_col.kurtosis(),
            'cv': (data_col.std() / data_col.mean()) * 100 if data_col.mean() != 0 else 0
        }
    
    def analyze_price_distribution(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of price distribution.
        
        Returns:
            Dictionary with price analysis results
        """
        price_stats = self.calculate_basic_stats('price')
        
        # Identify outliers using IQR method
        Q1 = price_stats['q1']
        Q3 = price_stats['q3']
        IQR = price_stats['iqr']
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.data[
            (self.data['price'] < lower_bound) | 
            (self.data['price'] > upper_bound)
        ]
        
        # Price categories
        price_categories = pd.qcut(
            self.data['price'], 
            q=3, 
            labels=['Budget', 'Mid-Range', 'Premium']
        )
        
        return {
            'basic_stats': price_stats,
            'outliers': {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            },
            'categories': {
                'Budget': (self.data['price'] <= self.data['price'].quantile(0.33)).sum(),
                'Mid-Range': ((self.data['price'] > self.data['price'].quantile(0.33)) & 
                             (self.data['price'] <= self.data['price'].quantile(0.67))).sum(),
                'Premium': (self.data['price'] > self.data['price'].quantile(0.67)).sum()
            },
            'distribution_shape': {
                'is_right_skewed': price_stats['skewness'] > 0,
                'is_leptokurtic': price_stats['kurtosis'] > 0,
                'skewness_interpretation': self._interpret_skewness(price_stats['skewness']),
                'kurtosis_interpretation': self._interpret_kurtosis(price_stats['kurtosis'])
            }
        }
    
    def analyze_by_group(self, group_column: str, target_column: str = 'price') -> pd.DataFrame:
        """
        Analyze target column statistics by groups.
        
        Args:
            group_column: Column to group by
            target_column: Column to analyze (default: price)
            
        Returns:
            DataFrame with group statistics
        """
        if group_column not in self.data.columns:
            raise ValueError(f"Group column '{group_column}' not found")
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found")
            
        group_stats = self.data.groupby(group_column)[target_column].agg([
            'count', 'mean', 'median', 'std', 'min', 'max',
            lambda x: x.quantile(0.25),  # Q1
            lambda x: x.quantile(0.75)   # Q3
        ]).round(2)
        
        group_stats.columns = ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3']
        
        # Add coefficient of variation
        group_stats['CV'] = (group_stats['Std'] / group_stats['Mean'] * 100).round(1)
        
        # Add IQR
        group_stats['IQR'] = group_stats['Q3'] - group_stats['Q1']
        
        # Sort by mean descending
        group_stats = group_stats.sort_values('Mean', ascending=False)
        
        return group_stats
    
    def calculate_efficiency_metrics(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate efficiency metrics for airlines.
        
        Returns:
            Dictionary with efficiency DataFrames
        """
        # Price per hour efficiency
        price_per_hour = self.data.groupby('airline').apply(
            lambda x: (x['price'] / x['duration']).mean()
        ).sort_values().round(0)
        
        # Duration efficiency (average flight time)
        duration_stats = self.analyze_by_group('airline', 'duration')
        
        # Days left efficiency (how far in advance bookings are made)
        advance_booking = self.analyze_by_group('airline', 'days_left')
        
        return {
            'price_per_hour': price_per_hour,
            'duration_efficiency': duration_stats,
            'advance_booking_patterns': advance_booking
        }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.
        
        Returns:
            Dictionary with complete analysis summary
        """
        # Basic dataset info
        dataset_info = {
            'total_flights': len(self.data),
            'airlines_count': self.data['airline'].nunique(),
            'date_range': {
                'min_days_left': self.data['days_left'].min(),
                'max_days_left': self.data['days_left'].max()
            },
            'flight_types': {
                'direct': (self.data['stops'] == 'zero').sum(),
                'one_stop': (self.data['stops'] == 'one').sum(),
                'multi_stop': (self.data['stops'] == 'two_or_more').sum() if 'two_or_more' in self.data['stops'].values else 0
            }
        }
        
        # Key insights
        insights = self._generate_key_insights()
        
        return {
            'dataset_overview': dataset_info,
            'price_analysis': self.analyze_price_distribution(),
            'airline_comparison': self.analyze_by_group('airline', 'price'),
            'time_analysis': self.analyze_by_group('departure_time', 'price'),
            'efficiency_metrics': self.calculate_efficiency_metrics(),
            'key_insights': insights,
            'data_quality': self._assess_data_quality()
        }
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value."""
        if abs(skewness) < 0.5:
            return "Approximately symmetric"
        elif skewness > 0.5:
            return "Right-skewed (positive skew)"
        else:
            return "Left-skewed (negative skew)"
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value."""
        if abs(kurtosis) < 0.5:
            return "Mesokurtic (normal-like tails)"
        elif kurtosis > 0.5:
            return "Leptokurtic (heavy tails)"
        else:
            return "Platykurtic (light tails)"
    
    def _generate_key_insights(self) -> List[str]:
        """Generate key insights from the analysis."""
        insights = []
        
        # Price insights
        price_stats = self.calculate_basic_stats('price')
        insights.append(f"Average flight price: ₹{price_stats['mean']:.0f}")
        insights.append(f"Price variability: {price_stats['cv']:.1f}% coefficient of variation")
        
        # Airline insights
        airline_stats = self.analyze_by_group('airline', 'price')
        most_expensive = airline_stats.index[0]
        cheapest = airline_stats.index[-1]
        insights.append(f"Most expensive airline: {most_expensive} (₹{airline_stats.loc[most_expensive, 'Mean']:.0f})")
        insights.append(f"Most budget-friendly: {cheapest} (₹{airline_stats.loc[cheapest, 'Mean']:.0f})")
        
        # Flight type insights
        direct_mean = self.data[self.data['stops'] == 'zero']['price'].mean()
        stop_mean = self.data[self.data['stops'] == 'one']['price'].mean()
        if not pd.isna(direct_mean) and not pd.isna(stop_mean):
            price_diff = abs(direct_mean - stop_mean)
            insights.append(f"Price difference between direct and one-stop flights: ₹{price_diff:.0f}")
        
        return insights
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess data quality metrics."""
        return {
            'missing_values': {
                col: self.data[col].isnull().sum() 
                for col in self.data.columns
            },
            'duplicate_rows': self.data.duplicated().sum(),
            'data_types': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum()
        }


def calculate_price_statistics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive price statistics.
    
    Args:
        data: Flight data DataFrame
        
    Returns:
        Dictionary with price statistics
    """
    analyzer = DescriptiveAnalyzer(data)
    return analyzer.calculate_basic_stats('price')


def analyze_airline_efficiency(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Analyze airline efficiency metrics.
    
    Args:
        data: Flight data DataFrame
        
    Returns:
        Dictionary with efficiency metrics
    """
    analyzer = DescriptiveAnalyzer(data)
    efficiency = analyzer.calculate_efficiency_metrics()
    return {
        'price_per_hour': efficiency['price_per_hour'],
        'avg_duration': efficiency['duration_efficiency']['Mean'],
        'advance_booking': efficiency['advance_booking_patterns']['Mean']
    }


def generate_summary_report(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive summary report.
    
    Args:
        data: Flight data DataFrame
        
    Returns:
        Complete analysis summary
    """
    analyzer = DescriptiveAnalyzer(data)
    return analyzer.generate_summary_report()


def compare_groups(data: pd.DataFrame, group_col: str, target_col: str = 'price') -> pd.DataFrame:
    """
    Compare groups using descriptive statistics.
    
    Args:
        data: Flight data DataFrame
        group_col: Column to group by
        target_col: Column to analyze
        
    Returns:
        DataFrame with group comparison
    """
    analyzer = DescriptiveAnalyzer(data)
    return analyzer.analyze_by_group(group_col, target_col)