"""
Analysis module for Airlines Dataset Statistical Analysis.

This module provides statistical analysis tools including:
- Descriptive statistics
- Statistical hypothesis tests
- Clustering analysis
- Correlation analysis
"""

from .descriptive_stats import (
    DescriptiveAnalyzer,
    calculate_price_statistics,
    analyze_airline_efficiency,
    generate_summary_report
)

from .statistical_tests import (
    StatisticalTester,
    test_normality,
    correlation_significance,
    compare_groups,
    anova_analysis
)

from .clustering import (
    MarketSegmentation,
    find_optimal_clusters,
    analyze_clusters,
    create_cluster_profiles
)

__all__ = [
    # Descriptive Statistics
    'DescriptiveAnalyzer',
    'calculate_price_statistics',
    'analyze_airline_efficiency',
    'generate_summary_report',
    
    # Statistical Tests
    'StatisticalTester',
    'test_normality',
    'correlation_significance',
    'compare_groups',
    'anova_analysis',
    
    # Clustering
    'MarketSegmentation',
    'find_optimal_clusters',
    'analyze_clusters',
    'create_cluster_profiles'
]

__version__ = '1.0.0'
__author__ = 'Airlines Analysis Team'