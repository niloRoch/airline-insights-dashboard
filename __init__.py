"""
Airlines Data Analysis Project
============================

A comprehensive analysis project for airline flights data between Delhi and Mumbai,
featuring statistical analysis, machine learning, and interactive visualizations.

Author: [Your Name]
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Nilo Rocha"
__description__ = "Airlines Data Analysis - Delhi to Mumbai Flight Patterns"

# Import main modules for easy access
from .data import data_loader, data_cleaner, feature_engineering
from .analysis import descriptive_stats, statistical_tests, clustering
from .visualization import charts, interactive_plots, dashboard_components
from .utils import helpers, config

__all__ = [
    "data_loader",
    "data_cleaner", 
    "feature_engineering",
    "descriptive_stats",
    "statistical_tests",
    "clustering",
    "charts",
    "interactive_plots",
    "dashboard_components",
    "helpers",
    "config"
]