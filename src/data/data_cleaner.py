"""
Data Cleaning Module
===================

Comprehensive data cleaning and preprocessing for airline datasets.
Handles missing values, outliers, data type conversions, and standardization.

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing class for airline datasets.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize DataCleaner.
        
        Args:
            verbose (bool): Whether to print detailed logging information
        """
        self.verbose = verbose
        self.cleaning_log = []
        self.original_shape = None
        self.cleaned_shape = None
        
    def clean_flights_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning pipeline for flights data.
        
        Args:
            df (pd.DataFrame): Raw flights data
            
        Returns:
            pd.DataFrame: Cleaned flights data
        """
        self.original_shape = df.shape
        self._log_step("Starting data cleaning pipeline", f"Original data: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Create a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Step 1: Clean column names
        df_clean = self._clean_column_names(df_clean)
        
        # Step 2: Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Step 3: Clean and standardize categorical variables
        df_clean = self._clean_categorical_variables(df_clean)
        
        # Step 4: Clean numerical variables
        df_clean = self._clean_numerical_variables(df_clean)
        
        # Step 5: Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Step 6: Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # Step 7: Final validation and type conversion
        df_clean = self._finalize_data_types(df_clean)
        
        self.cleaned_shape = df_clean.shape
        self._log_step("Data cleaning completed", f"Cleaned data: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
        
        return df_clean
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cleaned column names
        """
        original_columns = df.columns.tolist()
        
        # Clean column names: lowercase, replace spaces with underscores, remove special characters
        new_columns = []
        for col in df.columns:
            new_col = col.lower().strip()
            new_col = re.sub(r'[^\w\s]', '', new_col)  # Remove special characters
            new_col = re.sub(r'\s+', '_', new_col)     # Replace spaces with underscores
            new_col = re.sub(r'_+', '_', new_col)      # Replace multiple underscores with single
            new_col = new_col.strip('_')               # Remove leading/trailing underscores
            new_columns.append(new_col)
        
        df.columns = new_columns
        
        if original_columns != new_columns:
            self._log_step("Column names cleaned", f"Renamed columns: {dict(zip(original_columns, new_columns))}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies for each column.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        missing_before = df.isnull().sum().sum()
        
        if missing_before == 0:
            self._log_step("Missing values check", "No missing values found")
            return df
        
        # Strategy for each column type
        strategies = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            missing_pct = missing_count / len(df) * 100
            
            if col in ['airline', 'departure_time', 'stops']:
                # Categorical variables - use mode or specific logic
                if col == 'airline':
                    mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col].fillna(mode_value, inplace=True)
                    strategies[col] = f"Filled with mode: {mode_value}"
                    
                elif col == 'departure_time':
                    # Fill missing departure times with 'Morning' as most common
                    df[col].fillna('Morning', inplace=True)
                    strategies[col] = "Filled with 'Morning'"
                    
                elif col == 'stops':
                    # Fill missing stops with 'zero' (most common for domestic flights)
                    df[col].fillna('zero', inplace=True)
                    strategies[col] = "Filled with 'zero'"
                    
            elif col in ['price', 'duration', 'days_left']:
                # Numerical variables
                if missing_pct < 5:
                    # Small percentage - use median
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
                    strategies[col] = f"Filled with median: {median_value}"
                    
                elif missing_pct < 20:
                    # Medium percentage - use forward/backward fill or interpolation
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(method='bfill', inplace=True)
                    strategies[col] = "Forward/backward fill"
                    
                else:
                    # High percentage - consider dropping column or more sophisticated imputation
                    if col == 'price':
                        # Price is crucial - use airline-specific median
                        df[col] = df.groupby('airline')[col].transform(
                            lambda x: x.fillna(x.median())
                        )
                        strategies[col] = "Filled with airline-specific median"
                    else:
                        median_value = df[col].median()
                        df[col].fillna(median_value, inplace=True)
                        strategies[col] = f"Filled with overall median: {median_value}"
        
        missing_after = df.isnull().sum().sum()
        
        if strategies:
            self._log_step("Missing values handled", 
                          f"Reduced from {missing_before} to {missing_after} missing values. "
                          f"Strategies: {strategies}")
        
        return df
    
    def _clean_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize categorical variables.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cleaned categorical variables
        """
        cleaning_changes = {}
        
        # Clean airline names
        if 'airline' in df.columns:
            original_airlines = df['airline'].unique()
            
            # Standardize airline names
            airline_mapping = {
                'spicejet': 'SpiceJet',
                'vistara': 'Vistara', 
                'indigo': 'IndiGo',
                'air india': 'Air India',
                'go air': 'GoAir',
                'goair': 'GoAir',
                'air asia': 'AirAsia',
                'airasia': 'AirAsia'
            }
            
            # Apply mapping
            df['airline'] = df['airline'].str.strip().str.lower()
            for old_name, new_name in airline_mapping.items():
                df['airline'] = df['airline'].str.replace(old_name, new_name, case=False)
            
            # Capitalize properly
            df['airline'] = df['airline'].str.title()
            
            new_airlines = df['airline'].unique()
            if set(original_airlines) != set(new_airlines):
                cleaning_changes['airline'] = f"Standardized from {original_airlines} to {new_airlines}"
        
        # Clean departure_time
        if 'departure_time' in df.columns:
            original_times = df['departure_time'].unique()
            
            # Standardize departure time categories
            time_mapping = {
                'early_morning': 'Morning',
                'morning': 'Morning',
                'afternoon': 'Afternoon',
                'evening': 'Evening',
                'night': 'Night',
                'late_night': 'Night'
            }
            
            df['departure_time'] = df['departure_time'].str.strip().str.lower()
            for old_time, new_time in time_mapping.items():
                df['departure_time'] = df['departure_time'].str.replace(old_time, new_time, case=False)
            
            df['departure_time'] = df['departure_time'].str.title()
            
            new_times = df['departure_time'].unique()
            if set(original_times) != set(new_times):
                cleaning_changes['departure_time'] = f"Standardized from {original_times} to {new_times}"
        
        # Clean stops information
        if 'stops' in df.columns:
            original_stops = df['stops'].unique()
            
            # Standardize stops categories
            stops_mapping = {
                '0': 'zero',
                'non-stop': 'zero',
                'nonstop': 'zero',
                'direct': 'zero',
                '1': 'one',
                '1 stop': 'one',
                '2+': 'two_or_more',
                '2': 'two_or_more',
                '3': 'two_or_more',
                '2 or more': 'two_or_more'
            }
            
            df['stops'] = df['stops'].astype(str).str.strip().str.lower()
            for old_stop, new_stop in stops_mapping.items():
                df['stops'] = df['stops'].str.replace(old_stop, new_stop, case=False)
            
            new_stops = df['stops'].unique()
            if set(original_stops) != set(new_stops):
                cleaning_changes['stops'] = f"Standardized from {original_stops} to {new_stops}"
        
        if cleaning_changes:
            self._log_step("Categorical variables cleaned", str(cleaning_changes))
        
        return df
    
    def _clean_numerical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate numerical variables.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cleaned numerical variables
        """
        cleaning_changes = {}
        
        # Clean price column
        if 'price' in df.columns:
            original_count = len(df)
            
            # Remove currency symbols and convert to numeric
            if df['price'].dtype == 'object':
                df['price'] = df['price'].astype(str).str.replace(r'[₹,\$]', '', regex=True)
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Remove invalid prices (negative or zero)
            invalid_prices = (df['price'] <= 0) | (df['price'].isnull())
            df = df[~invalid_prices]
            
            removed_count = original_count - len(df)
            if removed_count > 0:
                cleaning_changes['price'] = f"Removed {removed_count} records with invalid prices"
        
        # Clean duration column
        if 'duration' in df.columns:
            original_count = len(df)
            
            # Convert duration to hours if it's in string format
            if df['duration'].dtype == 'object':
                df['duration'] = self._parse_duration_string(df['duration'])
            
            # Remove invalid durations
            invalid_duration = (df['duration'] <= 0) | (df['duration'] > 24) | (df['duration'].isnull())
            df = df[~invalid_duration]
            
            removed_count = original_count - len(df)
            if removed_count > 0:
                cleaning_changes['duration'] = f"Removed {removed_count} records with invalid durations"
        
        # Clean days_left column
        if 'days_left' in df.columns:
            original_count = len(df)
            
            # Ensure days_left is positive integer
            df['days_left'] = pd.to_numeric(df['days_left'], errors='coerce')
            invalid_days = (df['days_left'] < 0) | (df['days_left'] > 365) | (df['days_left'].isnull())
            df = df[~invalid_days]
            
            # Round to integers
            df['days_left'] = df['days_left'].round().astype(int)
            
            removed_count = original_count - len(df)
            if removed_count > 0:
                cleaning_changes['days_left'] = f"Removed {removed_count} records with invalid days_left"
        
        if cleaning_changes:
            self._log_step("Numerical variables cleaned", str(cleaning_changes))
        
        return df
    
    def _parse_duration_string(self, duration_series: pd.Series) -> pd.Series:
        """
        Parse duration strings and convert to hours.
        
        Args:
            duration_series (pd.Series): Series with duration strings
            
        Returns:
            pd.Series: Duration in hours as float
        """
        def parse_single_duration(duration_str):
            if pd.isnull(duration_str):
                return np.nan
            
            duration_str = str(duration_str).lower().strip()
            
            # Handle different duration formats
            hours = 0
            minutes = 0
            
            # Format: "2h 30m" or "2 hours 30 minutes"
            hour_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:h|hours?|hr)', duration_str)
            minute_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|minutes?|min)', duration_str)
            
            if hour_match:
                hours = float(hour_match.group(1))
            if minute_match:
                minutes = float(minute_match.group(1))
            
            # If only one number found, assume it's hours
            if hours == 0 and minutes == 0:
                number_match = re.search(r'(\d+(?:\.\d+)?)', duration_str)
                if number_match:
                    hours = float(number_match.group(1))
            
            return hours + (minutes / 60)
        
        return duration_series.apply(parse_single_duration)
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate records.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame without duplicates
        """
        original_count = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Remove duplicates based on key columns (more sophisticated)
        key_columns = ['airline', 'price', 'departure_time', 'duration', 'stops']
        available_key_columns = [col for col in key_columns if col in df.columns]
        
        if len(available_key_columns) > 2:
            df = df.drop_duplicates(subset=available_key_columns, keep='first')
        
        duplicates_removed = original_count - len(df)
        
        if duplicates_removed > 0:
            self._log_step("Duplicates removed", f"Removed {duplicates_removed} duplicate records")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in numerical variables using IQR method.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        outlier_changes = {}
        
        numerical_cols = ['price', 'duration', 'days_left']
        
        for col in numerical_cols:
            if col not in df.columns:
                continue
            
            original_count = len(df)
            
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # For price, be more conservative with lower bound (don't remove low prices)
            if col == 'price':
                lower_bound = max(lower_bound, 1000)  # Minimum reasonable price
                upper_bound = min(upper_bound, 50000)  # Maximum reasonable price
            
            # Remove outliers
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_count = outliers.sum()
            
            if outliers_count > 0:
                df = df[~outliers]
                outlier_changes[col] = f"Removed {outliers_count} outliers (bounds: {lower_bound:.1f} - {upper_bound:.1f})"
        
        if outlier_changes:
            self._log_step("Outliers handled", str(outlier_changes))
        
        return df
    
    def _finalize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finalize data types and perform final validation.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with finalized data types
        """
        type_changes = {}
        
        # Set proper data types
        if 'price' in df.columns:
            df['price'] = df['price'].astype(float).round(2)
            type_changes['price'] = 'float64'
        
        if 'duration' in df.columns:
            df['duration'] = df['duration'].astype(float).round(2)
            type_changes['duration'] = 'float64'
        
        if 'days_left' in df.columns:
            df['days_left'] = df['days_left'].astype(int)
            type_changes['days_left'] = 'int64'
        
        # Categorical variables as category type for memory efficiency
        categorical_cols = ['airline', 'departure_time', 'stops']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
                type_changes[col] = 'category'
        
        # Final validation
        self._validate_cleaned_data(df)
        
        if type_changes:
            self._log_step("Data types finalized", str(type_changes))
        
        return df
    
    def _validate_cleaned_data(self, df: pd.DataFrame) -> None:
        """
        Validate the cleaned dataset.
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        validation_issues = []
        
        # Check for remaining missing values
        if df.isnull().sum().sum() > 0:
            validation_issues.append("Missing values still present")
        
        # Check data ranges
        if 'price' in df.columns:
            if df['price'].min() <= 0:
                validation_issues.append("Invalid prices found")
            if df['price'].max() > 100000:
                validation_issues.append("Extremely high prices found")
        
        if 'duration' in df.columns:
            if df['duration'].min() <= 0:
                validation_issues.append("Invalid durations found")
            if df['duration'].max() > 24:
                validation_issues.append("Extremely long durations found")
        
        if 'days_left' in df.columns:
            if df['days_left'].min() < 0:
                validation_issues.append("Negative days_left found")
        
        # Check for empty DataFrame
        if len(df) == 0:
            validation_issues.append("Dataset is empty after cleaning")
        
        if validation_issues:
            warnings.warn(f"Validation issues: {'; '.join(validation_issues)}")
        else:
            self._log_step("Validation passed", "All data quality checks passed")
    
    def _log_step(self, step: str, details: str) -> None:
        """
        Log a cleaning step.
        
        Args:
            step (str): Step name
            details (str): Step details
        """
        log_entry = {
            'step': step,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.cleaning_log.append(log_entry)
        
        if self.verbose:
            logger.info(f"{step}: {details}")
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive cleaning report.
        
        Returns:
            Dict: Detailed cleaning report
        """
        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.cleaned_shape,
            'records_removed': self.original_shape[0] - self.cleaned_shape[0] if self.original_shape and self.cleaned_shape else 0,
            'removal_percentage': ((self.original_shape[0] - self.cleaned_shape[0]) / self.original_shape[0] * 100) if self.original_shape and self.cleaned_shape else 0,
            'cleaning_steps': len(self.cleaning_log),
            'cleaning_log': self.cleaning_log
        }
        
        return report

# Utility functions for specific cleaning tasks
def clean_airline_names(series: pd.Series) -> pd.Series:
    """
    Clean airline names to standard format.
    
    Args:
        series (pd.Series): Series with airline names
        
    Returns:
        pd.Series: Cleaned airline names
    """
    airline_mapping = {
        'spicejet': 'SpiceJet',
        'vistara': 'Vistara',
        'indigo': 'IndiGo', 
        'air india': 'Air India',
        'goair': 'GoAir',
        'airasia': 'AirAsia'
    }
    
    cleaned = series.str.strip().str.lower()
    for old_name, new_name in airline_mapping.items():
        cleaned = cleaned.str.replace(old_name, new_name, case=False)
    
    return cleaned.str.title()

def standardize_time_periods(series: pd.Series) -> pd.Series:
    """
    Standardize time period categories.
    
    Args:
        series (pd.Series): Series with time periods
        
    Returns:
        pd.Series: Standardized time periods
    """
    time_mapping = {
        'early_morning': 'Morning',
        'morning': 'Morning',
        'afternoon': 'Afternoon', 
        'evening': 'Evening',
        'night': 'Night',
        'late_night': 'Night'
    }
    
    cleaned = series.str.strip().str.lower()
    for old_time, new_time in time_mapping.items():
        cleaned = cleaned.str.replace(old_time, new_time, case=False)
    
    return cleaned.str.title()

def remove_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Remove outliers using IQR method.
    
    Args:
        series (pd.Series): Numerical series
        factor (float): IQR multiplication factor
        
    Returns:
        pd.Series: Series with outliers removed
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    return series[(series >= lower_bound) & (series <= upper_bound)]

# Main cleaning function for external use
def clean_flights_data(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean flights dataset using the DataCleaner class.
    
    Args:
        df (pd.DataFrame): Raw flights data
        verbose (bool): Whether to print detailed logs
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Cleaned data and cleaning report
    """
    cleaner = DataCleaner(verbose=verbose)
    cleaned_df = cleaner.clean_flights_data(df)
    report = cleaner.get_cleaning_report()
    
    return cleaned_df, report

if __name__ == "__main__":
    # Example usage and testing
    from data_loader import get_sample_data
    
    # Generate sample data for testing
    sample_df = get_sample_data(1000)
    
    # Add some data quality issues for testing
    sample_df.loc[10:15, 'price'] = np.nan
    sample_df.loc[20:22, 'airline'] = 'unknown airline'
    sample_df.loc[25, 'duration'] = -1
    sample_df.loc[30, 'price'] = 1000000  # Outlier
    
    print("Before cleaning:")
    print(f"Shape: {sample_df.shape}")
    print(f"Missing values: {sample_df.isnull().sum().sum()}")
    print(f"Airlines: {sample_df['airline'].unique()}")
    
    # Clean the data
    cleaned_df, report = clean_flights_data(sample_df, verbose=True)
    
    print("\nAfter cleaning:")
    print(f"Shape: {cleaned_df.shape}")
    print(f"Missing values: {cleaned_df.isnull().sum().sum()}")
    print(f"Airlines: {cleaned_df['airline'].unique()}")
    
    print(f"\nCleaning Report:")
    print(f"Records removed: {report['records_removed']}")
    print(f"Removal percentage: {report['removal_percentage']:.2f}%")
    print(f"Cleaning steps: {report['cleaning_steps']}")