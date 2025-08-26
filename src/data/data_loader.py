"""
Data Loading Module
==================

Handles loading and initial validation of airline data from various sources.
Supports multiple file formats and implements robust error handling.

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Comprehensive data loader for airline datasets with validation and error handling.
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir (str): Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.external_dir = self.data_dir / "external"
        
        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.external_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_flights_data(self, filename: str = "airlines_flights_data.csv") -> pd.DataFrame:
        """
        Load main flights dataset with comprehensive validation.
        
        Args:
            filename (str): Name of the flights data file
            
        Returns:
            pd.DataFrame: Loaded and validated flights data
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data validation fails
        """
        file_path = self.raw_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            # Load data with error handling
            logger.info(f"Loading flights data from {file_path}")
            
            # Try different encodings if UTF-8 fails
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully loaded data with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not load data with any supported encoding")
            
            # Basic validation
            self._validate_flights_data(df)
            
            logger.info(f"Successfully loaded {len(df)} records from {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading flights data: {str(e)}")
            raise
    
    def load_airport_codes(self, filename: str = "airport_codes.csv") -> pd.DataFrame:
        """
        Load airport codes reference data.
        
        Args:
            filename (str): Name of airport codes file
            
        Returns:
            pd.DataFrame: Airport codes data
        """
        file_path = self.external_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Airport codes file not found: {file_path}")
            # Return default airport data for Delhi-Mumbai route
            return self._get_default_airport_data()
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded airport codes data: {len(df)} airports")
            return df
        except Exception as e:
            logger.error(f"Error loading airport codes: {str(e)}")
            return self._get_default_airport_data()
    
    def load_processed_data(self, filename: str = "cleaned_flights_data.csv") -> pd.DataFrame:
        """
        Load processed/cleaned data.
        
        Args:
            filename (str): Name of processed data file
            
        Returns:
            pd.DataFrame: Processed flights data
        """
        file_path = self.processed_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Processed data file not found: {file_path}")
            logger.info("Loading and processing raw data instead...")
            return self.load_flights_data()
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded processed data: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            logger.info("Falling back to raw data...")
            return self.load_flights_data()
    
    def _validate_flights_data(self, df: pd.DataFrame) -> None:
        """
        Validate flights dataset structure and content.
        
        Args:
            df (pd.DataFrame): Flights data to validate
            
        Raises:
            ValueError: If validation fails
        """
        required_columns = ['airline', 'price', 'departure_time', 'duration', 'stops', 'days_left']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for completely empty dataset
        if len(df) == 0:
            raise ValueError("Dataset is empty")
        
        # Check for reasonable data ranges
        if 'price' in df.columns:
            price_issues = []
            if df['price'].min() <= 0:
                price_issues.append("Negative or zero prices found")
            if df['price'].max() > 100000:  # Reasonable upper limit for domestic flights
                price_issues.append("Extremely high prices found (>₹100,000)")
            
            if price_issues:
                warnings.warn(f"Price data issues: {'; '.join(price_issues)}")
        
        # Check duration data
        if 'duration' in df.columns:
            duration_issues = []
            if df['duration'].min() <= 0:
                duration_issues.append("Invalid duration values (≤0)")
            if df['duration'].max() > 24:  # Reasonable limit for domestic flights
                duration_issues.append("Extremely long flight durations (>24 hours)")
                
            if duration_issues:
                warnings.warn(f"Duration data issues: {'; '.join(duration_issues)}")
        
        logger.info("Data validation completed successfully")
    
    def _get_default_airport_data(self) -> pd.DataFrame:
        """
        Return default airport data for Delhi-Mumbai route.
        
        Returns:
            pd.DataFrame: Default airport codes
        """
        default_data = {
            'airport_code': ['DEL', 'BOM'],
            'airport_name': ['Indira Gandhi International Airport', 'Chhatrapati Shivaji International Airport'],
            'city': ['Delhi', 'Mumbai'],
            'state': ['Delhi', 'Maharashtra'],
            'country': ['India', 'India']
        }
        return pd.DataFrame(default_data)
    
    def get_data_info(self) -> Dict[str, Union[str, int, List[str]]]:
        """
        Get information about available data files.
        
        Returns:
            Dict: Information about data files and structure
        """
        info = {
            'raw_files': [],
            'processed_files': [],
            'external_files': [],
            'total_files': 0
        }
        
        # Check each directory
        for file_type, directory in [
            ('raw_files', self.raw_dir),
            ('processed_files', self.processed_dir),
            ('external_files', self.external_dir)
        ]:
            if directory.exists():
                files = [f.name for f in directory.glob('*.csv')]
                info[file_type] = files
        
        info['total_files'] = sum([len(files) for files in [info['raw_files'], info['processed_files'], info['external_files']]])
        
        return info
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Union[int, float, List[str]]]:
        """
        Perform comprehensive data quality check.
        
        Args:
            df (pd.DataFrame): Data to analyze
            
        Returns:
            Dict: Data quality metrics
        """
        quality_report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values_count': df.isnull().sum().sum(),
            'missing_values_percent': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            'duplicate_records': df.duplicated().sum(),
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Column-wise quality metrics
        column_quality = {}
        for col in df.columns:
            column_quality[col] = {
                'dtype': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'missing_percent': (df[col].isnull().sum() / len(df) * 100),
                'unique_values': df[col].nunique()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                column_quality[col].update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                })
        
        quality_report['column_details'] = column_quality
        
        return quality_report

# Convenience functions for quick data loading
def load_flights_data(data_dir: str = "../data") -> pd.DataFrame:
    """
    Quick function to load flights data.
    
    Args:
        data_dir (str): Path to data directory
        
    Returns:
        pd.DataFrame: Flights data
    """
    loader = DataLoader(data_dir)
    return loader.load_flights_data()

def load_processed_data(data_dir: str = "../data") -> pd.DataFrame:
    """
    Quick function to load processed data.
    
    Args:
        data_dir (str): Path to data directory
        
    Returns:
        pd.DataFrame: Processed flights data
    """
    loader = DataLoader(data_dir)
    return loader.load_processed_data()

def get_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample data for testing purposes.
    
    Args:
        n_samples (int): Number of sample records to generate
        
    Returns:
        pd.DataFrame: Sample flights data
    """
    np.random.seed(42)
    
    airlines = ['SpiceJet', 'Vistara', 'IndiGo', 'Air India', 'GoAir', 'AirAsia']
    departure_times = ['Morning', 'Afternoon', 'Evening', 'Night']
    stops = ['zero', 'one', 'two_or_more']
    
    sample_data = {
        'airline': np.random.choice(airlines, n_samples),
        'price': np.random.normal(8000, 2000, n_samples).astype(int).clip(2000, 25000),
        'departure_time': np.random.choice(departure_times, n_samples),
        'duration': np.random.normal(2.5, 0.5, n_samples).clip(1.5, 6.0),
        'stops': np.random.choice(stops, n_samples, p=[0.6, 0.3, 0.1]),
        'days_left': np.random.randint(1, 51, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some logical relationships
    # Premium airlines tend to be more expensive
    premium_airlines = ['Vistara', 'Air India']
    df.loc[df['airline'].isin(premium_airlines), 'price'] *= 1.2
    
    # Direct flights (zero stops) are more expensive
    df.loc[df['stops'] == 'zero', 'price'] *= 1.15
    
    # Flights with more stops take longer
    df.loc[df['stops'] == 'one', 'duration'] *= 1.3
    df.loc[df['stops'] == 'two_or_more', 'duration'] *= 1.6
    
    return df.round(2)

if __name__ == "__main__":
    # Example usage and testing
    loader = DataLoader()
    
    try:
        df = loader.load_flights_data()
        print(f"Loaded {len(df)} flight records")
        
        quality_report = loader.check_data_quality(df)
        print(f"Data Quality Report:")
        print(f"- Total Records: {quality_report['total_records']}")
        print(f"- Missing Values: {quality_report['missing_values_percent']:.2f}%")
        print(f"- Duplicate Records: {quality_report['duplicate_records']}")
        
    except FileNotFoundError:
        print("Sample data file not found. Generating sample data...")
        sample_df = get_sample_data(1000)
        print(f"Generated {len(sample_df)} sample records")
        print(sample_df.head())