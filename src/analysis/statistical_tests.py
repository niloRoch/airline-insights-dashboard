"""
Statistical Tests Module

This module provides comprehensive statistical hypothesis testing
for the Airlines Dataset, including normality tests, correlation analysis,
ANOVA, t-tests, and non-parametric alternatives.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from scipy.stats import (
    normaltest, shapiro, levene, bartlett, 
    ttest_ind, mannwhitneyu, f_oneway, kruskal,
    chi2_contingency, pearsonr, spearmanr,
    jarque_bera, anderson
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.formula.api import ols
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


class StatisticalTester:
    """
    Main class for statistical hypothesis testing.
    """
    
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        """
        Initialize the statistical tester.
        
        Args:
            data: DataFrame containing flight information
            alpha: Significance level for tests (default: 0.05)
        """
        self.data = data.copy()
        self.alpha = alpha
        self.results = {}
        
    def test_all_normality(self, variables: List[str] = None) -> Dict[str, Dict]:
        """
        Test normality for all specified variables.
        
        Args:
            variables: List of variable names to test
            
        Returns:
            Dictionary with normality test results
        """
        if variables is None:
            variables = ['price', 'duration', 'days_left']
            
        normality_results = {}
        
        for var in variables:
            if var in self.data.columns:
                normality_results[var] = test_normality(
                    self.data[var], var, self.alpha
                )
                
        self.results['normality'] = normality_results
        return normality_results
    
    def correlation_analysis(self, variables: List[str] = None) -> Dict[str, Dict]:
        """
        Comprehensive correlation analysis.
        
        Args:
            variables: List of variables to analyze
            
        Returns:
            Dictionary with correlation results
        """
        if variables is None:
            variables = ['price', 'duration', 'days_left']
            
        # Correlation matrix
        corr_matrix = self.data[variables].corr()
        
        # Test significance of correlations
        correlation_tests = {}
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:  # Avoid duplicates
                    pair = f"{var1}_vs_{var2}"
                    correlation_tests[pair] = correlation_significance(
                        self.data, var1, var2, self.alpha
                    )
        
        results = {
            'correlation_matrix': corr_matrix,
            'significance_tests': correlation_tests
        }
        
        self.results['correlation'] = results
        return results
    
    def compare_airlines(self, target_variable: str = 'price') -> Dict[str, Any]:
        """
        Compare airlines using ANOVA and post-hoc tests.
        
        Args:
            target_variable: Variable to compare across airlines
            
        Returns:
            Dictionary with comparison results
        """
        # Prepare data
        airline_groups = [
            self.data[self.data['airline'] == airline][target_variable].values
            for airline in self.data['airline'].unique()
        ]
        airline_names = self.data['airline'].unique()
        
        # Test homogeneity of variances
        levene_stat, levene_p = levene(*airline_groups)
        bartlett_stat, bartlett_p = bartlett(*airline_groups)
        
        # ANOVA
        f_stat, f_p = f_oneway(*airline_groups)
        
        # Kruskal-Wallis (non-parametric alternative)
        kw_stat, kw_p = kruskal(*airline_groups)
        
        results = {
            'homogeneity_tests': {
                'levene': {'statistic': levene_stat, 'p_value': levene_p, 'homogeneous': levene_p > self.alpha},
                'bartlett': {'statistic': bartlett_stat, 'p_value': bartlett_p, 'homogeneous': bartlett_p > self.alpha}
            },
            'anova': {
                'f_statistic': f_stat,
                'p_value': f_p,
                'significant': f_p < self.alpha
            },
            'kruskal_wallis': {
                'h_statistic': kw_stat,
                'p_value': kw_p,
                'significant': kw_p < self.alpha
            },
            'post_hoc': None
        }
        
        # Post-hoc test if significant
        if f_p < self.alpha:
            tukey_data = []
            tukey_labels = []
            
            for airline in self.data['airline'].unique():
                values = self.data[self.data['airline'] == airline][target_variable].values
                tukey_data.extend(values)
                tukey_labels.extend([airline] * len(values))
            
            tukey_result = pairwise_tukeyhsd(tukey_data, tukey_labels, alpha=self.alpha)
            results['post_hoc'] = {
                'method': 'Tukey HSD',
                'results': str(tukey_result),
                'summary': tukey_result.summary()
            }
        
        self.results['airline_comparison'] = results
        return results
    
    def compare_flight_types(self, target_variable: str = 'price') -> Dict[str, Any]:
        """
        Compare direct vs connecting flights.
        
        Args:
            target_variable: Variable to compare
            
        Returns:
            Dictionary with comparison results
        """
        # Separate groups
        direct_flights = self.data[self.data['stops'] == 'zero'][target_variable].values
        flights_with_stops = self.data[self.data['stops'] == 'one'][target_variable].values
        
        if len(direct_flights) == 0 or len(flights_with_stops) == 0:
            return {'error': 'Insufficient data for comparison'}
        
        # Descriptive statistics
        desc_stats = {
            'direct': {
                'n': len(direct_flights),
                'mean': np.mean(direct_flights),
                'std': np.std(direct_flights),
                'median': np.median(direct_flights)
            },
            'with_stops': {
                'n': len(flights_with_stops),
                'mean': np.mean(flights_with_stops),
                'std': np.std(flights_with_stops),
                'median': np.median(flights_with_stops)
            }
        }
        
        # Test homogeneity of variances
        levene_stat, levene_p = levene(direct_flights, flights_with_stops)
        equal_var = levene_p > self.alpha
        
        # T-test
        t_stat, t_p = ttest_ind(direct_flights, flights_with_stops, equal_var=equal_var)
        
        # Mann-Whitney U (non-parametric)
        u_stat, u_p = mannwhitneyu(direct_flights, flights_with_stops, alternative='two-sided')
        
        # Effect size (Cohen's d)
        effect_size = cohens_d(direct_flights, flights_with_stops)
        
        results = {
            'descriptive_stats': desc_stats,
            'homogeneity_test': {
                'levene_statistic': levene_stat,
                'levene_p': levene_p,
                'equal_variances': equal_var
            },
            't_test': {
                'statistic': t_stat,
                'p_value': t_p,
                'significant': t_p < self.alpha,
                'equal_var_assumed': equal_var
            },
            'mann_whitney': {
                'statistic': u_stat,
                'p_value': u_p,
                'significant': u_p < self.alpha
            },
            'effect_size': {
                'cohens_d': effect_size,
                'interpretation': interpret_cohens_d(effect_size)
            }
        }
        
        self.results['flight_type_comparison'] = results
        return results
    
    def analyze_time_effects(self, target_variable: str = 'price') -> Dict[str, Any]:
        """
        Analyze effects of departure time on target variable.
        
        Args:
            target_variable: Variable to analyze
            
        Returns:
            Dictionary with time analysis results
        """
        # Prepare data by time slots
        time_groups = [
            self.data[self.data['departure_time'] == time][target_variable].values
            for time in self.data['departure_time'].unique()
        ]
        time_names = self.data['departure_time'].unique()
        
        # ANOVA for departure times
        f_stat, f_p = f_oneway(*time_groups)
        kw_stat, kw_p = kruskal(*time_groups)
        
        # Test homogeneity
        levene_stat, levene_p = levene(*time_groups)
        
        results = {
            'time_slots': list(time_names),
            'descriptive_stats': {
                time: {
                    'mean': self.data[self.data['departure_time'] == time][target_variable].mean(),
                    'std': self.data[self.data['departure_time'] == time][target_variable].std(),
                    'count': len(self.data[self.data['departure_time'] == time])
                }
                for time in time_names
            },
            'anova': {
                'f_statistic': f_stat,
                'p_value': f_p,
                'significant': f_p < self.alpha
            },
            'kruskal_wallis': {
                'h_statistic': kw_stat,
                'p_value': kw_p,
                'significant': kw_p < self.alpha
            },
            'homogeneity': {
                'levene_statistic': levene_stat,
                'p_value': levene_p,
                'homogeneous': levene_p > self.alpha
            }
        }
        
        self.results['time_effects'] = results
        return results
    
    def regression_analysis(self, target: str = 'price') -> Dict[str, Any]:
        """
        Multiple regression analysis.
        
        Args:
            target: Target variable for regression
            
        Returns:
            Dictionary with regression results
        """
        # Prepare variables
        X = self.data[['duration', 'days_left']].copy()
        y = self.data[target].copy()
        
        # Add dummy variables for categorical variables
        airline_dummies = pd.get_dummies(self.data['airline'], prefix='airline')
        stops_dummies = pd.get_dummies(self.data['stops'], prefix='stops')
        time_dummies = pd.get_dummies(self.data['departure_time'], prefix='time')
        
        # Combine (drop one category each to avoid multicollinearity)
        X_full = pd.concat([
            X,
            airline_dummies.iloc[:, :-1],
            stops_dummies.iloc[:, :-1],
            time_dummies.iloc[:, :-1]
        ], axis=1)
        
        # Add constant
        X_full_const = sm.add_constant(X_full)
        
        # Fit model
        model = sm.OLS(y, X_full_const).fit()
        
        # Diagnostic tests
        residuals = model.resid
        
        # Normality of residuals
        shapiro_stat, shapiro_p = shapiro(residuals)
        
        # Heteroscedasticity test
        bp_stat, bp_p, bp_f_stat, bp_f_p = het_breuschpagan(model.resid, model.model.exog)
        
        results = {
            'model_summary': {
                'r_squared': model.rsquared,
                'r_squared_adj': model.rsquared_adj,
                'aic': model.aic,
                'bic': model.bic,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue
            },
            'coefficients': {
                'values': model.params.to_dict(),
                'pvalues': model.pvalues.to_dict(),
                'conf_int': model.conf_int().to_dict()
            },
            'diagnostics': {
                'residual_normality': {
                    'shapiro_stat': shapiro_stat,
                    'shapiro_p': shapiro_p,
                    'normal': shapiro_p > self.alpha
                },
                'heteroscedasticity': {
                    'bp_statistic': bp_stat,
                    'bp_p': bp_p,
                    'homoscedastic': bp_p > self.alpha
                },
                'residual_stats': {
                    'mean': np.mean(residuals),
                    'std': np.std(residuals)
                }
            }
        }
        
        self.results['regression'] = results
        return results
    
    def chi_square_tests(self) -> Dict[str, Any]:
        """
        Chi-square tests for categorical variable associations.
        
        Returns:
            Dictionary with chi-square test results
        """
        results = {}
        
        # Airline vs Stops
        if 'airline' in self.data.columns and 'stops' in self.data.columns:
            contingency_airline_stops = pd.crosstab(self.data['airline'], self.data['stops'])
            chi2_stat, chi2_p, chi2_dof, chi2_expected = chi2_contingency(contingency_airline_stops)
            
            # Contingency coefficient
            n = contingency_airline_stops.sum().sum()
            contingency_coeff = np.sqrt(chi2_stat / (chi2_stat + n))
            
            results['airline_vs_stops'] = {
                'chi2_statistic': chi2_stat,
                'p_value': chi2_p,
                'degrees_of_freedom': chi2_dof,
                'significant': chi2_p < self.alpha,
                'contingency_coefficient': contingency_coeff,
                'contingency_table': contingency_airline_stops
            }
        
        # Departure time vs Price category
        if 'departure_time' in self.data.columns:
            # Create price categories
            price_categories = pd.qcut(self.data['price'], q=3, labels=['Low', 'Medium', 'High'])
            contingency_time_price = pd.crosstab(self.data['departure_time'], price_categories)
            chi2_stat, chi2_p, chi2_dof, chi2_expected = chi2_contingency(contingency_time_price)
            
            results['time_vs_price_category'] = {
                'chi2_statistic': chi2_stat,
                'p_value': chi2_p,
                'degrees_of_freedom': chi2_dof,
                'significant': chi2_p < self.alpha,
                'contingency_table': contingency_time_price
            }
        
        self.results['chi_square'] = results
        return results
    
    def get_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete statistical analysis.
        
        Returns:
            Dictionary with all analysis results
        """
        complete_results = {}
        
        # Run all tests
        complete_results['normality'] = self.test_all_normality()
        complete_results['correlation'] = self.correlation_analysis()
        complete_results['airline_comparison'] = self.compare_airlines()
        complete_results['flight_type_comparison'] = self.compare_flight_types()
        complete_results['time_effects'] = self.analyze_time_effects()
        complete_results['regression'] = self.regression_analysis()
        complete_results['chi_square'] = self.chi_square_tests()
        
        # Add summary
        complete_results['summary'] = self._generate_analysis_summary()
        
        return complete_results
    
    def _generate_analysis_summary(self) -> Dict[str, str]:
        """Generate summary of key findings."""
        summary = {}
        
        if 'normality' in self.results:
            non_normal = [var for var, result in self.results['normality'].items() 
                         if not all(test['is_normal'] for test in result.values() if isinstance(test, dict) and 'is_normal' in test)]
            summary['normality'] = f"Non-normal variables: {', '.join(non_normal) if non_normal else 'All variables approximately normal'}"
        
        if 'airline_comparison' in self.results:
            significant = self.results['airline_comparison']['anova']['significant']
            summary['airline_differences'] = f"Airline price differences: {'Significant' if significant else 'Not significant'}"
        
        if 'flight_type_comparison' in self.results:
            significant = self.results['flight_type_comparison']['t_test']['significant']
            summary['flight_type_differences'] = f"Direct vs connecting flight prices: {'Significantly different' if significant else 'No significant difference'}"
        
        return summary


def test_normality(data: pd.Series, variable_name: str, alpha: float = 0.05) -> Dict[str, Dict]:
    """
    Test normality using multiple tests.
    
    Args:
        data: Series of data to test
        variable_name: Name of the variable
        alpha: Significance level
        
    Returns:
        Dictionary with normality test results
    """
    clean_data = data.dropna()
    n = len(clean_data)
    
    results = {}
    
    # Shapiro-Wilk (best for n < 5000)
    if n <= 5000:
        shapiro_stat, shapiro_p = shapiro(clean_data)
        results['Shapiro-Wilk'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': shapiro_p > alpha
        }
    
    # D'Agostino-Pearson
    dagostino_stat, dagostino_p = normaltest(clean_data)
    results['D_Agostino-Pearson'] = {
        'statistic': dagostino_stat,
        'p_value': dagostino_p,
        'is_normal': dagostino_p > alpha
    }
    
    # Jarque-Bera
    jb_stat, jb_p = jarque_bera(clean_data)
    results['Jarque-Bera'] = {
        'statistic': jb_stat,
        'p_value': jb_p,
        'is_normal': jb_p > alpha
    }
    
    # Anderson-Darling
    ad_result = anderson(clean_data, dist='norm')
    ad_critical = ad_result.critical_values[2]  # 5% significance level
    ad_is_normal = ad_result.statistic < ad_critical
    results['Anderson-Darling'] = {
        'statistic': ad_result.statistic,
        'critical_value': ad_critical,
        'is_normal': ad_is_normal
    }
    
    return results


def correlation_significance(df: pd.DataFrame, col1: str, col2: str, alpha: float = 0.05) -> Dict[str, Dict]:
    """
    Test significance of correlation between two variables.
    
    Args:
        df: DataFrame containing the data
        col1: First variable
        col2: Second variable
        alpha: Significance level
        
    Returns:
        Dictionary with correlation test results
    """
    # Remove NaNs
    common_idx = df[[col1, col2]].dropna().index
    data1 = df.loc[common_idx, col1]
    data2 = df.loc[common_idx, col2]
    
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(data1, data2)
    
    # Spearman correlation (non-parametric)
    spearman_r, spearman_p = spearmanr(data1, data2)
    
    return {
        'pearson': {
            'r': pearson_r,
            'p': pearson_p,
            'significant': pearson_p < alpha
        },
        'spearman': {
            'r': spearman_r,
            'p': spearman_p,
            'significant': spearman_p < alpha
        }
    }


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        x: First group data
        y: Second group data
        
    Returns:
        Cohen's d value
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "small effect"
    elif abs_d < 0.5:
        return "medium effect"
    elif abs_d < 0.8:
        return "large effect"
    else:
        return "very large effect"


def compare_groups(data: pd.DataFrame, group_col: str, target_col: str = 'price', alpha: float = 0.05) -> Dict[str, Any]:
    """
    Compare groups using appropriate statistical tests.
    
    Args:
        data: DataFrame containing the data
        group_col: Column defining groups
        target_col: Target variable to compare
        alpha: Significance level
        
    Returns:
        Dictionary with comparison results
    """
    tester = StatisticalTester(data, alpha)
    
    if group_col == 'airline':
        return tester.compare_airlines(target_col)
    elif group_col == 'stops':
        return tester.compare_flight_types(target_col)
    elif group_col == 'departure_time':
        return tester.analyze_time_effects(target_col)
    else:
        # Generic group comparison
        unique_groups = data[group_col].unique()
        if len(unique_groups) == 2:
            # Two-group comparison
            group1 = data[data[group_col] == unique_groups[0]][target_col].values
            group2 = data[data[group_col] == unique_groups[1]][target_col].values
            
            # Test equal variances
            levene_stat, levene_p = levene(group1, group2)
            equal_var = levene_p > alpha
            
            # T-test
            t_stat, t_p = ttest_ind(group1, group2, equal_var=equal_var)
            
            # Mann-Whitney U
            u_stat, u_p = mannwhitneyu(group1, group2, alternative='two-sided')
            
            return {
                'test_type': 'two_group',
                't_test': {'statistic': t_stat, 'p_value': t_p, 'significant': t_p < alpha},
                'mann_whitney': {'statistic': u_stat, 'p_value': u_p, 'significant': u_p < alpha},
                'equal_variances': equal_var
            }
        else:
            # Multi-group comparison
            groups = [data[data[group_col] == group][target_col].values for group in unique_groups]
            
            # ANOVA
            f_stat, f_p = f_oneway(*groups)
            
            # Kruskal-Wallis
            kw_stat, kw_p = kruskal(*groups)
            
            return {
                'test_type': 'multi_group',
                'anova': {'f_statistic': f_stat, 'p_value': f_p, 'significant': f_p < alpha},
                'kruskal_wallis': {'h_statistic': kw_stat, 'p_value': kw_p, 'significant': kw_p < alpha}
            }


def anova_analysis(data: pd.DataFrame, group_col: str, target_col: str = 'price') -> Dict[str, Any]:
    """
    Perform ANOVA analysis with post-hoc tests.
    
    Args:
        data: DataFrame containing the data
        group_col: Grouping variable
        target_col: Target variable
        
    Returns:
        Dictionary with ANOVA results
    """
    tester = StatisticalTester(data)
    
    if group_col == 'airline':
        return tester.compare_airlines(target_col)
    elif group_col == 'departure_time':
        return tester.analyze_time_effects(target_col)
    else:
        return compare_groups(data, group_col, target_col)