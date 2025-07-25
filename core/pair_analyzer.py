import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from .data_loader import load_stock_data, resample_to_frequency, align_data, get_price_series

class PairAnalyzer:
    def __init__(self, ticker1, ticker2, filepath1, filepath2, resample_freq='15min'):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.filepath1 = filepath1
        self.filepath2 = filepath2
        self.resample_freq = resample_freq
        self.results = {}
        
    def load_and_prepare_data(self):

        df1_1min = load_stock_data(self.filepath1)
        df2_1min = load_stock_data(self.filepath2)
        
        df1_resampled = resample_to_frequency(df1_1min, self.resample_freq)
        df2_resampled = resample_to_frequency(df2_1min, self.resample_freq)
        
        df1_aligned, df2_aligned = align_data(df1_resampled, df2_resampled)
        
        self.price1, self.price2 = get_price_series(df1_aligned, df2_aligned, 'close')

        self.returns1 = self.price1.pct_change().dropna()
        self.returns2 = self.price2.pct_change().dropna()
        
        return len(self.price1)
    
    def run_cointegration_tests(self):
        # Basic cointegration test
        coint_t, p_value, crit_value = coint(self.price1, self.price2)
        
        # OLS regression
        X = sm.add_constant(self.price2)
        model = sm.OLS(self.price1, X).fit()
        
        # Calculate residuals
        residuals = self.price1 - (model.params['const'] + model.params[self.price2.name] * self.price2)
        
        # ADF test on residuals
        adf_result = adfuller(residuals)
        
        # Calculate half-life and AR coefficient
        lag_residuals = residuals.shift(1).dropna()
        delta_residuals = residuals.diff().dropna()
        
        common_index = lag_residuals.index.intersection(delta_residuals.index)
        lag_residuals_aligned = lag_residuals[common_index]
        delta_residuals_aligned = delta_residuals[common_index]
        
        # AR(1) model
        ar_model = sm.OLS(delta_residuals_aligned, lag_residuals_aligned)
        ar_results = ar_model.fit()
        
        # AR coefficient
        ar_coefficient = ar_results.params.iloc[0] if len(ar_results.params) > 0 else np.nan
        
        # Half-life
        halflife = -np.log(2) / ar_coefficient if ar_coefficient < 0 else np.inf
        
        self.results.update({
            'pair': f"{self.ticker1}-{self.ticker2}",
            'coint_pvalue': p_value,
            'coint_statistic': coint_t,
            'beta': model.params[self.price2.name],
            'alpha': model.params['const'],
            'r_squared': model.rsquared,
            'durbin_watson': sm.stats.stattools.durbin_watson(model.resid),
            'adf_pvalue': adf_result[1],
            'adf_statistic': adf_result[0],
            'ar_coefficient': ar_coefficient,
            'halflife': halflife,
            'return_correlation': self.returns1.corr(self.returns2),
            'num_observations': len(self.price1)
        })
        
        return self.results