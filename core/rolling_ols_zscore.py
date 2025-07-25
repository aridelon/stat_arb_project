import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

class RollingOLSAnalyzer:
    def __init__(self, ols_window=390, zscore_window=None, min_observations=100):
        self.ols_window = ols_window
        self.zscore_window = zscore_window if zscore_window is not None else ols_window
        self.min_observations = min_observations
        self.bars_per_day = 26 
        
        self.ols_window_days = self.ols_window / self.bars_per_day
        self.zscore_window_days = self.zscore_window / self.bars_per_day
        
    def calculate_rolling_beta(self, price1, price2, method='ols'):
        common_index = price1.index.intersection(price2.index)
        price1 = price1[common_index]
        price2 = price2[common_index]
        
        betas = pd.Series(index=common_index, dtype=float)
        alphas = pd.Series(index=common_index, dtype=float)
        
        for i in range(self.ols_window, len(price1)):
            start_idx = i - self.ols_window
            
            y = price1.iloc[start_idx:i].values
            x = price2.iloc[start_idx:i].values
            
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                continue
                
            try:
                if method == 'ols':
                    X = sm.add_constant(x)
                    model = OLS(y, X).fit()
                    betas.iloc[i] = model.params[1]
                    alphas.iloc[i] = model.params[0]
                elif method == 'robust':
                    X = sm.add_constant(x)
                    model = sm.RLM(y, X).fit()
                    betas.iloc[i] = model.params[1]
                    alphas.iloc[i] = model.params[0]
                    
            except Exception as e:
                continue
        
        betas = betas.fillna(method='ffill')
        alphas = alphas.fillna(method='ffill')
        
        return betas, alphas
    
    def calculate_rolling_spread(self, price1, price2, betas):
        common_index = price1.index.intersection(price2.index).intersection(betas.index)
        
        price1_aligned = price1[common_index]
        price2_aligned = price2[common_index]
        betas_aligned = betas[common_index]
        
        spread = price1_aligned - betas_aligned * price2_aligned
        
        return spread
    
    def calculate_rolling_zscore(self, spread, custom_zscore_window=None):
        zscore_window = custom_zscore_window if custom_zscore_window is not None else self.zscore_window
            
        rolling_mean = spread.rolling(window=zscore_window, min_periods=self.min_observations).mean()
        rolling_std = spread.rolling(window=zscore_window, min_periods=self.min_observations).std()
        
        zscore = (spread - rolling_mean) / rolling_std
        
        zscore = zscore.replace([np.inf, -np.inf], np.nan)
        zscore = zscore.fillna(0)
        
        return zscore
    
    def analyze_pair(self, price1, price2, method='ols'):
        betas, alphas = self.calculate_rolling_beta(price1, price2, method)

        spread = self.calculate_rolling_spread(price1, price2, betas)

        zscore = self.calculate_rolling_zscore(spread)

        valid_betas = betas.dropna()
        valid_zscore = zscore.dropna()
        
        statistics = {
            'beta_mean': valid_betas.mean(),
            'beta_std': valid_betas.std(),
            'beta_min': valid_betas.min(),
            'beta_max': valid_betas.max(),
            'zscore_mean': valid_zscore.mean(),
            'zscore_std': valid_zscore.std(),
            'zscore_max_abs': valid_zscore.abs().max(),
            'pct_within_2sigma': (valid_zscore.abs() <= 2).sum() / len(valid_zscore) * 100,
            'ols_window_days': self.ols_window_days,
            'zscore_window_days': self.zscore_window_days
        }
        
        return {
            'betas': betas,
            'alphas': alphas,
            'spread': spread,
            'zscore': zscore,
            'statistics': statistics
        }
    
    def get_trading_signals(self, zscore, entry_threshold=2.0, exit_threshold=0.0):
        signals = pd.Series(0, index=zscore.index)
        position = 0
        
        for i in range(len(zscore)):
            z = zscore.iloc[i]
            
            if position == 0:  
                if z > entry_threshold:
                    position = -1  
                elif z < -entry_threshold:
                    position = 1  
                    
            elif position == 1:  
                if z >= exit_threshold:
                    position = 0 
                    
            elif position == -1:  
                if z <= -exit_threshold:
                    position = 0  

            signals.iloc[i] = position
            
        return signals