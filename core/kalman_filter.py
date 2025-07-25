import numpy as np
import pandas as pd
from typing import Tuple, Optional


class KalmanFilterPairs:
    def __init__(self, 
                 initial_beta: float = 1.0,
                 initial_variance: float = 1.0,
                 process_variance: float = 0.0001,
                 observation_variance: float = 0.001,
                 zscore_window: int = 45):

        self.beta = initial_beta
        self.P = initial_variance  
        self.Q = process_variance
        self.R = observation_variance
        self.zscore_window = zscore_window
        
        self.betas = []
        self.spreads = []
        self.innovations = []
        self.innovation_variances = []
        
    def update(self, y: float, x: float) -> Tuple[float, float, float]:
        beta_pred = self.beta  
        P_pred = self.P + self.Q 
        
        innovation = y - beta_pred * x
        innovation_var = x**2 * P_pred + self.R
        
        K = P_pred * x / innovation_var
        
        self.beta = beta_pred + K * innovation
        self.P = (1 - K * x) * P_pred
    
        spread = y - self.beta * x
        
        self.betas.append(self.beta)
        self.spreads.append(spread)
        self.innovations.append(innovation)
        self.innovation_variances.append(innovation_var)
        
        return self.beta, spread, innovation
    
    def fit_predict(self, y_series: pd.Series, x_series: pd.Series) -> pd.DataFrame:
        data = pd.DataFrame({'y': y_series, 'x': x_series}).dropna()
        
        if len(data) > 0:
            self.beta = np.cov(data['y'].iloc[:min(60, len(data))], 
                              data['x'].iloc[:min(60, len(data))])[0, 1] / \
                       np.var(data['x'].iloc[:min(60, len(data))])
        
        for idx, row in data.iterrows():
            self.update(row['y'], row['x'])

        results = pd.DataFrame({
            'beta': self.betas,
            'spread': self.spreads,
            'innovation': self.innovations
        }, index=data.index)
        
        results['spread_mean'] = results['spread'].rolling(
            window=self.zscore_window, min_periods=1).mean()
        results['spread_std'] = results['spread'].rolling(
            window=self.zscore_window, min_periods=1).std()
        results['zscore'] = (results['spread'] - results['spread_mean']) / results['spread_std']

        results['innovation_variance'] = self.innovation_variances
        
        return results
    
    def optimize_parameters(self, y_series: pd.Series, x_series: pd.Series,
                          validation_split: float = 0.3) -> dict:
        n = len(y_series)
        train_size = int(n * (1 - validation_split))
        
        Q_values = np.logspace(-6, -2, 20)  
        R_values = np.logspace(-4, -1, 20)
        
        best_likelihood = -np.inf
        best_params = {}
        
        for Q in Q_values:
            for R in R_values:
                self.__init__(process_variance=Q, 
                            observation_variance=R,
                            zscore_window=self.zscore_window)
                
                train_results = self.fit_predict(
                    y_series.iloc[:train_size],
                    x_series.iloc[:train_size]
                )
                
                val_y = y_series.iloc[train_size:].values
                val_x = x_series.iloc[train_size:].values
                
                log_likelihood = 0
                for i in range(len(val_y)):
                    _, _, innovation = self.update(val_y[i], val_x[i])
                    innovation_var = self.innovation_variances[-1]
                    log_likelihood += -0.5 * (np.log(2 * np.pi * innovation_var) + 
                                            innovation**2 / innovation_var)
                
                if log_likelihood > best_likelihood:
                    best_likelihood = log_likelihood
                    best_params = {
                        'Q': Q,
                        'R': R,
                        'log_likelihood': log_likelihood,
                        'train_size': train_size
                    }
        
        self.__init__(process_variance=best_params['Q'],
                     observation_variance=best_params['R'],
                     zscore_window=self.zscore_window)
        
        return best_params


class KalmanFilterAnalyzer:
    def __init__(self, zscore_window: int = 45,
                 process_variance: Optional[float] = None,
                 observation_variance: Optional[float] = None,
                 optimize_params: bool = True):

        self.zscore_window = zscore_window
        self.Q = process_variance if process_variance is not None else 0.0001
        self.R = observation_variance if observation_variance is not None else 0.001
        self.optimize_params = optimize_params
        self.kf = None
        self.optimization_results = None
        
    def calculate_spread_and_zscore(self, y_series: pd.Series, 
                                   x_series: pd.Series) -> pd.DataFrame:
        self.kf = KalmanFilterPairs(
            process_variance=self.Q,
            observation_variance=self.R,
            zscore_window=self.zscore_window
        )
        
        if self.optimize_params and self.Q == 0.0001 and self.R == 0.001:
            self.optimization_results = self.kf.optimize_parameters(y_series, x_series)
            self.Q = self.optimization_results['Q']
            self.R = self.optimization_results['R']

        results = self.kf.fit_predict(y_series, x_series)
        
        return results
    
    def get_diagnostics(self) -> dict:
        if self.kf is None:
            return {}
        
        diagnostics = {
            'final_beta': self.kf.betas[-1] if self.kf.betas else None,
            'beta_std': np.std(self.kf.betas) if self.kf.betas else None,
            'process_variance_Q': self.Q,
            'observation_variance_R': self.R,
            'optimization_results': self.optimization_results
        }
        
        if self.kf.innovations:
            innovations = np.array(self.kf.innovations)
            diagnostics['innovation_mean'] = np.mean(innovations)
            diagnostics['innovation_std'] = np.std(innovations)
            diagnostics['innovation_autocorr'] = np.corrcoef(
                innovations[:-1], innovations[1:])[0, 1] if len(innovations) > 1 else None
        
        return diagnostics