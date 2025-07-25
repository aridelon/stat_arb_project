import numpy as np
import pandas as pd
from typing import Tuple, Optional


class EnhancedKalmanFilterPairs:
    def __init__(self, 
                 delta: float = 1e-5,
                 zscore_window: int = 45,
                 initial_lookback: int = 1560): 

        if delta <= 0 or delta < 1e-8:
            delta = 1e-5
            
        self.delta = delta
        self.zscore_window = zscore_window
        self.initial_lookback = initial_lookback

        self.x = np.array([0.0, 1.0])  
        

        self.P = np.eye(2)
        
        self.R = None
        self.Q = None  

        self.states = []
        self.spreads = []
        self.innovations = []
        self.innovation_variances = []
        
    def initialize_with_data(self, y_init: pd.Series, x_init: pd.Series):
        y_vals = y_init.values if hasattr(y_init, 'values') else np.array(y_init)
        x_vals = x_init.values if hasattr(x_init, 'values') else np.array(x_init)
        
        X_matrix = np.column_stack([np.ones(len(x_vals)), x_vals])
        theta_ls = np.linalg.lstsq(X_matrix, y_vals, rcond=None)[0]
        mu_ls = theta_ls[0]
        gamma_ls = theta_ls[1]
        
        residuals = y_vals - (mu_ls + gamma_ls * x_vals)
        sigma2_eps = np.var(residuals)
        
        self.R = sigma2_eps

        sigma2_x = np.var(x_vals)
        
        sigma2_mu = self.delta * sigma2_eps
        sigma2_gamma = self.delta * sigma2_eps / sigma2_x
        self.Q = np.diag([sigma2_mu, sigma2_gamma])

        self.x = np.array([mu_ls, gamma_ls])
        
        self.P = np.diag([sigma2_eps, sigma2_eps / sigma2_x])
        
    def update(self, y_t: float, x_t: float) -> Tuple[float, float, float]:
        F = np.eye(2)
        
        H = np.array([1.0, x_t])
        
        x_pred = F @ self.x 
        P_pred = F @ self.P @ F.T + self.Q
        
        y_pred = H @ x_pred
        innovation = y_t - y_pred
        
        S = H @ P_pred @ H.T + self.R
        
        K = P_pred @ H.T / S

        self.x = x_pred + K * innovation
    
        I_KH = np.eye(2) - np.outer(K, H)
        self.P = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * self.R

        mu = self.x[0]
        gamma = self.x[1]

        spread = y_t - gamma * x_t - mu

        self.states.append(self.x.copy())
        self.spreads.append(spread)
        self.innovations.append(innovation)
        self.innovation_variances.append(S)
        
        return mu, gamma, spread
    
    def fit_predict(self, y_series: pd.Series, x_series: pd.Series) -> pd.DataFrame:
        data = pd.DataFrame({'y': y_series, 'x': x_series}).dropna()
        
        # Initialize using first portion of data
        init_size = min(self.initial_lookback, len(data) // 10)
        if init_size > 30:  # Need minimum data for initialization
            y_init = data['y'].iloc[:init_size]
            x_init = data['x'].iloc[:init_size]
            self.initialize_with_data(y_init, x_init)
        else:
            self.R = 0.001
            self.Q = np.diag([self.delta, self.delta])
        
        results = {
            'mu': [],
            'gamma': [],
            'spread': [],
            'raw_spread': [],
            'innovation': []
        }
        
        spread_buffer = []
        
        for idx, row in data.iterrows():
            mu, gamma, spread = self.update(row['y'], row['x'])
            
            results['mu'].append(mu)
            results['gamma'].append(gamma)
            results['spread'].append(spread)
            results['raw_spread'].append(row['y'] - gamma * row['x'])
            results['innovation'].append(self.innovations[-1] if self.innovations else 0)
            
            spread_buffer.append(spread)
            if len(spread_buffer) > self.zscore_window:
                spread_buffer.pop(0)
        
        results_df = pd.DataFrame(results, index=data.index)

        results_df['spread_mean'] = results_df['spread'].rolling(
            window=self.zscore_window, min_periods=20).mean()
        results_df['spread_std'] = results_df['spread'].rolling(
            window=self.zscore_window, min_periods=20).std()
        results_df['zscore'] = (
            (results_df['spread'] - results_df['spread_mean']) / 
            results_df['spread_std']
        )

        results_df['innovation_variance'] = self.innovation_variances
        
        return results_df
    
    def get_diagnostics(self) -> dict:
        if not self.states:
            return {}
        
        states_array = np.array(self.states)
        innovations_array = np.array(self.innovations)
        
        return {
            'final_mu': states_array[-1, 0] if len(states_array) > 0 else 0,
            'final_gamma': states_array[-1, 1] if len(states_array) > 0 else 1,
            'mu_range': [states_array[:, 0].min(), states_array[:, 0].max()] if len(states_array) > 0 else [0, 0],
            'gamma_range': [states_array[:, 1].min(), states_array[:, 1].max()] if len(states_array) > 0 else [1, 1],
            'mu_std': states_array[:, 0].std() if len(states_array) > 0 else 0,
            'gamma_std': states_array[:, 1].std() if len(states_array) > 0 else 0,
            'delta': self.delta,
            'R_observation_variance': self.R,
            'Q_mu': self.Q[0, 0] if self.Q is not None else None,
            'Q_gamma': self.Q[1, 1] if self.Q is not None else None,
            'innovation_mean': innovations_array.mean() if len(innovations_array) > 0 else 0,
            'innovation_std': innovations_array.std() if len(innovations_array) > 0 else 0
        }

class EnhancedKalmanFilterAnalyzer:
    def __init__(self, 
                 zscore_window: int = 45,
                 delta: float = None,
                 optimize_delta: bool = True):
        self.zscore_window = zscore_window
        self.delta = delta if delta is not None else 1e-5
        self.optimize_delta = optimize_delta
        self.kf = None
        self.optimization_results = None
        
    def calculate_spread_and_zscore(self, y_series: pd.Series, 
                                   x_series: pd.Series) -> pd.DataFrame:
        if self.optimize_delta and self.delta == 1e-5:
            self.delta = self._optimize_delta(y_series, x_series)

        self.kf = EnhancedKalmanFilterPairs(
            delta=self.delta,
            zscore_window=self.zscore_window
        )
        
        results = self.kf.fit_predict(y_series, x_series)
        
        return results
    
    def analyze_pair(self, y_series: pd.Series, x_series: pd.Series) -> dict:
        results_df = self.calculate_spread_and_zscore(y_series, x_series)

        diagnostics = self.kf.get_diagnostics()
        
        zscore_clean = results_df['zscore'].replace([np.inf, -np.inf], np.nan).dropna()
        
        statistics = {
            'zscore_window_days': self.zscore_window / 26,
            'delta': self.delta,
            'mu_mean': diagnostics['final_mu'],
            'mu_range': diagnostics['mu_range'],
            'mu_std': diagnostics['mu_std'],
            'beta_mean': results_df['gamma'].mean(),
            'beta_std': diagnostics['gamma_std'],
            'beta_min': diagnostics['gamma_range'][0],
            'beta_max': diagnostics['gamma_range'][1],
            'zscore_mean': zscore_clean.mean(),
            'zscore_std': zscore_clean.std(),
            'zscore_max_abs': zscore_clean.abs().max(),
            'pct_within_2sigma': (zscore_clean.abs() <= 2).sum() / len(zscore_clean) * 100,
            'innovation_mean': diagnostics['innovation_mean'],
            'innovation_std': diagnostics['innovation_std']
        }
        
        return {
            'betas': results_df['gamma'].rename('beta'),
            'spread': results_df['spread'],
            'zscore': results_df['zscore'],
            'mu': results_df['mu'],
            'raw_spread': results_df['raw_spread'],
            'statistics': statistics
        }
    
    def _optimize_delta(self, y_series: pd.Series, x_series: pd.Series) -> float:
        mask_2021 = y_series.index.year == 2021
        if mask_2021.any():
            y_opt = y_series[mask_2021]
            x_opt = x_series[mask_2021]
        else:
            split_idx = int(len(y_series) * 0.3)
            y_opt = y_series.iloc[:split_idx]
            x_opt = x_series.iloc[:split_idx]
        
        delta_values = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        best_score = np.inf
        best_delta = 1e-5
        
        
        for delta in delta_values:
            try:
                kf_test = EnhancedKalmanFilterPairs(
                    delta=delta,
                    zscore_window=self.zscore_window
                )
                results = kf_test.fit_predict(y_opt, x_opt)
                
                zscore_clean = results['zscore'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(zscore_clean) > 100: 
                    max_abs_z = zscore_clean.abs().max()
                    pct_within_2sig = (zscore_clean.abs() <= 2).mean()

                    if max_abs_z < 20:
                        score = max_abs_z - 10 * pct_within_2sig
                        
                        if score < best_score:
                            best_score = score
                            best_delta = delta
                            print(f"  δ={delta:.2e}: max|z|={max_abs_z:.1f}, within±2σ={pct_within_2sig:.1%}, score={score:.2f}")
                        
            except Exception as e:
                print(f"  δ={delta:.2e}: failed - {str(e)[:50]}")
                continue

        if best_delta < 1e-8:
            best_delta = 1e-6
        
        return best_delta