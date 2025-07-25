import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')

class RollingJohansenAnalyzer:
    def __init__(self, 
                 johansen_window=3276,      # 126 trading days (6 months) - longer for stability
                 zscore_window=1170,        # 45 trading days
                 min_observations=500,
                 significance_level=0.05,
                 auto_lag_selection=True,
                 smoothing_alpha=0.02,      # Very strong smoothing
                 max_weight_change=0.02,    # Max 2% change per update
                 update_frequency=26,       # Update daily, not every bar
                 eigenvalue_threshold=0.005, # Lower threshold
                 rank_persistence=5,        # Updates before rank change (5 days)
                 adaptive_smoothing=True):  # Use eigenvalue-based smoothing
        
        self.johansen_window = johansen_window
        self.zscore_window = zscore_window
        self.min_observations = min_observations
        self.significance_level = significance_level
        self.auto_lag_selection = auto_lag_selection
        self.smoothing_alpha = smoothing_alpha
        self.max_weight_change = max_weight_change
        self.update_frequency = update_frequency
        self.eigenvalue_threshold = eigenvalue_threshold
        self.rank_persistence = rank_persistence
        self.adaptive_smoothing = adaptive_smoothing
        self.bars_per_day = 26
        
        # Convert to days for display
        self.johansen_window_days = self.johansen_window / self.bars_per_day
        self.zscore_window_days = self.zscore_window / self.bars_per_day
        
        # State tracking
        self.previous_weights = None
        self.previous_rank = None
        self.rank_change_counter = 0
        self.weight_history = []
        self.spread_scale_factor = None
        
    def select_optimal_lag(self, data_matrix, maxlags=10):
        """Select optimal lag using BIC with stability constraints"""
        from statsmodels.tsa.api import VAR
        
        maxlags = min(maxlags, int(data_matrix.shape[0] / 20))
        
        try:
            model = VAR(data_matrix)
            lag_selection = model.select_order(maxlags=maxlags)
            optimal_lag = lag_selection.bic
        except:
            optimal_lag = 2
            
        # Cap at 4 for stability with 4-asset portfolios
        return max(1, min(optimal_lag, 4))
    
    def align_eigenvector_signs(self, current_evec, reference_weights):
        """Enhanced eigenvector alignment using multiple criteria"""
        if reference_weights is None:
            # First iteration - use consistent sign convention
            max_idx = np.argmax(np.abs(current_evec))
            if current_evec[max_idx] < 0:
                return -current_evec
            return current_evec
        
        # Method 1: Direct correlation
        dot_product = np.dot(current_evec, reference_weights)
        
        # Method 2: Check sum of absolute differences
        diff_positive = np.sum(np.abs(current_evec - reference_weights))
        diff_negative = np.sum(np.abs(-current_evec - reference_weights))
        
        # Use both criteria
        if dot_product < 0 or diff_negative < diff_positive:
            return -current_evec
            
        return current_evec
    
    def adaptive_smooth_weights(self, new_weights, previous_weights, eigenvalue):
        """Apply adaptive smoothing based on eigenvalue strength"""
        if previous_weights is None:
            return new_weights
        
        if self.adaptive_smoothing:
            # Weaker eigenvalue = less confidence = more smoothing
            # Scale smoothing between 0.01 and 0.05 based on eigenvalue
            confidence = min(1.0, eigenvalue / 0.02)  # Normalize by typical eigenvalue
            adaptive_alpha = self.smoothing_alpha * (1 + 2 * (1 - confidence))
            adaptive_alpha = np.clip(adaptive_alpha, 0.01, 0.1)
        else:
            adaptive_alpha = self.smoothing_alpha
        
        # Apply smoothing
        smoothed = adaptive_alpha * new_weights + (1 - adaptive_alpha) * previous_weights
        
        # Renormalize
        max_idx = np.argmax(np.abs(smoothed))
        if abs(smoothed[max_idx]) > 1e-8:
            smoothed = smoothed / smoothed[max_idx]
        
        return smoothed
    
    def constrain_weight_changes(self, new_weights, previous_weights):
        """Apply strict constraints on weight changes"""
        if previous_weights is None:
            return new_weights
        
        # Calculate proposed changes
        changes = new_weights - previous_weights
        
        # Apply L2 constraint on total change
        total_change = np.sqrt(np.sum(changes**2))
        if total_change > self.max_weight_change * np.sqrt(len(changes)):
            # Scale down changes proportionally
            changes = changes * (self.max_weight_change * np.sqrt(len(changes)) / total_change)
        
        # Apply individual constraints
        changes = np.clip(changes, -self.max_weight_change, self.max_weight_change)
        
        # Apply constrained changes
        constrained_weights = previous_weights + changes
        
        # Renormalize
        max_idx = np.argmax(np.abs(constrained_weights))
        if abs(constrained_weights[max_idx]) > 1e-8:
            return constrained_weights / constrained_weights[max_idx]
        return constrained_weights
    
    def determine_stable_rank(self, trace_stats, crit_vals, eigenvalues):
        """Determine rank with enhanced stability"""
        # Count significant eigenvalues
        current_rank = 0
        for i in range(len(trace_stats)):
            if (trace_stats[i] > crit_vals[i, 1] and 
                eigenvalues[i] > self.eigenvalue_threshold):
                current_rank = i + 1
            else:
                break
        
        # Apply persistence logic
        if self.previous_rank is not None:
            if current_rank != self.previous_rank:
                self.rank_change_counter += 1
                if self.rank_change_counter < self.rank_persistence:
                    return self.previous_rank
                else:
                    self.rank_change_counter = 0
                    return current_rank
            else:
                self.rank_change_counter = 0
                return current_rank
        
        return current_rank
    
    def calculate_rolling_johansen(self, prices_df):
        """Calculate rolling Johansen with daily updates"""
        n_assets = len(prices_df.columns)
        n_periods = len(prices_df)
        
        # Initialize storage
        weights = pd.DataFrame(index=prices_df.index, 
                             columns=prices_df.columns, 
                             dtype=float)
        ranks = pd.Series(index=prices_df.index, dtype=float)
        eigenvalues = pd.DataFrame(index=prices_df.index, 
                                 columns=[f'eigenvalue_{i}' for i in range(n_assets)], 
                                 dtype=float)
        lag_orders = pd.Series(index=prices_df.index, dtype=float)
        
        # Reset state
        self.previous_weights = None
        self.previous_rank = None
        self.rank_change_counter = 0
        self.weight_history = []
        
        print(f"Running rolling Johansen analysis with {self.johansen_window_days:.1f} day window...")
        print(f"Updating every {self.update_frequency} bars ({self.update_frequency/26:.1f} days)...")
        
        # Calculate update points
        update_points = list(range(self.johansen_window, n_periods, self.update_frequency))
        if update_points[-1] < n_periods - 1:
            update_points.append(n_periods - 1)
        
        total_updates = len(update_points)
        
        # Main rolling window loop - update less frequently
        for update_idx, i in enumerate(update_points):
            if update_idx % max(1, total_updates // 20) == 0:
                progress = update_idx / total_updates * 100
                print(f"  Progress: {progress:.1f}%")
            
            start_idx = i - self.johansen_window
            window_data = prices_df.iloc[start_idx:i].values
            
            try:
                # Select lag (update occasionally)
                if self.auto_lag_selection and update_idx % 20 == 0:  # Every ~20 days
                    k_ar_diff = self.select_optimal_lag(window_data)
                elif not hasattr(self, 'k_ar_diff'):
                    k_ar_diff = 2
                else:
                    k_ar_diff = self.k_ar_diff
                
                self.k_ar_diff = k_ar_diff
                
                # Run Johansen test
                johansen_result = coint_johansen(window_data, det_order=1, k_ar_diff=k_ar_diff)
                
                # Store eigenvalues
                current_eigenvalues = johansen_result.eig
                
                # Determine rank
                selected_rank = self.determine_stable_rank(
                    johansen_result.lr1,
                    johansen_result.cvt,
                    current_eigenvalues
                )
                
                self.previous_rank = selected_rank
                
                # Extract weights if cointegrated
                if selected_rank > 0:
                    # Get primary eigenvector
                    raw_eigenvec = johansen_result.evec[:, 0]
                    primary_eigenvalue = current_eigenvalues[0]
                    
                    # Align signs
                    aligned_eigenvec = self.align_eigenvector_signs(
                        raw_eigenvec,
                        self.previous_weights
                    )
                    
                    # Normalize
                    max_idx = np.argmax(np.abs(aligned_eigenvec))
                    normalized_weights = aligned_eigenvec / aligned_eigenvec[max_idx]
                    
                    # Apply adaptive smoothing
                    smoothed_weights = self.adaptive_smooth_weights(
                        normalized_weights,
                        self.previous_weights,
                        primary_eigenvalue
                    )
                    
                    # Apply constraints
                    final_weights = self.constrain_weight_changes(
                        smoothed_weights,
                        self.previous_weights
                    )
                    
                    self.previous_weights = final_weights.copy()
                    
                    # Store results for this update point
                    current_lag = k_ar_diff
                    current_rank = selected_rank
                    current_weights = final_weights
                    
                else:
                    # No cointegration
                    if self.previous_weights is not None:
                        current_weights = self.previous_weights
                    else:
                        current_weights = np.zeros(n_assets)
                    current_rank = 0
                    current_lag = k_ar_diff
                    current_eigenvalues = np.zeros(n_assets)
                
                # Fill forward until next update point
                next_point = update_points[update_idx + 1] if update_idx < len(update_points) - 1 else n_periods
                for j in range(i, min(next_point, n_periods)):
                    weights.iloc[j] = current_weights
                    ranks.iloc[j] = current_rank
                    lag_orders.iloc[j] = current_lag
                    for k in range(min(n_assets, len(current_eigenvalues))):
                        eigenvalues.iloc[j, k] = current_eigenvalues[k]
                        
            except Exception as e:
                # Handle failures
                if self.previous_weights is not None:
                    # Fill forward with previous values
                    next_point = update_points[update_idx + 1] if update_idx < len(update_points) - 1 else n_periods
                    for j in range(i, min(next_point, n_periods)):
                        weights.iloc[j] = self.previous_weights
                        if self.previous_rank is not None:
                            ranks.iloc[j] = self.previous_rank
                        if j > 0:
                            eigenvalues.iloc[j] = eigenvalues.iloc[j-1]
                            lag_orders.iloc[j] = lag_orders.iloc[j-1]
        
        print("  Progress: 100.0% - Complete!")
        
        # Forward fill any remaining NaNs at the beginning
        weights = weights.fillna(method='ffill').fillna(method='bfill')
        ranks = ranks.fillna(method='ffill').fillna(0)
        eigenvalues = eigenvalues.fillna(method='ffill').fillna(0)
        lag_orders = lag_orders.fillna(method='ffill').fillna(2)
        
        return weights, ranks, eigenvalues, lag_orders
    
    def calculate_rolling_spread(self, prices_df, weights_df):
        """Calculate spread with proper scaling"""
        # Ensure alignment
        common_index = prices_df.index.intersection(weights_df.index)
        prices_aligned = prices_df.loc[common_index]
        weights_aligned = weights_df.loc[common_index]
        
        # Drop any rows with NaN weights
        valid_mask = ~weights_aligned.isna().any(axis=1)
        prices_final = prices_aligned[valid_mask]
        weights_final = weights_aligned[valid_mask]
        
        # Calculate spread
        spread = (prices_final * weights_final).sum(axis=1)
        
        # Scale spread to match typical range
        if self.spread_scale_factor is None:
            # Use robust scaling based on first portion of data
            first_portion = spread.iloc[:self.zscore_window]
            if len(first_portion) > 100:
                # Use median absolute deviation for robust scaling
                median_val = first_portion.median()
                mad = (first_portion - median_val).abs().median()
                if mad > 1e-6:
                    self.spread_scale_factor = 10.0 / mad  # Target scale ~10
                else:
                    self.spread_scale_factor = 1.0
            else:
                self.spread_scale_factor = 1.0
        
        spread = spread * self.spread_scale_factor
        
        return spread
    
    def calculate_rolling_zscore(self, spread, custom_zscore_window=None):
        """Calculate rolling z-score with stability enhancements"""
        zscore_window = custom_zscore_window if custom_zscore_window is not None else self.zscore_window
        
        # Use expanding window at the beginning
        rolling_mean = spread.expanding(min_periods=self.min_observations).mean()
        rolling_std = spread.expanding(min_periods=self.min_observations).std()
        
        # Switch to rolling window when we have enough data
        mask = spread.index >= spread.index[zscore_window]
        rolling_mean.loc[mask] = spread.rolling(window=zscore_window).mean().loc[mask]
        rolling_std.loc[mask] = spread.rolling(window=zscore_window).std().loc[mask]
        
        # Prevent division by zero
        rolling_std = rolling_std.clip(lower=1e-8)
        
        zscore = (spread - rolling_mean) / rolling_std
        
        # Clip extreme values
        zscore = zscore.clip(-5, 5)
        
        return zscore
    
    def analyze_portfolio(self, prices_df):
        """Main analysis method with enhanced statistics"""
        
        # Step 1: Calculate rolling Johansen
        weights, ranks, eigenvalues, lag_orders = self.calculate_rolling_johansen(prices_df)
        
        # Step 2: Calculate spread
        spread = self.calculate_rolling_spread(prices_df, weights)
        
        # Step 3: Calculate z-score
        zscore = self.calculate_rolling_zscore(spread)
        
        # Step 4: Calculate comprehensive statistics
        valid_ranks = ranks.dropna()
        valid_weights = weights.dropna()
        valid_zscore = zscore.dropna()
        valid_eigenvalues = eigenvalues['eigenvalue_0'].dropna()
        
        statistics = {
            # Rank statistics
            'rank_mean': valid_ranks.mean(),
            'rank_mode': valid_ranks.mode()[0] if len(valid_ranks.mode()) > 0 else 0,
            'rank_changes': (valid_ranks.diff().abs() > 0).sum(),
            'pct_cointegrated': (valid_ranks > 0).sum() / len(valid_ranks) * 100,
            
            # Eigenvalue statistics
            'eigenvalue_1_mean': valid_eigenvalues.mean(),
            'eigenvalue_1_std': valid_eigenvalues.std(),
            'eigenvalue_1_min': valid_eigenvalues.min(),
            'eigenvalue_1_max': valid_eigenvalues.max(),
            
            # Z-score statistics
            'zscore_mean': valid_zscore.mean(),
            'zscore_std': valid_zscore.std(),
            'zscore_max_abs': valid_zscore.abs().max(),
            'pct_within_2sigma': (valid_zscore.abs() <= 2).sum() / len(valid_zscore) * 100,
            
            # Spread statistics
            'spread_mean': spread.mean(),
            'spread_std': spread.std(),
            
            # Window information
            'johansen_window_days': self.johansen_window_days,
            'zscore_window_days': self.zscore_window_days,
            'update_frequency_days': self.update_frequency / self.bars_per_day,
            
            # Parameters used
            'smoothing_alpha': self.smoothing_alpha,
            'max_weight_change': self.max_weight_change,
            'adaptive_smoothing': self.adaptive_smoothing
        }
        
        # Weight statistics for each asset
        for col in valid_weights.columns:
            statistics[f'weight_{col}_mean'] = valid_weights[col].mean()
            statistics[f'weight_{col}_std'] = valid_weights[col].std()
            statistics[f'weight_{col}_range'] = valid_weights[col].max() - valid_weights[col].min()
        
        return {
            'weights': weights,
            'ranks': ranks,
            'eigenvalues': eigenvalues,
            'lag_orders': lag_orders,
            'spread': spread,
            'zscore': zscore,
            'statistics': statistics
        }
    
    def get_trading_signals(self, zscore, entry_threshold=2.0, exit_threshold=0.0, 
                           min_holding_period=26):
        """Generate trading signals with minimum holding period"""
        signals = pd.Series(0, index=zscore.index)
        position = 0
        bars_in_position = 0
        
        for i in range(len(zscore)):
            z = zscore.iloc[i]
            
            if pd.isna(z):
                signals.iloc[i] = position
                continue
            
            if position == 0:  # No position
                if z > entry_threshold:
                    position = -1  # Short
                    bars_in_position = 0
                elif z < -entry_threshold:
                    position = 1  # Long
                    bars_in_position = 0
                    
            else:  # In position
                bars_in_position += 1
                
                # Only allow exit after minimum holding period
                if bars_in_position >= min_holding_period:
                    if position == 1 and z >= exit_threshold:
                        position = 0  # Exit long
                    elif position == -1 and z <= -exit_threshold:
                        position = 0  # Exit short
            
            signals.iloc[i] = position
            
        return signals
    
    def _calculate_half_life(self, spread_series):
        """Calculate half-life of mean reversion using OLS on AR(1) model"""
        try:
            # Remove NaN values
            spread_clean = spread_series.dropna()
            
            if len(spread_clean) < 50:
                return np.nan
            
            # Create lagged series for AR(1) regression
            y = spread_clean.iloc[1:].values  # spread(t)
            x = spread_clean.iloc[:-1].values  # spread(t-1)
            
            # Add constant for intercept
            X = np.column_stack([np.ones(len(x)), x])
            
            # OLS regression: spread(t) = alpha + beta * spread(t-1) + epsilon
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                beta = coeffs[1]
                
                # Half-life calculation
                if beta >= 1.0 or beta <= 0.0:
                    return np.nan
                
                half_life = -np.log(2) / np.log(beta)
                
                # Sanity check: half-life should be positive and reasonable
                if half_life <= 0 or half_life > len(spread_clean) / 2:
                    return np.nan
                
                return half_life
                
            except np.linalg.LinAlgError:
                return np.nan
                
        except Exception:
            return np.nan