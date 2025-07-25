import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import load_stock_data, resample_to_frequency, align_data, get_price_series
from core.rolling_ols_zscore import RollingOLSAnalyzer
from core.kalman_filter import KalmanFilterPairs


class KalmanRollingComparison:
    def __init__(self, start_year='2021', end_year='2024'):
        self.start_year = start_year
        self.end_year = end_year
        self.resample_freq = '15min'
        
        # Standard windows from your optimal configuration
        self.ols_days = 90
        self.zscore_days = 45
        self.ols_window = self.ols_days * 26  # 26 bars per trading day
        self.zscore_window = self.zscore_days * 26
        
        # Data paths
        self.data_base_path = '/Users/aridelondavidwinayu/Downloads/Project/EODHD API/us_equity_data'
        self.results_path = '/Users/aridelondavidwinayu/Downloads/Project/stat_arb_project/results'
        
        # Portfolio pairs
        self.pairs = [
            ('MET', 'TRV', 'Insurance'),
            ('AEP', 'WEC', 'Utilities'),
            ('CLF', 'SCCO', 'Metals'),
            ('VLO', 'PBF', 'Energy'),
            ('MAA', 'CPT', 'REITs')
        ]
        
    def find_ticker_file(self, ticker, year):
        """Search for ticker file across all sector folders"""
        year_path = os.path.join(self.data_base_path, year)
        
        for folder in os.listdir(year_path):
            folder_path = os.path.join(year_path, folder)
            if os.path.isdir(folder_path):
                file_path = os.path.join(folder_path, f'{ticker}_US_{year}_1min_ET_regular.csv')
                if os.path.exists(file_path):
                    return file_path
        return None
    
    def load_pair_data(self, ticker1, ticker2):
        """Load and process data for a pair"""
        all_price1 = []
        all_price2 = []
        
        years = range(int(self.start_year), int(self.end_year) + 1)
        
        for year in years:
            file1 = self.find_ticker_file(ticker1, str(year))
            file2 = self.find_ticker_file(ticker2, str(year))
            
            if not file1 or not file2:
                continue
            
            try:
                df1_1min = load_stock_data(file1)
                df2_1min = load_stock_data(file2)
                
                df1_resampled = resample_to_frequency(df1_1min, self.resample_freq)
                df2_resampled = resample_to_frequency(df2_1min, self.resample_freq)
                
                df1_aligned, df2_aligned = align_data(df1_resampled, df2_resampled)
                price1, price2 = get_price_series(df1_aligned, df2_aligned, 'close')
                
                all_price1.append(price1)
                all_price2.append(price2)
                
            except Exception as e:
                continue
        
        if not all_price1:
            return None, None
        
        return pd.concat(all_price1), pd.concat(all_price2)
    
    def run_rolling_ols(self, price1, price2):
        """Run rolling OLS analysis"""
        analyzer = RollingOLSAnalyzer(
            ols_window=self.ols_window,
            zscore_window=self.zscore_window
        )
        return analyzer.analyze_pair(price1, price2)
    
    def run_kalman_filter(self, price1, price2, optimize=True):
        """Run Kalman filter analysis"""
        # Initialize Kalman filter
        kf = KalmanFilterPairs(zscore_window=self.zscore_window)
        
        # Optimize parameters on 2021 data if requested
        if optimize:
            mask_2021 = price1.index.year == 2021
            if mask_2021.any():
                price1_2021 = price1[mask_2021]
                price2_2021 = price2[mask_2021]
                
                # Simple grid search
                Q_values = np.logspace(-6, -2, 10)
                R_values = np.logspace(-4, -1, 10)
                
                best_score = -np.inf
                best_Q = 0.0001
                best_R = 0.001
                
                split_idx = int(len(price1_2021) * 0.7)
                
                for Q in Q_values:
                    for R in R_values:
                        kf_test = KalmanFilterPairs(
                            process_variance=Q,
                            observation_variance=R,
                            zscore_window=self.zscore_window
                        )
                        
                        # Train
                        for i in range(split_idx):
                            kf_test.update(price1_2021.iloc[i], price2_2021.iloc[i])
                        
                        # Validate
                        log_likelihood = 0
                        for i in range(split_idx, len(price1_2021)):
                            _, _, innovation = kf_test.update(
                                price1_2021.iloc[i], price2_2021.iloc[i]
                            )
                            if kf_test.innovation_variances:
                                var = kf_test.innovation_variances[-1]
                                if var > 0:
                                    log_likelihood += -0.5 * (np.log(2*np.pi*var) + innovation**2/var)
                        
                        if log_likelihood > best_score:
                            best_score = log_likelihood
                            best_Q = Q
                            best_R = R
                
                kf.Q = best_Q
                kf.R = best_R
        
        # Initialize beta with simple OLS
        init_bars = min(60 * 26, len(price1) // 10)
        if init_bars > 0:
            init_cov = np.cov(price1.iloc[:init_bars], price2.iloc[:init_bars])
            init_var = np.var(price2.iloc[:init_bars])
            if init_var > 0:
                kf.beta = init_cov[0, 1] / init_var
        
        # Run filter
        results = {
            'beta': [],
            'spread': [],
            'zscore': []
        }
        
        spread_buffer = []
        
        for i in range(len(price1)):
            beta, spread, _ = kf.update(price1.iloc[i], price2.iloc[i])
            
            spread_buffer.append(spread)
            if len(spread_buffer) > self.zscore_window:
                spread_buffer.pop(0)
            
            if len(spread_buffer) >= 20:
                spread_mean = np.mean(spread_buffer)
                spread_std = np.std(spread_buffer)
                zscore = (spread - spread_mean) / spread_std if spread_std > 0 else 0
            else:
                zscore = 0
            
            results['beta'].append(beta)
            results['spread'].append(spread)
            results['zscore'].append(zscore)
        
        # Create results dictionary matching rolling OLS format
        kalman_results = {
            'betas': pd.Series(results['beta'], index=price1.index),
            'spread': pd.Series(results['spread'], index=price1.index),
            'zscore': pd.Series(results['zscore'], index=price1.index),
            'Q': kf.Q,
            'R': kf.R
        }
        
        return kalman_results
    
    def compare_single_pair(self, ticker1, ticker2, sector, optimize_kalman=True):
        """Compare methods for a single pair"""
        print(f"\nProcessing {ticker1}-{ticker2} ({sector})...")
        
        # Load data
        price1, price2 = self.load_pair_data(ticker1, ticker2)
        if price1 is None:
            print(f"No data available for {ticker1}-{ticker2}")
            return None
        
        # Run both methods
        rolling_results = self.run_rolling_ols(price1, price2)
        kalman_results = self.run_kalman_filter(price1, price2, optimize=optimize_kalman)
        
        # Create comparison plot
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(4, 2, height_ratios=[1.5, 1, 1, 1], width_ratios=[3, 1])
        
        # 1. Price series
        ax1 = fig.add_subplot(gs[0, 0])
        ax1_twin = ax1.twinx()
        ax1.plot(price1.index, price1.values, 'b-', alpha=0.7, label=ticker1)
        ax1_twin.plot(price2.index, price2.values, 'r-', alpha=0.7, label=ticker2)
        ax1.set_ylabel(f'{ticker1} Price', color='b')
        ax1_twin.set_ylabel(f'{ticker2} Price', color='r')
        ax1.set_title(f'{ticker1}-{ticker2} Pair: Kalman Filter vs Rolling OLS Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. Beta comparison
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(rolling_results['betas'].index, rolling_results['betas'].values, 
                 'g-', alpha=0.8, label=f'Rolling OLS ({self.ols_days}d)', linewidth=1.5)
        ax2.plot(kalman_results['betas'].index, kalman_results['betas'].values, 
                 'purple', alpha=0.8, label='Kalman Filter', linewidth=1.5)
        ax2.set_ylabel('Hedge Ratio (β)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. Spread comparison
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(rolling_results['spread'].index, rolling_results['spread'].values, 
                 'g-', alpha=0.6, label='Rolling OLS')
        ax3.plot(kalman_results['spread'].index, kalman_results['spread'].values, 
                 'purple', alpha=0.6, label='Kalman Filter')
        ax3.set_ylabel('Spread')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. Z-score comparison
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.plot(rolling_results['zscore'].index, rolling_results['zscore'].values, 
                 'g-', alpha=0.6, label='Rolling OLS')
        ax4.plot(kalman_results['zscore'].index, kalman_results['zscore'].values, 
                 'purple', alpha=0.6, label='Kalman Filter')
        ax4.axhline(y=2, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_ylabel('Z-Score')
        ax4.set_xlabel('Date')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        # 5. Z-score distribution comparison
        ax5 = fig.add_subplot(gs[:2, 1])
        rolling_z = rolling_results['zscore'].dropna()
        kalman_z = kalman_results['zscore'].dropna()
        
        ax5.hist(rolling_z, bins=50, alpha=0.5, label='Rolling OLS', 
                 color='green', density=True, orientation='horizontal')
        ax5.hist(kalman_z, bins=50, alpha=0.5, label='Kalman', 
                 color='purple', density=True, orientation='horizontal')
        ax5.axhline(y=2, color='red', linestyle='--', alpha=0.5)
        ax5.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        ax5.set_ylim(ax4.get_ylim())
        ax5.set_xlabel('Density')
        ax5.set_title('Z-Score Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance metrics
        ax6 = fig.add_subplot(gs[2:, 1])
        ax6.axis('off')
        
        # Calculate metrics
        rolling_within_2sig = (np.abs(rolling_z) <= 2).mean() * 100
        kalman_within_2sig = (np.abs(kalman_z) <= 2).mean() * 100
        rolling_max_z = np.max(np.abs(rolling_z))
        kalman_max_z = np.max(np.abs(kalman_z))
        
        metrics_text = f"""Performance Metrics:

Rolling OLS ({self.ols_days}/{self.zscore_days}d):
  Within ±2σ: {rolling_within_2sig:.1f}%
  Max |Z|: {rolling_max_z:.2f}σ
  
Kalman Filter:
  Within ±2σ: {kalman_within_2sig:.1f}%
  Max |Z|: {kalman_max_z:.2f}σ
  
Kalman Parameters:
  Q: {kalman_results['Q']:.6f}
  R: {kalman_results['R']:.6f}
  
Beta Statistics:
  Rolling β std: {rolling_results['betas'].std():.4f}
  Kalman β std: {kalman_results['betas'].std():.4f}
"""
        
        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
                 verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        # Format dates
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add year dividers
        for year in range(int(self.start_year), int(self.end_year) + 1):
            year_start = pd.Timestamp(f'{year}-01-01')
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axvline(x=year_start, color='gray', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_filename = f'{self.results_path}/kalman_vs_rolling_{ticker1}_{ticker2}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_filename}")
        plt.close(fig)
        
        return {
            'rolling_within_2sig': rolling_within_2sig,
            'kalman_within_2sig': kalman_within_2sig,
            'rolling_max_z': rolling_max_z,
            'kalman_max_z': kalman_max_z
        }
    
    def compare_all_pairs(self, optimize_kalman=True):
        """Compare all portfolio pairs"""
        comparison_data = []
        
        for ticker1, ticker2, sector in self.pairs:
            metrics = self.compare_single_pair(ticker1, ticker2, sector, optimize_kalman)
            
            if metrics:
                comparison_data.append({
                    'Pair': f'{ticker1}-{ticker2}',
                    'Sector': sector,
                    'Rolling_Within_2σ': metrics['rolling_within_2sig'],
                    'Kalman_Within_2σ': metrics['kalman_within_2sig'],
                    'Rolling_Max_Z': metrics['rolling_max_z'],
                    'Kalman_Max_Z': metrics['kalman_max_z']
                })
        
        # Create summary plot
        df_comparison = pd.DataFrame(comparison_data)
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 1, height_ratios=[2, 2, 1])
        
        # 1. Max |Z| comparison
        ax1 = fig.add_subplot(gs[0])
        x = np.arange(len(df_comparison))
        width = 0.35
        
        ax1.bar(x - width/2, df_comparison['Rolling_Max_Z'], width, 
                label='Rolling OLS', color='green', alpha=0.7)
        ax1.bar(x + width/2, df_comparison['Kalman_Max_Z'], width, 
                label='Kalman Filter', color='purple', alpha=0.7)
        
        ax1.set_ylabel('Maximum |Z-Score|')
        ax1.set_title('Maximum Absolute Z-Score Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{row['Pair']}\n({row['Sector']})" 
                             for _, row in df_comparison.iterrows()])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5σ threshold')
        
        # 2. Within ±2σ comparison
        ax2 = fig.add_subplot(gs[1])
        ax2.bar(x - width/2, df_comparison['Rolling_Within_2σ'], width, 
                label='Rolling OLS', color='green', alpha=0.7)
        ax2.bar(x + width/2, df_comparison['Kalman_Within_2σ'], width, 
                label='Kalman Filter', color='purple', alpha=0.7)
        
        ax2.set_ylabel('% Within ±2σ')
        ax2.set_title('Percentage of Observations Within ±2σ')
        ax2.set_xticks(x)
        ax2.set_xticklabels([row['Pair'] for _, row in df_comparison.iterrows()])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% target')
        
        # 3. Summary statistics
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')
        
        # Calculate average improvements
        avg_rolling_within = df_comparison['Rolling_Within_2σ'].mean()
        avg_kalman_within = df_comparison['Kalman_Within_2σ'].mean()
        avg_rolling_max = df_comparison['Rolling_Max_Z'].mean()
        avg_kalman_max = df_comparison['Kalman_Max_Z'].mean()
        
        summary_text = f"""Portfolio Summary:
Average Within ±2σ: Rolling OLS = {avg_rolling_within:.1f}%, Kalman = {avg_kalman_within:.1f}%
Average Max |Z|: Rolling OLS = {avg_rolling_max:.1f}σ, Kalman = {avg_kalman_max:.1f}σ
Average Improvement: {(avg_rolling_max - avg_kalman_max)/avg_rolling_max*100:.0f}% reduction in max |Z|"""
        
        ax3.text(0.5, 0.5, summary_text, transform=ax3.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        plt.suptitle('Kalman Filter vs Rolling OLS: Portfolio Comparison', fontsize=14)
        plt.tight_layout()
        
        # Save
        output_filename = f'{self.results_path}/kalman_vs_rolling_portfolio_comparison.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        df_comparison.to_csv(f'{self.results_path}/kalman_vs_rolling_metrics.csv', index=False)
        
        print(f"\nSaved portfolio comparison: {output_filename}")
        print("\nResults summary:")
        print(df_comparison.to_string(index=False))
        
        return df_comparison


def main():
    if len(sys.argv) > 1:
        optimize_kalman = sys.argv[1].lower() in ['true', '1', 'yes']
    else:
        optimize_kalman = True
    
    print("Kalman Filter vs Rolling OLS Comparison")
    print("="*60)
    print(f"Optimize Kalman parameters: {optimize_kalman}")
    
    comparison = KalmanRollingComparison()
    df_results = comparison.compare_all_pairs(optimize_kalman)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()