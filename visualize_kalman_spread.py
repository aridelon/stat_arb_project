import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import load_stock_data, resample_to_frequency, align_data, get_price_series
from core.kalman_filter import KalmanFilterPairs

class KalmanSpreadVisualizer:
    def __init__(self, ticker1, ticker2, zscore_days=45, start_year='2021', end_year='2024',
                 process_variance=None, observation_variance=None):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.zscore_days = zscore_days
        self.zscore_window = zscore_days * 26
        self.start_year = start_year
        self.end_year = end_year
        self.resample_freq = '15min'
        self.process_variance = process_variance
        self.observation_variance = observation_variance
        self.data_base_path = '/Users/aridelondavidwinayu/Downloads/Project/EODHD API/us_equity_data'
        self.results_path = '/Users/aridelondavidwinayu/Downloads/Project/stat_arb_project/results'
        self.static_beta = None
        self.static_pvalue = None
        self.static_halflife = None
        self.rolling_results = None

    def find_ticker_file(self, ticker, year):
        year_path = os.path.join(self.data_base_path, year)
        for folder in os.listdir(year_path):
            folder_path = os.path.join(year_path, folder)
            if os.path.isdir(folder_path):
                file_path = os.path.join(folder_path, f'{ticker}_US_{year}_1min_ET_regular.csv')
                if os.path.exists(file_path):
                    return file_path
        return None

    def load_2021_statistics(self):
        pair_name1 = f"{self.ticker1}-{self.ticker2}"
        pair_name2 = f"{self.ticker2}-{self.ticker1}"
        for filename in os.listdir(self.results_path):
            if filename.endswith('_2021_summary.csv'):
                filepath = os.path.join(self.results_path, filename)
                df = pd.read_csv(filepath)
                if pair_name1 in df['pair'].values:
                    row = df[df['pair'] == pair_name1].iloc[0]
                    self.static_beta = row['beta']
                    self.static_pvalue = row['coint_pvalue']
                    self.static_halflife = row['halflife']
                    return True
                elif pair_name2 in df['pair'].values:
                    row = df[df['pair'] == pair_name2].iloc[0]
                    self.static_beta = 1 / row['beta']
                    self.static_pvalue = row['coint_pvalue']
                    self.static_halflife = row['halflife']
                    self.ticker1, self.ticker2 = self.ticker2, self.ticker1
                    return True
        return False

    def load_and_process_data(self):
        all_price1 = []
        all_price2 = []
        years = range(int(self.start_year), int(self.end_year) + 1)
        for year in years:
            file1 = self.find_ticker_file(self.ticker1, str(year))
            file2 = self.find_ticker_file(self.ticker2, str(year))
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
            except Exception:
                continue
        if not all_price1:
            raise ValueError("No data loaded!")
        self.price1 = pd.concat(all_price1)
        self.price2 = pd.concat(all_price2)

    def optimize_kalman_parameters(self):
        mask_2021 = self.price1.index.year == 2021
        if not mask_2021.any():
            return 0.0001, 0.001
        price1_2021 = self.price1[mask_2021]
        price2_2021 = self.price2[mask_2021]
        Q_values = np.logspace(-6, -2, 15)
        R_values = np.logspace(-4, -1, 15)
        best_score = -np.inf
        best_Q = 0.0001
        best_R = 0.001
        split_idx = int(len(price1_2021) * 0.7)
        for Q in Q_values:
            for R in R_values:
                kf = KalmanFilterPairs(
                    process_variance=Q,
                    observation_variance=R,
                    zscore_window=self.zscore_window
                )
                for i in range(split_idx):
                    kf.update(price1_2021.iloc[i], price2_2021.iloc[i])
                log_likelihood = 0
                for i in range(split_idx, len(price1_2021)):
                    beta, spread, innovation = kf.update(
                        price1_2021.iloc[i], price2_2021.iloc[i]
                    )
                    if len(kf.innovation_variances) > i - split_idx:
                        var = kf.innovation_variances[-(i-split_idx+1)]
                        if var > 0:
                            log_likelihood += -0.5 * (np.log(2*np.pi*var) + innovation**2/var)
                if log_likelihood > best_score:
                    best_score = log_likelihood
                    best_Q = Q
                    best_R = R
        return best_Q, best_R

    def run_kalman_filter(self):
        if self.process_variance is None or self.observation_variance is None:
            Q, R = self.optimize_kalman_parameters()
        else:
            Q = self.process_variance
            R = self.observation_variance
        self.kf = KalmanFilterPairs(
            process_variance=Q,
            observation_variance=R,
            zscore_window=self.zscore_window
        )
        init_bars = min(60 * 26, len(self.price1) // 10)
        if init_bars > 0:
            init_cov = np.cov(self.price1.iloc[:init_bars], self.price2.iloc[:init_bars])
            init_var = np.var(self.price2.iloc[:init_bars])
            if init_var > 0:
                self.kf.beta = init_cov[0, 1] / init_var
        results = {
            'beta': [],
            'spread': [],
            'zscore': [],
            'innovation': []
        }
        spread_buffer = []
        for i in range(len(self.price1)):
            beta, spread, innovation = self.kf.update(
                self.price1.iloc[i], self.price2.iloc[i]
            )
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
            results['innovation'].append(innovation)
        self.kalman_results = pd.DataFrame(results, index=self.price1.index)
        self.Q = Q
        self.R = R

    def calculate_static_spread(self):
        if self.static_beta is None:
            return None, None
        spread = self.price1 - self.static_beta * self.price2
        mask_2021 = spread.index.year == 2021
        if mask_2021.any():
            mean_2021 = spread[mask_2021].mean()
            std_2021 = spread[mask_2021].std()
        else:
            mean_2021 = spread.mean()
            std_2021 = spread.std()
        zscore = (spread - mean_2021) / std_2021
        return spread, zscore

    def plot_kalman_analysis(self):
        static_spread, static_zscore = self.calculate_static_spread()
        fig = plt.figure(figsize=(16, 14))
        gs = gridspec.GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 0.8], width_ratios=[3, 1])
        title_text = f'{self.ticker1}-{self.ticker2}: Kalman Filter Analysis\n'
        title_text += f'(Z-score window: {self.zscore_days}d, Q={self.Q:.6f}, R={self.R:.6f})'
        fig.suptitle(title_text, fontsize=16, fontweight='bold')
        ax1 = plt.subplot(gs[0, 0])
        ax1_twin = ax1.twinx()
        ax1.plot(self.price1.index, self.price1.values, 'b-', alpha=0.7, label=self.ticker1)
        ax1_twin.plot(self.price2.index, self.price2.values, 'r-', alpha=0.7, label=self.ticker2)
        ax1.set_ylabel(f'{self.ticker1} Price', color='b')
        ax1_twin.set_ylabel(f'{self.ticker2} Price', color='r')
        ax1.set_title('Price Series', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax2 = plt.subplot(gs[1, 0])
        ax2.plot(self.kalman_results.index, self.kalman_results['beta'], 
                'purple', linewidth=1.5, label='Kalman Beta')
        if self.static_beta:
            ax2.axhline(y=self.static_beta, color='red', linestyle='--', 
                       label=f'Static Beta ({self.static_beta:.3f})')
        ax2.set_ylabel('Beta (Hedge Ratio)', fontsize=10)
        ax2.set_title('Dynamic Beta Evolution', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        beta_mean = self.kalman_results['beta'].mean()
        beta_std = self.kalman_results['beta'].std()
        ax2.text(0.02, 0.98, f'β: {beta_mean:.3f} ± {beta_std:.3f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax3 = plt.subplot(gs[2, 0])
        ax3.plot(self.kalman_results.index, self.kalman_results['spread'], 
                'purple', linewidth=0.8, alpha=0.8, label='Kalman Spread')
        if static_spread is not None:
            ax3.plot(static_spread.index, static_spread.values, 
                    'red', linewidth=0.5, alpha=0.4, label='Static Spread')
        ax3.set_ylabel('Spread', fontsize=10)
        ax3.set_title('Spread Comparison', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(gs[3, 0])
        ax4.plot(self.kalman_results.index, self.kalman_results['zscore'], 
                'purple', linewidth=0.8, alpha=0.8, label='Kalman Z-Score')
        if static_zscore is not None:
            ax4.plot(static_zscore.index, static_zscore.values, 
                    'red', linewidth=0.5, alpha=0.4, label='Static Z-Score')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
        ax4.axhline(y=-2, color='gray', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Z-Score', fontsize=10)
        ax4.set_xlabel('Date')
        ax4.set_title('Z-Score Comparison', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        kalman_max_z = self.kalman_results['zscore'].abs().max()
        ax4.text(0.02, 0.98, f'Kalman Max |Z|: {kalman_max_z:.1f}', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        if static_zscore is not None:
            static_max_z = static_zscore.abs().max()
            ax4.text(0.02, 0.88, f'Static Max |Z|: {static_max_z:.1f}', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax5 = plt.subplot(gs[1:3, 1])
        kalman_z = self.kalman_results['zscore'].dropna()
        ax5.hist(kalman_z, bins=50, alpha=0.7, label='Kalman', 
                color='purple', density=True, orientation='horizontal')
        if static_zscore is not None:
            static_z = static_zscore.dropna()
            ax5.hist(static_z, bins=50, alpha=0.5, label='Static', 
                    color='red', density=True, orientation='horizontal')
        ax5.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
        ax5.axhline(y=-2, color='gray', linestyle='--', alpha=0.5)
        ax5.set_ylim(ax4.get_ylim())
        ax5.set_xlabel('Density')
        ax5.set_title('Z-Score Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        ax6 = plt.subplot(gs[3, 1])
        innovations = self.kalman_results['innovation'].dropna()
        ax6.hist(innovations, bins=30, alpha=0.7, color='green', density=True)
        ax6.set_xlabel('Innovation')
        ax6.set_ylabel('Density')
        ax6.set_title('Innovation Distribution')
        ax6.grid(True, alpha=0.3)
        innovation_mean = innovations.mean()
        innovation_std = innovations.std()
        ax6.text(0.05, 0.95, f'μ={innovation_mean:.2e}\nσ={innovation_std:.3f}', 
                transform=ax6.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax7 = plt.subplot(gs[4, :])
        ax7.axis('off')
        kalman_within_2sig = (np.abs(kalman_z) <= 2).mean() * 100
        summary_text = f"""
Kalman Filter Summary:
• Process variance Q: {self.Q:.6f}, Observation variance R: {self.R:.6f}
• Beta: {self.kalman_results['beta'].mean():.3f} ± {self.kalman_results['beta'].std():.3f} (range: {self.kalman_results['beta'].min():.3f} to {self.kalman_results['beta'].max():.3f})
• Z-Score: {kalman_z.mean():.3f} ± {kalman_z.std():.3f} (max |Z|: {kalman_max_z:.1f})
• {kalman_within_2sig:.1f}% of observations within ±2σ
• Innovation: mean={innovations.mean():.2e}, std={innovations.std():.3f} (should be ~0 and constant if model is good)
"""
        if static_zscore is not None:
            static_within_2sig = (np.abs(static_z) <= 2).mean() * 100
            summary_text += f"""
Static Method Comparison:
• Beta: {self.static_beta:.3f} (fixed from 2021)
• Max |Z|: {static_max_z:.1f} (deteriorates over time)
• {static_within_2sig:.1f}% of observations within ±2σ
• Improvement: Kalman reduces max |Z| by {(static_max_z - kalman_max_z)/static_max_z*100:.0f}%
"""
        ax7.text(0.1, 0.5, summary_text, transform=ax7.transAxes, 
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        for year in range(int(self.start_year), int(self.end_year) + 1):
            year_start = pd.Timestamp(f'{year}-01-01')
            if year_start >= self.price1.index[0] and year_start <= self.price1.index[-1]:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.axvline(x=year_start, color='gray', linestyle=':', alpha=0.3)
        plt.tight_layout()
        output_filename = f'{self.ticker1}-{self.ticker2}_kalman_analysis_z{self.zscore_days}d.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        results_df = pd.DataFrame({
            'price1': self.price1,
            'price2': self.price2,
            'kalman_beta': self.kalman_results['beta'],
            'kalman_spread': self.kalman_results['spread'],
            'kalman_zscore': self.kalman_results['zscore'],
            'static_spread': static_spread if static_spread is not None else np.nan,
            'static_zscore': static_zscore if static_zscore is not None else np.nan
        })
        csv_filename = f'results/{self.ticker1}-{self.ticker2}_kalman_results.csv'
        results_df.to_csv(csv_filename)
        plt.show()

    def run(self):
        self.load_2021_statistics()
        self.load_and_process_data()
        self.run_kalman_filter()
        self.plot_kalman_analysis()


def main():
    if len(sys.argv) < 3:
        return
    ticker1 = sys.argv[1].upper()
    ticker2 = sys.argv[2].upper()
    zscore_days = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    start_year = sys.argv[4] if len(sys.argv) > 4 else '2021'
    end_year = sys.argv[5] if len(sys.argv) > 5 else '2024'
    visualizer = KalmanSpreadVisualizer(ticker1, ticker2, zscore_days, start_year, end_year)
    visualizer.run()

if __name__ == "__main__":
    main()