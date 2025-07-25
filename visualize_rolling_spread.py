import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import load_stock_data, resample_to_frequency, align_data, get_price_series
from core.rolling_ols_zscore import RollingOLSAnalyzer

class RollingSpreadVisualizer:
    def __init__(self, ticker1, ticker2, ols_days=15, zscore_days=None, start_year='2021', end_year='2024'):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.ols_days = ols_days
        self.zscore_days = zscore_days if zscore_days is not None else ols_days
        
        # Convert TRADING days to 15-min bars
        # Note: ols_days and zscore_days are in TRADING days, not calendar days
        self.ols_window = ols_days * 26  # 26 bars per trading day
        self.zscore_window = self.zscore_days * 26
        
        self.start_year = start_year
        self.end_year = end_year
        self.resample_freq = '15min'
        
        # Data paths
        self.data_base_path = '/Users/aridelondavidwinayu/Downloads/Project/EODHD API/us_equity_data'
        self.results_path = '/Users/aridelondavidwinayu/Downloads/Project/stat_arb_project/results'
        
        # For comparison with static method
        self.static_beta = None
        self.static_pvalue = None
        self.static_halflife = None
        
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
    
    def load_2021_statistics(self):
        """Load the 2021 static OLS results for comparison"""
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
                    print(f"\n2021 Static Analysis Results:")
                    print(f"Beta: {self.static_beta:.4f}")
                    print(f"P-value: {self.static_pvalue:.5f}")
                    print(f"Half-life: {self.static_halflife:.1f} bars ({self.static_halflife/26:.1f} days)")
                    return True
                    
                elif pair_name2 in df['pair'].values:
                    row = df[df['pair'] == pair_name2].iloc[0]
                    self.static_beta = 1 / row['beta']
                    self.static_pvalue = row['coint_pvalue']
                    self.static_halflife = row['halflife']
                    self.ticker1, self.ticker2 = self.ticker2, self.ticker1
                    print(f"\n2021 Static Analysis Results (reversed to {self.ticker1}-{self.ticker2}):")
                    print(f"Beta: {self.static_beta:.4f}")
                    print(f"P-value: {self.static_pvalue:.5f}")
                    print(f"Half-life: {self.static_halflife:.1f} bars ({self.static_halflife/26:.1f} days)")
                    return True
        
        print(f"WARNING: Could not find static results for {pair_name1}")
        return False
    
    def load_and_process_data(self):
        """Load data for all years"""
        all_price1 = []
        all_price2 = []
        
        years = range(int(self.start_year), int(self.end_year) + 1)
        
        for year in years:
            print(f"Loading {year} data...", end=' ')
            
            file1 = self.find_ticker_file(self.ticker1, str(year))
            file2 = self.find_ticker_file(self.ticker2, str(year))
            
            if not file1 or not file2:
                print("SKIPPED - missing files")
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
                
                print(f"OK ({len(price1)} points)")
                
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        if not all_price1:
            raise ValueError("No data loaded!")
        
        self.price1 = pd.concat(all_price1)
        self.price2 = pd.concat(all_price2)
        
        print(f"\nTotal data points: {len(self.price1)}")
        print(f"Date range: {self.price1.index[0]} to {self.price1.index[-1]}")
    
    def calculate_static_spread(self):
        """Calculate spread using static 2021 beta for comparison"""
        if self.static_beta is None:
            return None, None
            
        spread = self.price1 - self.static_beta * self.price2
        
        # Use 2021 data for normalization
        mask_2021 = spread.index.year == 2021
        if mask_2021.any():
            mean_2021 = spread[mask_2021].mean()
            std_2021 = spread[mask_2021].std()
        else:
            mean_2021 = spread.mean()
            std_2021 = spread.std()
            
        zscore = (spread - mean_2021) / std_2021
        
        return spread, zscore
    
    def plot_comparison(self, rolling_results):
        """Create comprehensive comparison plot"""
        # Calculate static spread for comparison
        static_spread, static_zscore = self.calculate_static_spread()
        
        # Create figure with GridSpec for better layout control
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 0.7], width_ratios=[1, 1])
        
        # Title
        title_text = f'{self.ticker1}-{self.ticker2}: Static vs Rolling OLS Comparison\n'
        title_text += f'(OLS: {self.ols_days}d, Z-score: {self.zscore_days}d windows)'
        fig.suptitle(title_text, fontsize=16, fontweight='bold')
        
        # Plot 1: Beta evolution
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(rolling_results['betas'].index, rolling_results['betas'].values, 
                'b-', linewidth=1, label='Rolling Beta')
        if self.static_beta:
            ax1.axhline(y=self.static_beta, color='r', linestyle='--', 
                       label=f'Static Beta ({self.static_beta:.3f})')
        ax1.set_ylabel('Beta (Hedge Ratio)', fontsize=10)
        ax1.set_title('Hedge Ratio Evolution', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Static Spread
        ax2 = plt.subplot(gs[1, 0])
        if static_spread is not None:
            ax2.plot(static_spread.index, static_spread.values, 'darkred', linewidth=0.8, alpha=0.8)
            ax2.set_title('Static Spread', fontsize=12)
        ax2.set_ylabel('Spread', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rolling Spread
        ax3 = plt.subplot(gs[1, 1], sharey=ax2)
        ax3.plot(rolling_results['spread'].index, rolling_results['spread'].values, 
                'darkgreen', linewidth=0.8, alpha=0.8)
        ax3.set_title('Rolling OLS Spread', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Static Z-Score
        ax4 = plt.subplot(gs[2, 0], sharex=ax2)
        if static_zscore is not None:
            ax4.plot(static_zscore.index, static_zscore.values, 'red', linewidth=0.8, alpha=0.8)
            ax4.set_title('Static Z-Score', fontsize=12)
            # Find max absolute z-score for annotation
            max_abs_z = static_zscore.abs().max()
            ax4.text(0.02, 0.98, f'Max |Z|: {max_abs_z:.1f}', 
                    transform=ax4.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax4.set_ylabel('Z-Score', fontsize=10)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
        ax4.axhline(y=-2, color='gray', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Rolling Z-Score
        ax5 = plt.subplot(gs[2, 1], sharex=ax3, sharey=ax4)
        ax5.plot(rolling_results['zscore'].index, rolling_results['zscore'].values, 
                'green', linewidth=0.8, alpha=0.8)
        ax5.set_title('Rolling OLS Z-Score', fontsize=12)
        max_abs_z_rolling = rolling_results['zscore'].abs().max()
        ax5.text(0.02, 0.98, f'Max |Z|: {max_abs_z_rolling:.1f}', 
                transform=ax5.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax5.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
        ax5.axhline(y=-2, color='gray', linestyle='--', alpha=0.5)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Summary Statistics
        ax6 = plt.subplot(gs[3, :])
        ax6.axis('off')
        
        # Create summary text
        stats = rolling_results['statistics']
        summary_text = f"""
Rolling OLS Summary:
• OLS Window: {stats['ols_window_days']:.1f} days, Z-Score Window: {stats['zscore_window_days']:.1f} days
• Beta: {stats['beta_mean']:.3f} ± {stats['beta_std']:.3f} (range: {stats['beta_min']:.3f} to {stats['beta_max']:.3f})
• Z-Score: {stats['zscore_mean']:.3f} ± {stats['zscore_std']:.3f} (max |Z|: {stats['zscore_max_abs']:.1f})
• {stats['pct_within_2sigma']:.1f}% of observations within ±2σ

Static Method Summary:
• Beta: {self.static_beta:.3f} (fixed)
• Max |Z|: {static_zscore.abs().max():.1f} (deteriorates over time)
• {(static_zscore.abs() <= 2).sum() / len(static_zscore) * 100:.1f}% of observations within ±2σ
        """
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        # Format x-axes
        for ax in [ax2, ax3, ax4, ax5]:
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        # Add year dividers
        for year in range(int(self.start_year), int(self.end_year) + 1):
            year_start = pd.Timestamp(f'{year}-01-01')
            if year_start >= self.price1.index[0] and year_start <= self.price1.index[-1]:
                for ax in [ax1, ax2, ax3, ax4, ax5]:
                    ax.axvline(x=year_start, color='gray', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_filename = f'{self.ticker1}-{self.ticker2}_rolling_vs_static_ols{self.ols_days}d_z{self.zscore_days}d.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as: {output_filename}")
        
        plt.show()
    
    def run(self):
        """Main execution method"""
        print(f"\nRolling OLS Analysis: {self.ticker1}-{self.ticker2}")
        print(f"OLS window: {self.ols_days} trading days ({self.ols_window} bars)")
        print(f"Z-score window: {self.zscore_days} trading days ({self.zscore_window} bars)")
        print("="*60)
        
        # Sanity check for window sizes
        if self.ols_window > 10000:
            print(f"WARNING: Very large OLS window ({self.ols_window} bars). This may take a while...")
        
        # Load static results for comparison
        self.load_2021_statistics()
        
        # Load data
        self.load_and_process_data()
        
        # Check if we have enough data
        if len(self.price1) < self.ols_window:
            print(f"ERROR: Not enough data! Have {len(self.price1)} bars but need {self.ols_window} for OLS window")
            print(f"Try reducing the OLS window or loading more historical data")
            return
        
        # Run rolling OLS analysis with different windows
        rolling_analyzer = RollingOLSAnalyzer(
            ols_window=self.ols_window,
            zscore_window=self.zscore_window
        )
        rolling_results = rolling_analyzer.analyze_pair(self.price1, self.price2)
        
        # Print statistics
        print("\nRolling OLS Statistics:")
        for key, value in rolling_results['statistics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
        
        # Create visualization
        self.plot_comparison(rolling_results)

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_rolling_spread.py <TICKER1> <TICKER2> [ols_trading_days] [zscore_trading_days] [start_year] [end_year]")
        print("\nIMPORTANT: Days are in TRADING days (252 per year), not calendar days!")
        print("\nExamples:")
        print("  python visualize_rolling_spread.py TRV AFL                    # Both windows 15 trading days")
        print("  python visualize_rolling_spread.py TRV AFL 252                # 1 year OLS, same for z-score")
        print("  python visualize_rolling_spread.py TRV AFL 252 126            # 1yr OLS, 6mo z-score")
        print("  python visualize_rolling_spread.py VLO PBF 60 30              # 60d OLS, 30d z-score")
        print("\nTypical configurations:")
        print("  - Long-term stable: OLS=252 days (1yr), Z-score=126 days (6mo)")
        print("  - Medium-term: OLS=60 days (~3mo), Z-score=30 days (~6wk)")
        print("  - Short-term reactive: OLS=20 days (1mo), Z-score=10 days (2wk)")
        return
    
    ticker1 = sys.argv[1].upper()
    ticker2 = sys.argv[2].upper()
    
    # Parse arguments with smart defaults
    if len(sys.argv) > 3:
        ols_days = int(sys.argv[3])
        zscore_days = int(sys.argv[4]) if len(sys.argv) > 4 else ols_days
    else:
        ols_days = 15
        zscore_days = 15
        
    start_year = sys.argv[5] if len(sys.argv) > 5 else '2021'
    end_year = sys.argv[6] if len(sys.argv) > 6 else '2024'
    
    visualizer = RollingSpreadVisualizer(ticker1, ticker2, ols_days, zscore_days, start_year, end_year)
    visualizer.run()

if __name__ == "__main__":
    main()