import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import load_stock_data, resample_to_frequency, align_data, get_price_series

class StaticSpreadVisualizer:
    def __init__(self, ticker1, ticker2, start_year='2021', end_year='2024'):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.start_year = start_year
        self.end_year = end_year
        self.resample_freq = '15min'
        
        # Data paths
        self.data_base_path = '/Users/aridelondavidwinayu/Downloads/Project/EODHD API/us_equity_data'
        self.results_path = '/Users/aridelondavidwinayu/Downloads/Project/stat_arb_project/results/pairs_based/cointegration_test_result/2021'
        
        # Will be populated by load_2021_statistics()
        self.beta_2021 = None
        self.alpha_2021 = None
        self.spread_mean_2021 = None
        self.spread_std_2021 = None
        
    def find_ticker_file(self, ticker, year):
        """Search for ticker file across all sector folders"""
        year_path = os.path.join(self.data_base_path, year)
        
        # Search all folders in the year directory
        for folder in os.listdir(year_path):
            folder_path = os.path.join(year_path, folder)
            if os.path.isdir(folder_path):
                file_path = os.path.join(folder_path, f'{ticker}_US_{year}_1min_ET_regular.csv')
                if os.path.exists(file_path):
                    return file_path
        
        return None
    
    def load_2021_statistics(self):
        """Load the 2021 static OLS results for this pair"""
        # Try both orderings
        pair_name1 = f"{self.ticker1}-{self.ticker2}"
        pair_name2 = f"{self.ticker2}-{self.ticker1}"
        
        # Search through all sector summary files
        for filename in os.listdir(self.results_path):
            if filename.endswith('_2021_summary.csv'):
                filepath = os.path.join(self.results_path, filename)
                df = pd.read_csv(filepath)
                
                # Check for pair in either order
                if pair_name1 in df['pair'].values:
                    row = df[df['pair'] == pair_name1].iloc[0]
                    self.beta_2021 = row['beta']
                    self.alpha_2021 = row['alpha'] if 'alpha' in row else 0
                    self.coint_pvalue = row['coint_pvalue']
                    self.halflife = row['halflife']
                    print(f"Found {pair_name1} in {filename}")
                    print(f"2021 Beta: {self.beta_2021:.4f}")
                    print(f"2021 p-value: {self.coint_pvalue:.5f}")
                    print(f"2021 Half-life: {self.halflife:.1f} bars ({self.halflife/26:.1f} days)")
                    return True
                    
                elif pair_name2 in df['pair'].values:
                    row = df[df['pair'] == pair_name2].iloc[0]
                    # Reverse the relationship
                    self.beta_2021 = 1 / row['beta']
                    self.alpha_2021 = -row['alpha'] / row['beta'] if 'alpha' in row else 0
                    self.coint_pvalue = row['coint_pvalue']
                    self.halflife = row['halflife']
                    self.ticker1, self.ticker2 = self.ticker2, self.ticker1  # Swap order
                    print(f"Found {pair_name2} in {filename} (reversed to {self.ticker1}-{self.ticker2})")
                    print(f"2021 Beta: {self.beta_2021:.4f}")
                    print(f"2021 p-value: {self.coint_pvalue:.5f}")
                    print(f"2021 Half-life: {self.halflife:.1f} bars ({self.halflife/26:.1f} days)")
                    return True
        
        print(f"ERROR: Could not find pair {pair_name1} or {pair_name2} in 2021 results!")
        return False
    
    def calculate_2021_spread_stats(self):
        """Calculate spread statistics from 2021 data for normalization"""
        print("\nCalculating 2021 spread statistics for z-score normalization...")
        
        # Load 2021 data
        file1_2021 = self.find_ticker_file(self.ticker1, '2021')
        file2_2021 = self.find_ticker_file(self.ticker2, '2021')
        
        if not file1_2021 or not file2_2021:
            print("WARNING: Could not find 2021 data files for spread statistics")
            return False
        
        # Load and process 2021 data
        df1_1min = load_stock_data(file1_2021)
        df2_1min = load_stock_data(file2_2021)
        
        df1_resampled = resample_to_frequency(df1_1min, self.resample_freq)
        df2_resampled = resample_to_frequency(df2_1min, self.resample_freq)
        
        df1_aligned, df2_aligned = align_data(df1_resampled, df2_resampled)
        price1_2021, price2_2021 = get_price_series(df1_aligned, df2_aligned, 'close')
        
        # Calculate 2021 spread
        spread_2021 = price1_2021 - self.beta_2021 * price2_2021
        
        self.spread_mean_2021 = spread_2021.mean()
        self.spread_std_2021 = spread_2021.std()
        
        print(f"2021 Spread Mean: {self.spread_mean_2021:.4f}")
        print(f"2021 Spread Std: {self.spread_std_2021:.4f}")
        
        return True
    
    def load_and_process_data(self):
        """Load data for all years from start_year to end_year"""
        all_price1 = []
        all_price2 = []
        
        years = range(int(self.start_year), int(self.end_year) + 1)
        
        for year in years:
            print(f"\nLoading {year} data...")
            
            # Find files
            file1 = self.find_ticker_file(self.ticker1, str(year))
            file2 = self.find_ticker_file(self.ticker2, str(year))
            
            if not file1 or not file2:
                print(f"Skipping {year} - missing data files")
                continue
            
            # Load and process
            try:
                df1_1min = load_stock_data(file1)
                df2_1min = load_stock_data(file2)
                
                df1_resampled = resample_to_frequency(df1_1min, self.resample_freq)
                df2_resampled = resample_to_frequency(df2_1min, self.resample_freq)
                
                df1_aligned, df2_aligned = align_data(df1_resampled, df2_resampled)
                price1, price2 = get_price_series(df1_aligned, df2_aligned, 'close')
                
                all_price1.append(price1)
                all_price2.append(price2)
                
                print(f"Loaded {len(price1)} data points for {year}")
                
            except Exception as e:
                print(f"Error loading {year}: {e}")
                continue
        
        if not all_price1:
            raise ValueError("No data loaded!")
        
        # Concatenate all years
        self.price1 = pd.concat(all_price1)
        self.price2 = pd.concat(all_price2)
        
        # Calculate spread using 2021 static beta
        self.spread = self.price1 - self.beta_2021 * self.price2
        
        # Calculate z-score using 2021 statistics
        if self.spread_mean_2021 is not None and self.spread_std_2021 is not None:
            self.zscore = (self.spread - self.spread_mean_2021) / self.spread_std_2021
        else:
            # Fallback to full period statistics
            self.zscore = (self.spread - self.spread.mean()) / self.spread.std()
        
        print(f"\nTotal data points: {len(self.price1)}")
        print(f"Date range: {self.price1.index[0]} to {self.price1.index[-1]}")
    
    def save_results_to_csv(self):
        """Save all calculated data to CSV for easy backtesting"""
        # Create results dataframe
        results_df = pd.DataFrame({
            'datetime': self.price1.index,
            'price1': self.price1.values,
            'price2': self.price2.values,
            'static_beta': self.beta_2021,  # Constant value
            'spread': self.spread.values,
            'zscore': self.zscore.values,
            'spread_mean_2021': self.spread_mean_2021,
            'spread_std_2021': self.spread_std_2021
        })
        
        # Add metadata columns
        results_df['ticker1'] = self.ticker1
        results_df['ticker2'] = self.ticker2
        results_df['method'] = 'static_ols'
        results_df['beta_calculation'] = '2021_fixed'
        results_df['coint_pvalue_2021'] = self.coint_pvalue
        results_df['halflife_2021_bars'] = self.halflife
        
        # Save to CSV
        csv_filename = f'results/{self.ticker1}-{self.ticker2}_static_spread_{self.start_year}-{self.end_year}.csv'
        os.makedirs('results', exist_ok=True)
        results_df.to_csv(csv_filename, index=False)
        print(f"\nData saved to: {csv_filename}")
        
        # Also save a summary file
        summary_df = pd.DataFrame({
            'ticker1': [self.ticker1],
            'ticker2': [self.ticker2],
            'beta_2021': [self.beta_2021],
            'spread_mean_2021': [self.spread_mean_2021],
            'spread_std_2021': [self.spread_std_2021],
            'coint_pvalue_2021': [self.coint_pvalue],
            'halflife_bars': [self.halflife],
            'halflife_days': [self.halflife / 26],
            'zscore_max_abs': [self.zscore.abs().max()],
            'zscore_pct_within_2sigma': [(self.zscore.abs() <= 2).mean() * 100],
            'data_start': [self.price1.index[0]],
            'data_end': [self.price1.index[-1]],
            'num_observations': [len(self.price1)]
        })
        
        summary_filename = f'results/{self.ticker1}-{self.ticker2}_static_summary.csv'
        summary_df.to_csv(summary_filename, index=False)
        print(f"Summary saved to: {summary_filename}")
        
        return results_df
    
    def plot_spread_and_zscore(self):
        """Create visualization of spread and z-score"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot 1: Spread
        ax1.plot(self.spread.index, self.spread.values, color='blue', linewidth=0.8)
        ax1.set_title(f'Spread of {self.ticker1}-{self.ticker2} ({self.resample_freq}) using 2021 beta={self.beta_2021:.3f}', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Spread', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.spread.mean(), color='red', linestyle='--', alpha=0.5, label='Mean')
        
        # Add year dividers
        for year in range(int(self.start_year), int(self.end_year) + 1):
            year_start = pd.Timestamp(f'{year}-01-01')
            if year_start >= self.spread.index[0] and year_start <= self.spread.index[-1]:
                ax1.axvline(x=year_start, color='gray', linestyle=':', alpha=0.5)
                ax2.axvline(x=year_start, color='gray', linestyle=':', alpha=0.5)
        
        # Plot 2: Z-Score
        ax2.plot(self.zscore.index, self.zscore.values, color='orange', linewidth=0.8)
        ax2.set_title('Normalized Z-Score of Spread', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Z-Score', fontsize=12)
        ax2.set_xlabel('Datetime', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add z-score thresholds
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='±2σ')
        ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=1, color='orange', linestyle='--', alpha=0.3, label='±1σ')
        ax2.axhline(y=-1, color='orange', linestyle='--', alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Add legends
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_filename = f'{self.ticker1}-{self.ticker2}_static_spread_{self.start_year}-{self.end_year}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as: {output_filename}")
        
        plt.show()
    
    def print_summary_statistics(self):
        """Print summary statistics of the spread"""
        print("\n" + "="*60)
        print("SPREAD SUMMARY STATISTICS")
        print("="*60)
        
        # Overall statistics
        print(f"\nOverall ({self.start_year}-{self.end_year}):")
        print(f"  Spread - Mean: {self.spread.mean():.4f}, Std: {self.spread.std():.4f}")
        print(f"  Z-Score - Mean: {self.zscore.mean():.4f}, Std: {self.zscore.std():.4f}")
        print(f"  Z-Score range: [{self.zscore.min():.2f}, {self.zscore.max():.2f}]")
        print(f"  % within ±2σ: {(self.zscore.abs() <= 2).mean() * 100:.1f}%")
        
        # By year statistics
        for year in range(int(self.start_year), int(self.end_year) + 1):
            year_mask = self.spread.index.year == year
            if year_mask.any():
                year_spread = self.spread[year_mask]
                year_zscore = self.zscore[year_mask]
                
                print(f"\n{year}:")
                print(f"  Spread - Mean: {year_spread.mean():.4f}, Std: {year_spread.std():.4f}")
                print(f"  Z-Score - Mean: {year_zscore.mean():.4f}, Std: {year_zscore.std():.4f}")
                print(f"  Z-Score range: [{year_zscore.min():.2f}, {year_zscore.max():.2f}]")
                print(f"  % within ±2σ: {(year_zscore.abs() <= 2).mean() * 100:.1f}%")
    
    def run(self):
        """Main execution method"""
        print(f"Static Spread Analysis: {self.ticker1}-{self.ticker2}")
        print("="*60)
        
        # Load 2021 statistics
        if not self.load_2021_statistics():
            return
        
        # Calculate 2021 spread statistics
        self.calculate_2021_spread_stats()
        
        # Load and process all data
        self.load_and_process_data()
        
        # Save results to CSV
        self.save_results_to_csv()
        
        # Create visualizations
        self.plot_spread_and_zscore()
        
        # Print statistics
        self.print_summary_statistics()

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_static_spread.py <TICKER1> <TICKER2> [start_year] [end_year]")
        print("Example: python visualize_static_spread.py TRV AFL 2021 2024")
        print("\nNEW: Now saves spread and z-score data to CSV for backtesting!")
        print("\nRecommended pairs from analysis:")
        print("  - TRV AFL   (insurance)")
        print("  - MET TRV   (insurance)")
        print("  - AEP WEC   (utilities)")
        print("  - CLF SCCO  (metals)")
        print("  - VLO PBF   (refiners)")
        return
    
    ticker1 = sys.argv[1].upper()
    ticker2 = sys.argv[2].upper()
    start_year = sys.argv[3] if len(sys.argv) > 3 else '2021'
    end_year = sys.argv[4] if len(sys.argv) > 4 else '2024'
    
    visualizer = StaticSpreadVisualizer(ticker1, ticker2, start_year, end_year)
    visualizer.run()

if __name__ == "__main__":
    main()