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
from core.data_loader import load_stock_data, resample_to_frequency, align_data
from core.johansen_analyzer import JohansenAnalyzer


class PortfolioSpreadTrainTestVisualizer:
    def __init__(self, tickers, train_year='2021', test_end_year='2024'):
        self.tickers = tickers
        self.train_year = train_year
        self.test_end_year = test_end_year
        self.resample_freq = '15min'
        self.bars_per_day = 26
        
        # Data paths
        self.data_base_path = '/Users/aridelondavidwinayu/Downloads/Project/EODHD API/us_equity_data'
        
        # Will be populated
        self.train_data = None
        self.full_data = None
        self.johansen_weights = None
        self.johansen_rank = None
        
    def find_ticker_file(self, ticker, year):
        """Search for ticker file across all sector folders"""
        year_path = os.path.join(self.data_base_path, year)
        
        if not os.path.exists(year_path):
            return None
        
        for folder in os.listdir(year_path):
            folder_path = os.path.join(year_path, folder)
            if os.path.isdir(folder_path):
                file_path = os.path.join(folder_path, f'{ticker}_US_{year}_1min_ET_regular.csv')
                if os.path.exists(file_path):
                    return file_path
        return None
    
    def load_data_for_years(self, start_year, end_year):
        """Load data for specified year range"""
        all_data = {ticker: [] for ticker in self.tickers}
        years = list(range(int(start_year), int(end_year) + 1))
        years_loaded = []
        
        for year in years:
            year_data = {}
            missing_tickers = []
            
            for ticker in self.tickers:
                file_path = self.find_ticker_file(ticker, str(year))
                if not file_path:
                    missing_tickers.append(ticker)
                    continue
                    
                try:
                    df_1min = load_stock_data(file_path)
                    df_resampled = resample_to_frequency(df_1min, self.resample_freq)
                    year_data[ticker] = df_resampled['close']
                except Exception as e:
                    missing_tickers.append(ticker)
            
            if len(year_data) == len(self.tickers):
                combined = pd.DataFrame(year_data)
                combined = combined.dropna()
                
                for ticker in self.tickers:
                    all_data[ticker].append(combined[ticker])
                
                years_loaded.append(year)
            else:
                print(f"  Skipping {year} - missing data for {missing_tickers}")
        
        # Concatenate all years
        for ticker in self.tickers:
            if all_data[ticker]:
                all_data[ticker] = pd.concat(all_data[ticker])
            else:
                return None
        
        price_data = pd.DataFrame(all_data)
        print(f"  Loaded {len(price_data)} observations from years: {years_loaded}")
        
        return price_data
    
    def run_johansen_on_train(self):
        """Run Johansen test on training data only"""
        print(f"\n1. Training Johansen model on {self.train_year} data...")
        
        # Load training data
        self.train_data = self.load_data_for_years(self.train_year, self.train_year)
        if self.train_data is None:
            raise ValueError(f"Could not load training data for {self.train_year}")
        
        print(f"  Training data: {self.train_data.index[0]} to {self.train_data.index[-1]}")
        
        # Run Johansen test
        johansen = JohansenAnalyzer()
        johansen_test = johansen.perform_johansen_test(self.train_data)
        
        self.johansen_rank = johansen_test['rank']
        
        if johansen_test['rank'] > 0:
            # Get primary cointegrating vector
            self.johansen_weights = johansen.construct_portfolio_weights(0)
            
            print(f"  Cointegration rank: {johansen_test['rank']}")
            print(f"  Eigenvalues: {[f'{e:.4f}' for e in johansen_test['eigenvalues'][:3]]}")
            print(f"  Weights: {dict(zip(self.tickers, [f'{w:.3f}' for w in self.johansen_weights]))}")
            
            # Calculate in-sample statistics
            train_spread = johansen.calculate_portfolio_spread(self.train_data, self.johansen_weights)
            train_zscore = johansen._calculate_zscore(train_spread, johansen.zscore_window)
            
            print(f"\n  Training Period Statistics:")
            print(f"    Half-life: {johansen._calculate_half_life(train_spread):.1f} bars")
            print(f"    Max |Z|: {train_zscore.abs().max():.1f}")
            print(f"    % within ±2σ: {(train_zscore.abs() <= 2).sum() / len(train_zscore) * 100:.1f}%")
            
        else:
            raise ValueError("No cointegration found in training data!")
    
    def apply_weights_to_full_period(self):
        """Apply training weights to full period data"""
        print(f"\n2. Applying {self.train_year} weights to full period ({self.train_year}-{self.test_end_year})...")
        
        # Load full period data
        self.full_data = self.load_data_for_years(self.train_year, self.test_end_year)
        if self.full_data is None:
            raise ValueError(f"Could not load full period data")
        
        print(f"  Full data: {self.full_data.index[0]} to {self.full_data.index[-1]}")
        
        # Calculate spread using training weights
        self.full_spread = (self.full_data * self.johansen_weights).sum(axis=1)
        
        # Calculate z-scores
        window = 45 * self.bars_per_day  # 45-day window
        spread_mean = self.full_spread.rolling(window=window, min_periods=window//2).mean()
        spread_std = self.full_spread.rolling(window=window, min_periods=window//2).std()
        self.full_zscore = (self.full_spread - spread_mean) / spread_std
        
        # Calculate out-of-sample statistics
        oos_mask = self.full_data.index.year > int(self.train_year)
        if oos_mask.any():
            oos_spread = self.full_spread[oos_mask]
            oos_zscore = self.full_zscore[oos_mask]
            
            print(f"\n  Out-of-Sample Statistics ({int(self.train_year)+1}-{self.test_end_year}):")
            print(f"    Max |Z|: {oos_zscore.abs().max():.1f}")
            print(f"    % within ±2σ: {(oos_zscore.abs() <= 2).sum() / len(oos_zscore) * 100:.1f}%")
    
    def plot_train_test_analysis(self):
        """Create comprehensive train/test visualization"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 2, height_ratios=[1.5, 1.5, 1.5, 1], width_ratios=[5, 1])
        
        # Title
        title = f"Portfolio Train/Test Analysis: {' + '.join(self.tickers)}\n"
        title += f"Trained on {self.train_year}, Applied to {self.train_year}-{self.test_end_year}"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Price series
        ax1 = plt.subplot(gs[0, 0])
        normalized_prices = self.full_data / self.full_data.iloc[0] * 100
        
        for ticker in self.tickers:
            ax1.plot(normalized_prices.index, normalized_prices[ticker], 
                    label=ticker, alpha=0.7, linewidth=1)
        
        # Mark training period
        train_end = pd.Timestamp(f'{self.train_year}-12-31')
        ax1.axvspan(self.full_data.index[0], train_end, alpha=0.2, color='gray', label='Training Period')
        
        ax1.set_ylabel('Normalized Price (100 = start)', fontsize=10)
        ax1.set_title('Asset Price Evolution', fontsize=12)
        ax1.legend(loc='best', ncol=len(self.tickers)+1)
        ax1.grid(True, alpha=0.3)
        
        # 2. Portfolio spread
        ax2 = plt.subplot(gs[1, 0], sharex=ax1)
        ax2.plot(self.full_spread.index, self.full_spread, 'purple', linewidth=1, alpha=0.8)
        ax2.axvspan(self.full_data.index[0], train_end, alpha=0.2, color='gray')
        ax2.axhline(y=self.full_spread.mean(), color='red', linestyle='--', alpha=0.5, label='Mean')
        ax2.set_ylabel('Spread', fontsize=10)
        ax2.set_title('Portfolio Spread (using fixed training weights)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Z-score
        ax3 = plt.subplot(gs[2, 0], sharex=ax1)
        ax3.plot(self.full_zscore.index, self.full_zscore, 'orange', linewidth=1, alpha=0.8)
        ax3.axvspan(self.full_data.index[0], train_end, alpha=0.2, color='gray')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='±2σ')
        ax3.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Z-Score', fontsize=10)
        ax3.set_xlabel('Date')
        ax3.set_title('Z-Score Evolution', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-5, 5)
        
        # 4. Weights display
        ax4 = plt.subplot(gs[0, 1])
        colors = plt.cm.RdBu(np.linspace(0, 1, len(self.tickers)))
        bars = ax4.bar(range(len(self.tickers)), self.johansen_weights, color=colors, alpha=0.7)
        ax4.set_xticks(range(len(self.tickers)))
        ax4.set_xticklabels(self.tickers, rotation=45)
        ax4.set_ylabel('Weight')
        ax4.set_title(f'Johansen Weights\n(from {self.train_year})')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0, color='black', linewidth=0.5)
        
        # Add weight values on bars
        for i, (bar, weight) in enumerate(zip(bars, self.johansen_weights)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02 * np.sign(height),
                    f'{weight:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # 5. Z-score distribution comparison
        ax5 = plt.subplot(gs[1:3, 1])
        
        # Split into train and test periods
        train_mask = self.full_data.index.year == int(self.train_year)
        test_mask = self.full_data.index.year > int(self.train_year)
        
        if train_mask.any():
            train_z = self.full_zscore[train_mask].dropna()
            ax5.hist(train_z, bins=30, alpha=0.5, label=f'Train ({self.train_year})',
                    color='blue', density=True, orientation='horizontal')
        
        if test_mask.any():
            test_z = self.full_zscore[test_mask].dropna()
            ax5.hist(test_z, bins=30, alpha=0.5, label=f'Test ({int(self.train_year)+1}-{self.test_end_year})',
                    color='red', density=True, orientation='horizontal')
        
        # Add normal distribution
        z_range = np.linspace(-5, 5, 100)
        normal_dist = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*z_range**2)
        ax5.plot(normal_dist, z_range, 'k--', label='N(0,1)', linewidth=2)
        
        ax5.set_ylim(-5, 5)
        ax5.set_xlabel('Density')
        ax5.set_title('Z-Score Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
        ax5.axhline(y=-2, color='gray', linestyle='--', alpha=0.5)
        
        # 6. Performance metrics table
        ax6 = plt.subplot(gs[3, :])
        ax6.axis('off')
        
        # Calculate metrics for different periods
        metrics_data = []
        
        # Training period
        if train_mask.any():
            train_spread = self.full_spread[train_mask]
            train_z = self.full_zscore[train_mask].dropna()
            metrics_data.append([
                f'Training ({self.train_year})',
                f'{len(train_z)}',
                f'{train_z.abs().max():.1f}',
                f'{(train_z.abs() <= 2).mean() * 100:.1f}%',
                f'{train_spread.std():.2f}'
            ])
        
        # Test period
        if test_mask.any():
            test_spread = self.full_spread[test_mask]
            test_z = self.full_zscore[test_mask].dropna()
            metrics_data.append([
                f'Test ({int(self.train_year)+1}-{self.test_end_year})',
                f'{len(test_z)}',
                f'{test_z.abs().max():.1f}',
                f'{(test_z.abs() <= 2).mean() * 100:.1f}%',
                f'{test_spread.std():.2f}'
            ])
        
        # Full period
        full_z = self.full_zscore.dropna()
        metrics_data.append([
            f'Full ({self.train_year}-{self.test_end_year})',
            f'{len(full_z)}',
            f'{full_z.abs().max():.1f}',
            f'{(full_z.abs() <= 2).mean() * 100:.1f}%',
            f'{self.full_spread.std():.2f}'
        ])
        
        # Create table
        table = ax6.table(cellText=metrics_data,
                         colLabels=['Period', 'Observations', 'Max |Z|', '% within ±2σ', 'Spread Std'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        
        # Format dates
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add year dividers
        for year in range(int(self.train_year), int(self.test_end_year) + 1):
            year_start = pd.Timestamp(f'{year}-01-01')
            if year_start >= self.full_data.index[0] and year_start <= self.full_data.index[-1]:
                for ax in [ax1, ax2, ax3]:
                    ax.axvline(x=year_start, color='gray', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_filename = f"{'_'.join(self.tickers)}_{self.train_year}_{self.test_end_year}.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as: {output_filename}")
        
        plt.show()
    
    def run(self):
        """Main execution method"""
        print(f"\nPortfolio Train/Test Analysis: {' + '.join(self.tickers)}")
        print(f"Training on {self.train_year}, Testing through {self.test_end_year}")
        print("="*60)
        
        # Train Johansen model
        self.run_johansen_on_train()
        
        # Apply to full period
        self.apply_weights_to_full_period()
        
        # Create visualization
        self.plot_train_test_analysis()


def main():
    if len(sys.argv) < 4:
        print("Usage: python visualize_portfolio_spread.py <TICKER1> <TICKER2> <TICKER3> [TICKER4...] [train_year] [test_end_year]")
        print("\nExamples:")
        print("  python visualize_portfolio_spread.py AEE EVRG PEG PPL          # Train on 2021, test through 2024")
        print("  python visualize_portfolio_spread.py JPM MET PGR TRV 2021 2023 # Train on 2021, test through 2023")
        print("  python visualize_portfolio_spread.py GOOG ORCL TXN UBER 2020 2024")
        print("\nNote: This trains Johansen on one year and applies those weights forward")
        return
    
    # Parse arguments
    tickers = []
    train_year = '2021'
    test_end_year = '2024'
    
    # Count year arguments
    year_count = 0
    
    for arg in sys.argv[1:]:
        if arg.isdigit() and len(arg) == 4:
            year_count += 1
            if year_count == 1:
                train_year = arg
            elif year_count == 2:
                test_end_year = arg
        else:
            tickers.append(arg.upper())
    
    if len(tickers) < 3:
        print("ERROR: Need at least 3 tickers for portfolio analysis")
        return
    
    print(f"\nParsed arguments:")
    print(f"  Tickers: {tickers}")
    print(f"  Train year: {train_year}")
    print(f"  Test end year: {test_end_year}")
    
    visualizer = PortfolioSpreadTrainTestVisualizer(tickers, train_year, test_end_year)
    visualizer.run()


if __name__ == "__main__":
    main()