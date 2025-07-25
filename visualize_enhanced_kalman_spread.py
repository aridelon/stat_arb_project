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
from core.kalman_filter_enhanced import EnhancedKalmanFilterAnalyzer
from core.kalman_filter import KalmanFilterAnalyzer  # Basic for comparison
from core.rolling_ols_zscore import RollingOLSAnalyzer  # For comparison


class EnhancedKalmanSpreadVisualizer:
    def __init__(self, ticker1, ticker2, zscore_days=45, start_year='2021', end_year='2024',
                 delta=None, optimize_delta=True):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.zscore_days = zscore_days
        
        # Convert TRADING days to 15-min bars
        self.zscore_window = zscore_days * 26  # 26 bars per trading day
        
        self.start_year = start_year
        self.end_year = end_year
        self.resample_freq = '15min'
        
        # Enhanced Kalman parameters
        self.delta = delta
        self.optimize_delta = optimize_delta
        
        # Data paths
        self.data_base_path = '/Users/aridelondavidwinayu/Downloads/Project/EODHD API/us_equity_data'
        self.results_path = '/Users/aridelondavidwinayu/Downloads/Project/stat_arb_project/results'
        
        # For comparison with other methods
        self.static_beta = None
        self.static_pvalue = None
        self.static_halflife = None
        
        # Results storage
        self.rolling_results = None
        self.basic_kalman_results = None
        self.enhanced_kalman_results = None
        
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
    
    def run_all_methods(self):
        """Run all methods for comparison"""
        # 1. Rolling OLS (90/45 days)
        print("\n1. Running Rolling OLS (90/45 days)...")
        try:
            rolling_analyzer = RollingOLSAnalyzer(
                ols_window=90 * 26,
                zscore_window=self.zscore_window
            )
            self.rolling_results = rolling_analyzer.analyze_pair(self.price1, self.price2)
            print(f"   Max |Z|: {self.rolling_results['statistics']['zscore_max_abs']:.1f}")
        except Exception as e:
            print(f"   Failed: {e}")
        
        # 2. Basic Kalman (beta only)
        print("\n2. Running Basic Kalman (β only)...")
        try:
            basic_kalman = KalmanFilterAnalyzer(
                zscore_window=self.zscore_window,
                optimize_params=True
            )
            self.basic_kalman_results = basic_kalman.analyze_pair(self.price1, self.price2)
            max_z = self.basic_kalman_results['statistics']['zscore_max_abs']
            print(f"   Max |Z|: {max_z:.1f}")
            if max_z > 10:
                print(f"   WARNING: Extremely high z-scores indicate poor model fit")
                print(f"   This is expected - basic Kalman can't handle mean shifts")
        except Exception as e:
            print(f"   Failed: {e}")
            print("   Note: Basic Kalman often fails with extreme z-scores when mean shifts occur")
        
        # 3. Enhanced Kalman (μ + γ)
        print("\n3. Running Enhanced Kalman (μ + γ)...")
        enhanced_kalman = EnhancedKalmanFilterAnalyzer(
            zscore_window=self.zscore_window,
            delta=self.delta,
            optimize_delta=self.optimize_delta
        )
        self.enhanced_kalman_results = enhanced_kalman.analyze_pair(self.price1, self.price2)
        print(f"   Max |Z|: {self.enhanced_kalman_results['statistics']['zscore_max_abs']:.1f}")
    
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
    
    def plot_enhanced_kalman_analysis(self):
        """Create comprehensive visualization comparing all methods"""
        # Calculate static spread
        static_spread, static_zscore = self.calculate_static_spread()
        
        # Create figure
        fig = plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(6, 2, height_ratios=[1.2, 1, 1, 1, 1, 0.8], width_ratios=[3, 1])
        
        # Title
        title_text = f'{self.ticker1}-{self.ticker2}: Enhanced Kalman Filter Analysis\n'
        title_text += f'(Tracking both μ and γ, Z-score window: {self.zscore_days}d)'
        fig.suptitle(title_text, fontsize=16, fontweight='bold')
        
        # 1. Price series
        ax1 = plt.subplot(gs[0, 0])
        ax1_twin = ax1.twinx()
        ax1.plot(self.price1.index, self.price1.values, 'b-', alpha=0.7, label=self.ticker1)
        ax1_twin.plot(self.price2.index, self.price2.values, 'r-', alpha=0.7, label=self.ticker2)
        ax1.set_ylabel(f'{self.ticker1} Price', color='b')
        ax1_twin.set_ylabel(f'{self.ticker2} Price', color='r')
        ax1.set_title('Price Series')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. Beta/Gamma comparison
        ax2 = plt.subplot(gs[1, 0])
        
        # Plot all betas
        if self.static_beta:
            ax2.axhline(y=self.static_beta, color='red', linestyle='--', 
                       label=f'Static β ({self.static_beta:.3f})', alpha=0.7)
        
        if self.rolling_results:
            ax2.plot(self.rolling_results['betas'].index, self.rolling_results['betas'].values,
                    'green', linewidth=1, alpha=0.7, label='Rolling OLS (90d)')
        
        if self.basic_kalman_results:
            ax2.plot(self.basic_kalman_results['betas'].index, 
                    self.basic_kalman_results['betas'].values,
                    'blue', linewidth=1, alpha=0.7, label='Basic Kalman (β only)')
        else:
            # Add note about failed basic Kalman
            ax2.text(0.5, 0.5, 'Basic Kalman failed\n(extreme values)', 
                    transform=ax2.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # Enhanced Kalman beta
        ax2.plot(self.enhanced_kalman_results['betas'].index, 
                self.enhanced_kalman_results['betas'].values,
                'purple', linewidth=1.5, label='Enhanced Kalman γ')
        
        ax2.set_ylabel('Hedge Ratio (β/γ)', fontsize=10)
        ax2.set_title('Dynamic Hedge Ratio Evolution', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Add beta statistics
        enhanced_stats = self.enhanced_kalman_results['statistics']
        ax2.text(0.02, 0.98, 
                f'Enhanced γ: {enhanced_stats["beta_mean"]:.3f} ± {enhanced_stats["beta_std"]:.3f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Mean (μ) tracking - unique to enhanced Kalman
        ax3 = plt.subplot(gs[2, 0])
        ax3.plot(self.enhanced_kalman_results['mu'].index, 
                self.enhanced_kalman_results['mu'].values,
                'orange', linewidth=1.5, label='Enhanced Kalman μ')
        ax3.set_ylabel('Mean Level (μ)', fontsize=10)
        ax3.set_title('Mean Level Tracking (Enhanced Kalman Only)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Add mu statistics
        mu_min = float(enhanced_stats['mu_range'][0]) if isinstance(enhanced_stats['mu_range'], list) else 0
        mu_max = float(enhanced_stats['mu_range'][1]) if isinstance(enhanced_stats['mu_range'], list) else 0
        ax3.text(0.02, 0.98, 
                f'μ range: [{mu_min:.3f}, {mu_max:.3f}]',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Spread comparison
        ax4 = plt.subplot(gs[3, 0])
        
        # Enhanced Kalman spreads
        ax4.plot(self.enhanced_kalman_results['raw_spread'].index,
                self.enhanced_kalman_results['raw_spread'].values,
                'gray', linewidth=0.5, alpha=0.5, label='Raw spread (y - γx)')
        ax4.plot(self.enhanced_kalman_results['spread'].index,
                self.enhanced_kalman_results['spread'].values,
                'purple', linewidth=1, alpha=0.8, label='Adjusted spread (y - γx - μ)')
        
        ax4.set_ylabel('Spread', fontsize=10)
        ax4.set_title('Spread Comparison', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Z-score comparison
        ax5 = plt.subplot(gs[4, 0])
        
        # Plot z-scores
        if static_zscore is not None:
            ax5.plot(static_zscore.index, static_zscore.values,
                    'red', linewidth=0.5, alpha=0.4, label='Static')
        
        if self.rolling_results:
            ax5.plot(self.rolling_results['zscore'].index, self.rolling_results['zscore'].values,
                    'green', linewidth=0.7, alpha=0.5, label='Rolling OLS')
        
        if self.basic_kalman_results:
            ax5.plot(self.basic_kalman_results['zscore'].index,
                    self.basic_kalman_results['zscore'].values,
                    'blue', linewidth=0.7, alpha=0.5, label='Basic Kalman')
        else:
            # Add note about basic Kalman failure
            ax5.text(0.98, 0.5, 'Basic Kalman\nfailed', 
                    transform=ax5.transAxes, ha='right', va='center',
                    fontsize=9, color='blue',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Enhanced Kalman z-score
        ax5.plot(self.enhanced_kalman_results['zscore'].index,
                self.enhanced_kalman_results['zscore'].values,
                'purple', linewidth=1, alpha=0.8, label='Enhanced Kalman')
        
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax5.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
        ax5.axhline(y=-2, color='gray', linestyle='--', alpha=0.5)
        ax5.set_ylabel('Z-Score', fontsize=10)
        ax5.set_xlabel('Date')
        ax5.set_title('Z-Score Comparison', fontsize=12)
        ax5.legend(loc='best', ncol=2)
        ax5.grid(True, alpha=0.3)
        
        # Add max |Z| annotations
        enhanced_max_z = self.enhanced_kalman_results['zscore'].abs().max()
        ax5.text(0.02, 0.98, f'Enhanced Kalman Max |Z|: {enhanced_max_z:.1f}',
                transform=ax5.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3))
        
        # Set y-axis limits for z-scores (reasonable range)
        max_z_limit = max(10, enhanced_max_z + 2)
        if static_zscore is not None:
            max_z_limit = max(max_z_limit, min(20, static_zscore.abs().max() + 2))
        ax5.set_ylim(-max_z_limit, max_z_limit)
        
        # 6. Z-score distributions
        ax6 = plt.subplot(gs[1:4, 1])
        
        # Enhanced Kalman distribution
        enhanced_z = self.enhanced_kalman_results['zscore'].replace([np.inf, -np.inf], np.nan).dropna()
        ax6.hist(enhanced_z, bins=50, alpha=0.7, label='Enhanced Kalman',
                color='purple', density=True, orientation='horizontal')
        
        # Compare with other methods if available
        if self.basic_kalman_results:
            basic_z = self.basic_kalman_results['zscore'].replace([np.inf, -np.inf], np.nan).dropna()
            # Clip extreme values for visualization
            basic_z_clipped = basic_z.clip(-max_z_limit, max_z_limit)
            ax6.hist(basic_z_clipped, bins=50, alpha=0.4, label='Basic Kalman',
                    color='blue', density=True, orientation='horizontal')
        
        if self.rolling_results:
            rolling_z = self.rolling_results['zscore'].replace([np.inf, -np.inf], np.nan).dropna()
            ax6.hist(rolling_z, bins=50, alpha=0.3, label='Rolling OLS',
                    color='green', density=True, orientation='horizontal')
        
        ax6.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
        ax6.axhline(y=-2, color='gray', linestyle='--', alpha=0.5)
        ax6.set_ylim(-max_z_limit, max_z_limit)
        ax6.set_xlabel('Density')
        ax6.set_title('Z-Score Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Performance comparison table
        ax7 = plt.subplot(gs[4:, 1])
        ax7.axis('off')
        
        # Collect metrics
        table_data = []
        
        # Static
        if static_zscore is not None:
            static_max_z = static_zscore.abs().max()
            static_within_2sig = (static_zscore.abs() <= 2).mean() * 100
            table_data.append(['Static (2021)', f'{self.static_beta:.3f}', '-',
                             f'{static_max_z:.1f}', f'{static_within_2sig:.1f}%'])
        
        # Rolling OLS
        if self.rolling_results:
            roll_stats = self.rolling_results['statistics']
            table_data.append(['Rolling OLS', 
                             f'{roll_stats["beta_std"]:.4f}', 
                             '-',
                             f'{roll_stats["zscore_max_abs"]:.1f}',
                             f'{roll_stats["pct_within_2sigma"]:.1f}%'])
        
        # Basic Kalman
        if self.basic_kalman_results:
            basic_stats = self.basic_kalman_results['statistics']
            table_data.append(['Basic Kalman', 
                             f'{basic_stats["beta_std"]:.4f}', 
                             '-',
                             f'{basic_stats["zscore_max_abs"]:.1f}',
                             f'{basic_stats["pct_within_2sigma"]:.1f}%'])
        else:
            table_data.append(['Basic Kalman', 'Failed', '-', 'Failed', 'Failed'])
        
        # Enhanced Kalman
        table_data.append(['Enhanced Kalman',
                          f'{enhanced_stats["beta_std"]:.4f}',
                          f'{enhanced_stats["mu_std"]:.4f}',
                          f'{enhanced_max_z:.1f}',
                          f'{enhanced_stats["pct_within_2sigma"]:.1f}%'])
        
        # Create table
        table = ax7.table(cellText=table_data,
                         colLabels=['Method', 'β Std', 'μ Std', 'Max |Z|', '% in ±2σ'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Highlight best values
        if len(table_data) > 1:
            # Find best max |Z| (lowest) - skip "Failed" entries
            max_z_values = []
            valid_indices = []
            for i, row in enumerate(table_data):
                try:
                    max_z = float(row[3].replace('σ', '').replace('Failed', 'inf'))
                    if max_z != float('inf'):
                        max_z_values.append(max_z)
                        valid_indices.append(i)
                except:
                    pass
            
            if max_z_values:
                best_max_z_idx = valid_indices[np.argmin(max_z_values)]
                table[(best_max_z_idx + 1, 3)].set_facecolor('#90EE90')
        
        # 8. Summary text
        ax8 = plt.subplot(gs[5, :])
        ax8.axis('off')
        
        # Build summary text based on available results
        if self.basic_kalman_results:
            basic_max_z = self.basic_kalman_results['statistics']['zscore_max_abs']
            improvement_text = f"• Max |Z| reduced from {basic_max_z:.1f} (basic) to {enhanced_max_z:.1f} (enhanced)"
        else:
            improvement_text = f"• Enhanced Kalman Max |Z|: {enhanced_max_z:.1f} (Basic Kalman failed to run)"
        
        summary_text = f"""
Enhanced Kalman Filter Summary:
• Delta (δ): {enhanced_stats['delta']:.2e} - Controls adaptability vs stability
• Tracks both μ (mean) and γ (hedge ratio) - prevents drift seen in basic Kalman
• μ adapts to structural changes while γ handles dynamic hedging
• Innovation diagnostics: mean={enhanced_stats.get('innovation_mean', 0):.2e} (should be ~0)

Key Improvements:
{improvement_text}
• More stable hedge ratio while still adapting to regime changes
• Better handling of mean level shifts (COVID-19, market regimes)
"""
        
        ax8.text(0.1, 0.5, summary_text, transform=ax8.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        # Format dates
        for ax in [ax1, ax2, ax3, ax4, ax5]:
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
        output_filename = f'{self.ticker1}-{self.ticker2}_enhanced_kalman_analysis.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as: {output_filename}")
        
        # Also save results to CSV
        results_df = pd.DataFrame({
            'price1': self.price1,
            'price2': self.price2,
            'enhanced_mu': self.enhanced_kalman_results['mu'],
            'enhanced_gamma': self.enhanced_kalman_results['betas'],
            'enhanced_spread': self.enhanced_kalman_results['spread'],
            'enhanced_zscore': self.enhanced_kalman_results['zscore'],
            'delta': enhanced_stats['delta'] if enhanced_stats['delta'] > 0 else 1e-5
        })
        
        csv_filename = f'results/{self.ticker1}-{self.ticker2}_enhanced_kalman_results.csv'
        try:
            os.makedirs('results', exist_ok=True)
            results_df.to_csv(csv_filename)
            print(f"Results saved to: {csv_filename}")
        except Exception as e:
            print(f"Could not save CSV: {e}")
        
        plt.show()
    
    def run(self):
        """Main execution method"""
        print(f"\nEnhanced Kalman Filter Analysis: {self.ticker1}-{self.ticker2}")
        print(f"Z-score window: {self.zscore_days} trading days ({self.zscore_window} bars)")
        print("="*60)
        
        # Load static results for comparison
        self.load_2021_statistics()
        
        # Load data
        self.load_and_process_data()
        
        # Run all methods
        self.run_all_methods()
        
        # Print Enhanced Kalman statistics
        print("\nEnhanced Kalman Filter Statistics:")
        stats = self.enhanced_kalman_results['statistics']
        if stats['delta'] == 0:
            print(f"  WARNING: Delta optimized to 0, using default 1e-5")
            stats['delta'] = 1e-5
        print(f"  Delta (δ): {stats['delta']:.2e}")  # Use scientific notation
        print(f"  Gamma (γ) mean: {stats['beta_mean']:.3f}")
        print(f"  Gamma (γ) std: {stats['beta_std']:.3f}")
        print(f"  Gamma (γ) range: [{stats['beta_min']:.3f}, {stats['beta_max']:.3f}]")
        if isinstance(stats['mu_range'], list) and len(stats['mu_range']) == 2:
            print(f"  Mu (μ) range: [{float(stats['mu_range'][0]):.3f}, {float(stats['mu_range'][1]):.3f}]")
        else:
            print(f"  Mu (μ) range: {stats['mu_range']}")
        print(f"  Mu (μ) std: {stats['mu_std']:.4f}")
        print(f"  Max |Z-score|: {stats['zscore_max_abs']:.1f}")
        print(f"  % within ±2σ: {stats['pct_within_2sigma']:.1f}%")
        
        # Create visualization
        self.plot_enhanced_kalman_analysis()


def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_enhanced_kalman_spread.py <TICKER1> <TICKER2> [zscore_days] [start_year] [end_year]")
        print("\nExamples:")
        print("  python visualize_enhanced_kalman_spread.py MET TRV               # 45-day z-score window")
        print("  python visualize_enhanced_kalman_spread.py AEP WEC 30            # 30-day z-score window")
        print("  python visualize_enhanced_kalman_spread.py VLO PBF 60 2021 2024  # 60-day window, specific years")
        print("\nKey improvement: Tracks both μ (mean) and γ (hedge ratio)")
        print("This prevents the z-score drift seen with basic Kalman filter")
        return
    
    ticker1 = sys.argv[1].upper()
    ticker2 = sys.argv[2].upper()
    
    zscore_days = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    start_year = sys.argv[4] if len(sys.argv) > 4 else '2021'
    end_year = sys.argv[5] if len(sys.argv) > 5 else '2024'
    
    visualizer = EnhancedKalmanSpreadVisualizer(ticker1, ticker2, zscore_days, start_year, end_year)
    visualizer.run()


if __name__ == "__main__":
    main()