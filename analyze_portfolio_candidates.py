import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.johansen_analyzer import JohansenAnalyzer
from core.data_loader import load_stock_data, resample_to_frequency, align_data


class PortfolioJohansenAnalyzer:
    """Analyze portfolio candidates using Johansen cointegration test"""
    
    def __init__(self, 
                 data_base_path='/Users/aridelondavidwinayu/Downloads/Project/EODHD API/us_equity_data',
                 results_path='results'):
        self.data_base_path = Path(data_base_path)
        self.results_path = Path(results_path)
        self.resample_freq = '15min'
        
    def find_ticker_file(self, ticker: str, year: str) -> str:
        """Find ticker file in sector folders"""
        year_path = self.data_base_path / year
        
        for folder in year_path.iterdir():
            if folder.is_dir():
                file_path = folder / f'{ticker}_US_{year}_1min_ET_regular.csv'
                if file_path.exists():
                    return str(file_path)
        return None
    
    def load_portfolio_data(self, tickers: List[str], year: str) -> pd.DataFrame:
        """Load and align data for portfolio assets"""
        price_data = {}
        
        for ticker in tickers:
            file_path = self.find_ticker_file(ticker, year)
            if not file_path:
                print(f"  WARNING: No data file for {ticker}")
                return None
                
            try:
                # Load and resample
                df_1min = load_stock_data(file_path)
                df_resampled = resample_to_frequency(df_1min, self.resample_freq)
                price_data[ticker] = df_resampled['close']
            except Exception as e:
                print(f"  ERROR loading {ticker}: {e}")
                return None
        
        # Combine and align
        df = pd.DataFrame(price_data)
        df = df.dropna()
        
        if len(df) < 1000:  # Need sufficient data
            print(f"  WARNING: Only {len(df)} observations - need more data")
            return None
            
        return df
    
    def analyze_portfolio(self, portfolio_row: pd.Series, year: str) -> Dict:
        """Run Johansen analysis on a single portfolio"""
        # Parse assets
        assets_str = portfolio_row['assets']
        assets = [a.strip() for a in assets_str.strip('[]').replace("'", "").split(',')]
        
        print(f"\nAnalyzing {portfolio_row['portfolio_id']}: {' + '.join(assets)}")
        print(f"  Connectivity: {portfolio_row['connectivity']:.1%}, Score: {portfolio_row['score']:.3f}")
        
        # Load data
        price_data = self.load_portfolio_data(assets, year)
        if price_data is None:
            return None
        
        print(f"  Loaded {len(price_data)} observations")
        
        # Initialize Johansen analyzer
        johansen = JohansenAnalyzer()  # Auto lag selection now works with your version
        
        try:
            # Perform Johansen test
            results = johansen.perform_johansen_test(price_data)
            
            # Calculate spread if cointegrated
            spread_stats = {}
            if results['rank'] > 0:
                # Get primary cointegrating vector
                weights = johansen.construct_portfolio_weights(0)
                spread = johansen.calculate_portfolio_spread(price_data, weights)
                zscore = johansen._calculate_zscore(spread, johansen.zscore_window)
                
                # Calculate statistics
                spread_stats = {
                    'weights': dict(zip(assets, weights)),
                    'spread_mean': spread.mean(),
                    'spread_std': spread.std(),
                    'half_life': johansen._calculate_half_life(spread),
                    'max_abs_zscore': zscore.abs().max(),
                    'pct_within_2sigma': (zscore.abs() <= 2).sum() / len(zscore) * 100
                }
                
                print(f"  Cointegration rank: {results['rank']}")
                print(f"  Primary eigenvalue: {results['eigenvalues'][0]:.4f}")
                print(f"  Half-life: {spread_stats['half_life']:.1f} bars ({spread_stats['half_life']/26:.1f} days)")
                print(f"  Max |Z|: {spread_stats['max_abs_zscore']:.1f}")
            else:
                print(f"  No cointegration found (rank = 0)")
            
            return {
                'portfolio_id': portfolio_row['portfolio_id'],
                'assets': assets,
                'connectivity': portfolio_row['connectivity'],
                'score': portfolio_row['score'],
                'johansen_results': results,
                'spread_stats': spread_stats
            }
            
        except Exception as e:
            print(f"  ERROR in Johansen test: {str(e)}")
            return None
    
    def analyze_sector_portfolios(self, csv_file: Path, year: str, top_n: int = 3) -> List[Dict]:
        """Analyze top N portfolios from a sector CSV file"""
        print(f"\n{'='*60}")
        print(f"Processing: {csv_file.name}")
        print('='*60)
        
        # Load portfolio candidates
        df = pd.read_csv(csv_file)
        sector = df['sector'].iloc[0] if 'sector' in df.columns else csv_file.stem.split('_')[2]
        
        # Get top N portfolios
        top_portfolios = df.head(top_n)
        
        results = []
        for idx, row in top_portfolios.iterrows():
            result = self.analyze_portfolio(row, year)
            if result:
                result['sector'] = sector
                results.append(result)
        
        return results
    
    def analyze_all_sectors(self, year: str, top_n: int = 2) -> pd.DataFrame:
        """Analyze top portfolios from all sectors"""
        # Find all portfolio candidate files
        pattern = f"portfolio_candidates_*_{year}.csv"
        csv_files = sorted(self.results_path.glob(pattern))
        
        print(f"\nFound {len(csv_files)} sector files for {year}")
        
        all_results = []
        summary_data = []
        
        for csv_file in csv_files:
            sector_results = self.analyze_sector_portfolios(csv_file, year, top_n)
            all_results.extend(sector_results)
            
            # Create summary entries
            for result in sector_results:
                if result and result['johansen_results']['rank'] > 0:
                    try:
                        summary_data.append({
                            'sector': result['sector'],
                            'portfolio_id': result['portfolio_id'],
                            'assets': ' + '.join(result['assets']),
                            'num_assets': len(result['assets']),
                            'connectivity': result['connectivity'],
                            'coint_rank': result['johansen_results']['rank'],
                            'eigenvalue_1': result['johansen_results']['eigenvalues'][0],
                            'half_life_days': result['spread_stats']['half_life'] / 26 if result['spread_stats']['half_life'] else np.nan,
                            'max_abs_zscore': result['spread_stats']['max_abs_zscore'],
                            'pct_within_2sigma': result['spread_stats']['pct_within_2sigma']
                        })
                    except Exception as e:
                        print(f"  Warning: Could not create summary for {result['portfolio_id']}: {str(e)}")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save detailed results
        self.save_detailed_results(all_results, f"johansen_analysis_all_{year}")
        
        return summary_df
    
    def save_detailed_results(self, results: List[Dict], filename_prefix: str):
        """Save detailed Johansen analysis results"""
        # Save summary
        summary_data = []
        for r in results:
            entry = {
                'sector': r['sector'],
                'portfolio_id': r['portfolio_id'],
                'assets': ' + '.join(r['assets']),
                'connectivity': r['connectivity'],
                'coint_rank': r['johansen_results']['rank'],
                'lag_order': r['johansen_results']['lag_order']
            }
            
            # Add eigenvalues
            for i, eig in enumerate(r['johansen_results']['eigenvalues'][:3]):
                entry[f'eigenvalue_{i+1}'] = eig
            
            # Add spread stats if cointegrated
            if r['spread_stats']:
                entry.update({
                    'half_life_days': r['spread_stats']['half_life'] / 26,
                    'max_abs_zscore': r['spread_stats']['max_abs_zscore'],
                    'pct_within_2sigma': r['spread_stats']['pct_within_2sigma']
                })
                
                # Add weights
                for asset, weight in r['spread_stats']['weights'].items():
                    entry[f'weight_{asset}'] = weight
            
            summary_data.append(entry)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_path / f"{filename_prefix}_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary to: {summary_file}")
        
        return summary_df
    
    def print_best_portfolios(self, summary_df: pd.DataFrame):
        """Print the best cointegrated portfolios"""
        if summary_df.empty:
            print("\nNo cointegrated portfolios found!")
            return
        
        print("\n" + "="*80)
        print("BEST COINTEGRATED PORTFOLIOS")
        print("="*80)
        
        # Sort by rank and eigenvalue
        sorted_df = summary_df.sort_values(['coint_rank', 'eigenvalue_1'], 
                                         ascending=[False, False])
        
        # Print top 10
        print(f"\nTop {min(10, len(sorted_df))} Portfolios:")
        print("-"*80)
        
        for idx, row in sorted_df.head(10).iterrows():
            print(f"\n{row['sector'].upper()} - {row['portfolio_id']}:")
            print(f"  Assets: {row['assets']}")
            print(f"  Rank: {row['coint_rank']}, Eigenvalue: {row['eigenvalue_1']:.4f}")
            print(f"  Half-life: {row['half_life_days']:.1f} days")
            print(f"  Max |Z|: {row['max_abs_zscore']:.1f}, Within ±2σ: {row['pct_within_2sigma']:.1f}%")
        
        # Best by sector
        print("\n\nBest Portfolio by Sector:")
        print("-"*80)
        
        for sector in sorted_df['sector'].unique():
            sector_best = sorted_df[sorted_df['sector'] == sector].iloc[0]
            print(f"\n{sector.upper()}:")
            print(f"  {sector_best['assets']} (Rank={sector_best['coint_rank']})")


def main():
    # Parse arguments
    sector = 'all'
    year = '2021'
    top_n = 3
    
    if len(sys.argv) > 1:
        sector = sys.argv[1].lower()
    if len(sys.argv) > 2:
        year = sys.argv[2]
    if len(sys.argv) > 3:
        top_n = int(sys.argv[3])
    
    # Initialize analyzer
    analyzer = PortfolioJohansenAnalyzer()
    
    print(f"\nJohansen/VECM Analysis of Portfolio Candidates")
    print(f"Year: {year}, Top N per sector: {top_n}")
    
    if sector == 'all':
        # Analyze all sectors
        summary_df = analyzer.analyze_all_sectors(year, top_n)
        
        # Print results
        analyzer.print_best_portfolios(summary_df)
        
        # Generate visualization commands
        if not summary_df.empty:
            print("\n\nTo visualize the best portfolios, run:")
            print("-"*60)
            
            for idx, row in summary_df.nlargest(3, 'coint_rank').iterrows():
                assets = row['assets'].split(' + ')
                cmd = f"python visualize_portfolio_spread.py {' '.join(assets)} {year}"
                print(cmd)
    
    else:
        # Analyze single sector
        csv_file = analyzer.results_path / f"portfolio_candidates_{sector}_{year}.csv"
        if not csv_file.exists():
            print(f"ERROR: File not found: {csv_file}")
            print("\nAvailable files:")
            for f in analyzer.results_path.glob(f"portfolio_candidates_*_{year}.csv"):
                print(f"  {f.name}")
            return
        
        results = analyzer.analyze_sector_portfolios(csv_file, year, top_n)
        
        # Save and display
        if results:
            summary_df = analyzer.save_detailed_results(results, f"johansen_analysis_{sector}_{year}")
            analyzer.print_best_portfolios(summary_df)


if __name__ == "__main__":
    main()