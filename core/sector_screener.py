import os
import pandas as pd
from itertools import combinations
from .pair_analyzer import PairAnalyzer

class SectorScreener:
    def __init__(self, sector_name, folder_name, ticker_dict, 
                 data_base_path='/Users/aridelondavidwinayu/Downloads/Project/EODHD API/us_equity_data',
                 year='2024', resample_freq='15min'):
        self.sector_name = sector_name
        self.folder_name = folder_name
        self.ticker_dict = ticker_dict
        self.data_base_path = data_base_path
        self.year = year
        self.resample_freq = resample_freq
        self.results = []
        self.available_tickers = {}
        self.missing_tickers = []
    
    def find_ticker_file(self, ticker):
        primary_path = os.path.join(self.data_base_path, self.year, self.folder_name,
                                   f'{ticker}_US_{self.year}_1min_ET_regular.csv')
        if os.path.exists(primary_path):
            return primary_path
        
        year_path = os.path.join(self.data_base_path, self.year)
        for folder in os.listdir(year_path):
            folder_path = os.path.join(year_path, folder)
            if os.path.isdir(folder_path):
                file_path = os.path.join(folder_path, f'{ticker}_US_{self.year}_1min_ET_regular.csv')
                if os.path.exists(file_path):
                    return file_path
        
        return None
    
    def check_data_availability(self):
        all_tickers = [ticker for group in self.ticker_dict.values() for ticker in group]
        
        for ticker in all_tickers:
            filepath = self.find_ticker_file(ticker)
            if filepath:
                self.available_tickers[ticker] = filepath

                actual_folder = filepath.split('/')[-2]
                if actual_folder != self.folder_name:
                    print(f"  Found {ticker} in {actual_folder}")
            else:
                self.missing_tickers.append(ticker)
        
        if self.missing_tickers:
            print(f"Missing data for: {', '.join(self.missing_tickers)}")
        
        return list(self.available_tickers.keys())
    
    def get_subsector(self, ticker):
        for subsector, tickers in self.ticker_dict.items():
            if ticker in tickers:
                return subsector
        return 'unknown'
    
    def analyze_pair(self, ticker1, ticker2):
        file1 = self.available_tickers[ticker1]
        file2 = self.available_tickers[ticker2]
        
        try:
            analyzer = PairAnalyzer(ticker1, ticker2, file1, file2, self.resample_freq)
            
            num_obs = analyzer.load_and_prepare_data()
            
            results = analyzer.run_cointegration_tests()
            results['data_points'] = num_obs
            
            results['subsector1'] = self.get_subsector(ticker1)
            results['subsector2'] = self.get_subsector(ticker2)
            
            return results
            
        except Exception as e:
            print(f"  Error: {str(e)[:50]}...")
            return None
    
    def screen_all_pairs(self):
        ticker_list = list(self.available_tickers.keys())
        pairs = list(combinations(ticker_list, 2))
        total_pairs = len(pairs)
        
        for i, (ticker1, ticker2) in enumerate(pairs, 1):
            print(f"[{i}/{total_pairs}] Analyzing {ticker1}-{ticker2}...", end=' ')
            
            result = self.analyze_pair(ticker1, ticker2)
            if result:
                self.results.append(result)
                print(f"p-value: {result['coint_pvalue']:.4f}")
            else:
                print("Failed")
        
        return self.results
    
    def get_results_summary(self):
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        df['same_subsector'] = df['subsector1'] == df['subsector2']
        df = df.sort_values('coint_pvalue')
        
        summary_cols = ['pair', 'subsector1', 'subsector2', 'same_subsector',
                       'coint_pvalue', 'adf_pvalue', 'ar_coefficient', 'halflife', 
                       'return_correlation', 'beta', 'durbin_watson']
        
        return df[summary_cols]
    
    def save_results(self, results_path='./results'):
        os.makedirs(results_path, exist_ok=True)
        
        if self.results:
            base_name = f'{self.sector_name.lower()}_{self.year}'
            
            df_full = pd.DataFrame(self.results)
            df_full.to_csv(os.path.join(results_path, f'{base_name}_full.csv'), 
                        index=False)
            
            df_summary = self.get_results_summary()
            df_summary.to_csv(os.path.join(results_path, f'{base_name}_summary.csv'), 
                            index=False)
            
            locations_df = pd.DataFrame([
                {'ticker': ticker, 'filepath': path, 'folder': path.split('/')[-2]}
                for ticker, path in self.available_tickers.items()
            ])
            locations_df.to_csv(os.path.join(results_path, f'{base_name}_ticker_locations.csv'),
                            index=False)
            
            print(f"\nResults saved to {results_path}/")
            return df_summary
        
        return pd.DataFrame()
    
    def run_analysis(self):
        self.check_data_availability()
        
        if not self.available_tickers:
            print(f"No data available for {self.sector_name}")
            return pd.DataFrame()
        
        self.screen_all_pairs()
        
        summary = self.save_results()
        
        self.display_results(summary)
        
        return summary
    
    def display_results(self, summary_df):
        total_pairs = len(self.results)
        if total_pairs == 0:
            print("No pairs analyzed")
            return
            
        cointegrated_005 = sum(1 for r in self.results if r['coint_pvalue'] < 0.05)
        cointegrated_010 = sum(1 for r in self.results if r['coint_pvalue'] < 0.10)

        if not summary_df.empty:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(summary_df.head(10).round(4))