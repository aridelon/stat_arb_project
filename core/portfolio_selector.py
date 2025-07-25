import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict, Counter
import os
import json

class SectorGraph:
    def __init__(self, sector_name, year='2021'):
        self.sector_name = sector_name
        self.year = year
        self.graph = nx.Graph()
        self.ticker_subsectors = {}
        self.pair_data = {}
        
    def load_sector_pairs(self, filepath):
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} pairs from {filepath}")
            
            df_coint = df[df['coint_pvalue'] < 0.05].copy()
            print(f"Found {len(df_coint)} cointegrated pairs (p < 0.05)")
            
            for _, row in df_coint.iterrows():
                tickers = row['pair'].split('-')
                if len(tickers) != 2:
                    continue
                    
                ticker1, ticker2 = tickers
                
                self.ticker_subsectors[ticker1] = row.get('subsector1', 'unknown')
                self.ticker_subsectors[ticker2] = row.get('subsector2', 'unknown')
                
                self.graph.add_edge(ticker1, ticker2, 
                                  coint_pvalue=row['coint_pvalue'],
                                  halflife=row.get('halflife', np.nan),
                                  correlation=row.get('return_correlation', np.nan))

                self.pair_data[frozenset([ticker1, ticker2])] = {
                    'coint_pvalue': row['coint_pvalue'],
                    'halflife': row.get('halflife', np.nan),
                    'correlation': row.get('return_correlation', np.nan)
                }
            
            print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True
            
        except Exception as e:
            print(f"Error loading sector pairs: {e}")
            return False
    
    def get_all_tickers(self):
        return list(self.graph.nodes())
    
    def get_ticker_subsector(self, ticker):
        return self.ticker_subsectors.get(ticker, 'unknown')
    
    def get_edge_data(self, ticker1, ticker2):
        pair_key = frozenset([ticker1, ticker2])
        return self.pair_data.get(pair_key, {})


class PortfolioFinder:    
    def __init__(self, min_size=4, max_size=6, min_connectivity=0.6):
        self.min_size = min_size
        self.max_size = max_size
        self.min_connectivity = min_connectivity
        
    def find_near_cliques(self, graph):
        portfolios = []
        nodes = list(graph.nodes())
        
        for size in range(self.min_size, self.max_size + 1):
            for node_subset in combinations(nodes, size):
                subgraph = graph.subgraph(node_subset)
                actual_edges = subgraph.number_of_edges()
                possible_edges = size * (size - 1) // 2
                connectivity = actual_edges / possible_edges if possible_edges > 0 else 0
                
                if connectivity >= self.min_connectivity:
                    p_values = []
                    for edge in subgraph.edges():
                        edge_data = graph.get_edge_data(edge[0], edge[1])
                        p_values.append(edge_data.get('coint_pvalue', 1.0))
                    
                    avg_pvalue = np.mean(p_values) if p_values else 1.0
                    
                    portfolio = {
                        'assets': sorted(list(node_subset)),
                        'size': size,
                        'actual_edges': actual_edges,
                        'possible_edges': possible_edges,
                        'connectivity': connectivity,
                        'avg_pvalue': avg_pvalue,
                        'missing_pairs': self._get_missing_pairs(node_subset, subgraph)
                    }
                    portfolios.append(portfolio)
        
        return portfolios
    
    def _get_missing_pairs(self, nodes, subgraph):
        missing = []
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if not subgraph.has_edge(node1, node2):
                    missing.append(f"{node1}-{node2}")
        return missing


class PortfolioRanker:
    def __init__(self):
        self.scores = []
        
    def score_portfolio(self, portfolio):
        connectivity_score = portfolio['connectivity']
        
        strength_factor = 1 / (1 + portfolio['avg_pvalue'])
        
        score = connectivity_score * strength_factor
        
        portfolio['connectivity_score'] = connectivity_score
        portfolio['strength_factor'] = strength_factor
        portfolio['score'] = score
        
        return score
    
    def rank_portfolios(self, portfolios):
        for portfolio in portfolios:
            self.score_portfolio(portfolio)
        
        ranked = sorted(portfolios, key=lambda x: x['score'], reverse=True)
        
        for i, portfolio in enumerate(ranked):
            portfolio['rank'] = i + 1
            
        return ranked
    
    def select_top_k(self, portfolios, k=10):
        ranked = self.rank_portfolios(portfolios)
        
        return ranked[:k]


class SectorPortfolioSelector:
    
    def __init__(self, results_path, min_size=4, max_size=6, 
                 min_connectivity=0.6, top_k=10):
        self.results_path = results_path
        self.min_size = min_size
        self.max_size = max_size
        self.min_connectivity = min_connectivity
        self.top_k = top_k
        
        self.portfolio_finder = PortfolioFinder(min_size, max_size, min_connectivity)
        self.ranker = PortfolioRanker()
        
    def process_sector(self, sector_name, year='2021'):        
        filepath = os.path.join(self.results_path, f"{sector_name}_{year}_summary.csv")
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return []
        
        graph_builder = SectorGraph(sector_name, year)
        if not graph_builder.load_sector_pairs(filepath):
            return []

        portfolios = self.portfolio_finder.find_near_cliques(graph_builder.graph)
        
        if not portfolios:
            return []
   
        for portfolio in portfolios:
            composition = self._analyze_composition(portfolio['assets'], graph_builder)
            portfolio['composition'] = composition
            portfolio['sector'] = sector_name
        
        top_portfolios = self.ranker.select_top_k(portfolios, self.top_k)
        
        self._display_sector_results(sector_name, top_portfolios)
        
        return top_portfolios
    
    def _analyze_composition(self, assets, graph_builder):
        subsector_counts = Counter()
        for asset in assets:
            subsector = graph_builder.get_ticker_subsector(asset)
            subsector_counts[subsector] += 1
        
        composition_parts = []
        for subsector, count in subsector_counts.most_common():
            if subsector != 'unknown':
                composition_parts.append(f"{subsector}({count})")
        
        return ', '.join(composition_parts) if composition_parts else 'mixed'
    
    def _display_sector_results(self, sector_name, portfolios):
        for p in portfolios[:10]:
            assets_str = ','.join(p['assets'])
            if len(assets_str) > 28:
                assets_str = assets_str[:25] + '...'
            
            print(f"{p['rank']:<5} {assets_str:<30} {p['size']:<5} "
                  f"{p['connectivity']:<6.2f} {p['avg_pvalue']:<8.4f} "
                  f"{p['score']:<8.3f} {p['composition']:<30}")
    
    def process_all_sectors(self):
        all_portfolios = []
        
        sector_files = [f for f in os.listdir(self.results_path) 
                       if f.endswith('_2021_summary.csv')]
        
        sectors = [f.replace('_2021_summary.csv', '') for f in sector_files]
        
        for sector in sorted(sectors):
            sector_portfolios = self.process_sector(sector)
            all_portfolios.extend(sector_portfolios)
        
        return all_portfolios
    
    def export_results(self, portfolios, output_filename='portfolio_candidates.csv'):
        if not portfolios:
            return
        
        records = []
        for i, p in enumerate(portfolios):
            record = {
                'portfolio_id': f"{p['sector'].upper()[:3]}_{i+1:03d}",
                'sector': p['sector'],
                'assets': ','.join(p['assets']),
                'num_assets': p['size'],
                'edges': p['actual_edges'],
                'connectivity': p['connectivity'],
                'avg_pvalue': p['avg_pvalue'],
                'score': p['score'],
                'composition': p['composition'],
                'missing_pairs': ';'.join(p['missing_pairs']) if p['missing_pairs'] else ''
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        output_path = os.path.join(self.results_path, output_filename)
        df.to_csv(output_path, index=False)
        print(f"\nPortfolio candidates saved to: {output_path}")
        
        return df