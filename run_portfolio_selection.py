import sys
import os

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.portfolio_selector import SectorPortfolioSelector

def print_usage():
    print("Usage: python run_portfolio_selection.py <sector> [year] [min_connectivity]")
    print("\nExamples:")
    print("  python run_portfolio_selection.py financials")
    print("  python run_portfolio_selection.py energy 2021 0.7")
    print("  python run_portfolio_selection.py all  # process all sectors")
    print("\nAvailable sectors:")
    print("  financials, energy, technology, consumer_staples, healthcare,")
    print("  reits, industrials, utilities, materials")

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print_usage()
        return
    
    # Parse arguments
    sector_name = sys.argv[1].lower()
    year = sys.argv[2] if len(sys.argv) > 2 else '2021'
    min_connectivity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    
    # Valid sectors
    valid_sectors = [
        'financials', 'energy', 'technology', 'consumer_staples', 
        'health_care', 'reits', 'industrials', 'utilities', 'materials'
    ]
    
    # Check if valid sector or 'all'
    if sector_name != 'all' and sector_name not in valid_sectors:
        print(f"Error: Unknown sector '{sector_name}'")
        print_usage()
        return
    
    print("Portfolio Selection from Cointegrated Pairs")
    print("=" * 60)
    print(f"Sector: {sector_name}")
    print(f"Year: {year}")
    print(f"Min Connectivity: {min_connectivity}")
    print(f"Portfolio Size: 4-6 assets")
    
    # Set paths
    results_path = '/Users/aridelondavidwinayu/Downloads/Project/stat_arb_project/results'
    
    # Initialize selector
    selector = SectorPortfolioSelector(
        results_path=results_path,
        min_size=4,
        max_size=6,
        min_connectivity=min_connectivity,
        top_k=10  # Keep top 10 portfolios per sector
    )
    
    # Process sector(s)
    if sector_name == 'all':
        all_portfolios = selector.process_all_sectors()
        output_filename = f'portfolio_candidates_all_{year}.csv'
    else:
        all_portfolios = selector.process_sector(sector_name, year)
        output_filename = f'portfolio_candidates_{sector_name}_{year}.csv'
    
    # Export results
    if all_portfolios:
        df_results = selector.export_results(all_portfolios, output_filename)
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total portfolios found: {len(all_portfolios)}")
        
        if sector_name == 'all':
            # Group by sector
            sector_counts = {}
            for p in all_portfolios:
                sector = p['sector']
                if sector not in sector_counts:
                    sector_counts[sector] = 0
                sector_counts[sector] += 1
            
            print("\nPortfolios by sector:")
            for sector, count in sorted(sector_counts.items()):
                print(f"  {sector}: {count}")
        
        # Connectivity distribution
        connectivities = [p['connectivity'] for p in all_portfolios]
        if connectivities:
            print(f"\nConnectivity distribution:")
            print(f"  Min: {min(connectivities):.2f}")
            print(f"  Max: {max(connectivities):.2f}")
            print(f"  Mean: {sum(connectivities)/len(connectivities):.2f}")
        
        # Size distribution
        sizes = [p['size'] for p in all_portfolios]
        print(f"\nPortfolio size distribution:")
        for size in range(4, 7):
            count = sizes.count(size)
            if count > 0:
                print(f"  {size} assets: {count} portfolios")
        
        # Show top 5 portfolios
        print(f"\nTop 5 portfolios by score:")
        print("-" * 80)
        print(f"{'Assets':<35} {'Conn':<6} {'AvgP':<8} {'Score':<8} {'Composition':<30}")
        print("-" * 80)
        
        for p in all_portfolios[:5]:
            assets_str = ','.join(p['assets'])
            if len(assets_str) > 33:
                assets_str = assets_str[:30] + '...'
            
            print(f"{assets_str:<35} {p['connectivity']:<6.2f} "
                  f"{p['avg_pvalue']:<8.4f} {p['score']:<8.3f} "
                  f"{p['composition']:<30}")
        
        # Mention missing pairs for top portfolio
        if all_portfolios and all_portfolios[0]['missing_pairs']:
            print(f"\nNote: Top portfolio missing pairs: {', '.join(all_portfolios[0]['missing_pairs'])}")
    
    else:
        print(f"\nNo portfolios found meeting the criteria for {sector_name}!")

if __name__ == "__main__":
    main()