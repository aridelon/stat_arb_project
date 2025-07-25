import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

def analyze_johansen_results(csv_file='results/johansen_analysis_all_2021_summary.csv'):
    # Read the CSV
    df = pd.read_csv(csv_file)
    
    print("COMPREHENSIVE JOHANSEN ANALYSIS RESULTS")
    print("=" * 80)
    
    print("\nRANK INTERPRETATION GUIDE:")
    print("-" * 40)
    print("For N assets, cointegration rank r means:")
    print("  • r = 0: No cointegration")
    print("  • 0 < r < N: Has r cointegrating vectors (good!)")
    print("  • r = N: All assets are stationary (no cointegration)")
    print("  • Higher rank (up to N-1) = MORE cointegrating vectors = BETTER")
    print("  • Example: 4 assets with rank 3 = 3 cointegrating vectors (best case)")
    
    # Basic statistics
    print(f"\nTotal portfolios analyzed: {len(df)}")
    print(f"Sectors covered: {df['sector'].nunique()}")
    print(f"Portfolios per sector: {len(df) // df['sector'].nunique()}")
    
    weight_cols = [col for col in df.columns if col.startswith('weight_')]
    df['num_assets'] = df[weight_cols].notna().sum(axis=1)

    for idx, row in df.iterrows():
        if row['num_assets'] == 0:
            # Count assets from the string
            if pd.notna(row['assets']):
                num_assets = len([a.strip() for a in row['assets'].split('+') if a.strip()])
                df.at[idx, 'num_assets'] = num_assets
            else:
                df.at[idx, 'num_assets'] = 4  # Default assumption
    
    # Identify portfolios with cointegration
    # Cointegration exists when 0 < rank < num_assets
    df['has_cointegration'] = (df['coint_rank'] > 0) & (df['coint_rank'] < df['num_assets'])
    df['num_coint_vectors'] = df['coint_rank']  # Rank IS the number of cointegrating vectors
    
    print(f"\nPortfolios with cointegration: {df['has_cointegration'].sum()} out of {len(df)}")
    
    # Show distribution of ranks
    print("\nCointegration Rank Distribution:")
    print("(Higher rank = more cointegrating relationships, but rank = num_assets means NO cointegration)")
    print("-" * 80)
    rank_dist = df.groupby(['coint_rank', 'num_assets']).size().reset_index(name='count')
    for _, row in rank_dist.iterrows():
        if row['coint_rank'] == 0:
            status = "NO cointegration"
        elif row['coint_rank'] == row['num_assets']:
            status = "NO cointegration (all assets stationary)"
        else:
            status = f"{row['coint_rank']} cointegrating vectors (good!)"
        print(f"  Rank {row['coint_rank']} (for {row['num_assets']}-asset portfolios): {row['count']} portfolios - {status}")
    
    # Best portfolios by sector
    print("\n\nBEST PORTFOLIOS BY SECTOR (highest rank = most cointegrating relationships):")
    print("-" * 80)
    
    for sector in sorted(df['sector'].unique()):
        sector_df = df[df['sector'] == sector]
        # Get portfolios with cointegration
        sector_coint = sector_df[sector_df['has_cointegration']]
        
        if len(sector_coint) > 0:
            # Best = highest number of cointegrating vectors
            best = sector_coint.loc[sector_coint['num_coint_vectors'].idxmax()]
        else:
            # No cointegration in sector
            best = sector_df.iloc[0]  # Just show first one
        
        print(f"\n{sector.upper()}:")
        print(f"  Portfolio: {best['portfolio_id']}")
        print(f"  Assets: {best['assets']}")
        print(f"  Rank: {best['coint_rank']} (out of {best['num_assets']} assets)")
        
        if best['has_cointegration']:
            print(f"  ✅ HAS COINTEGRATION: {best['num_coint_vectors']} cointegrating vectors")
            if pd.notna(best['half_life_days']):
                print(f"  Half-life: {best['half_life_days']:.1f} days")
                print(f"  Max |Z|: {best['max_abs_zscore']:.1f}")
                print(f"  % within ±2σ: {best['pct_within_2sigma']:.1f}%")
        else:
            print(f"  ❌ NO COINTEGRATION (rank = {best['coint_rank']})")
    
    # Overall best portfolios
    print("\n\nTOP 5 PORTFOLIOS OVERALL (by number of cointegrating vectors):")
    print("-" * 80)
    
    best_overall = df[df['has_cointegration']].nlargest(5, 'num_coint_vectors')
    
    for idx, row in best_overall.iterrows():
        print(f"\n{row['sector'].upper()} - {row['portfolio_id']}:")
        print(f"  Assets: {row['assets']}")
        print(f"  Cointegrating vectors: {row['num_coint_vectors']} (rank {row['coint_rank']})")
        print(f"  Eigenvalue 1: {row['eigenvalue_1']:.4f}")
        
        if pd.notna(row['half_life_days']):
            print(f"  Half-life: {row['half_life_days']:.1f} days")
            print(f"  Max |Z|: {row['max_abs_zscore']:.1f}")
            
        # Show weights for the primary cointegrating vector
        print("  Weights:")
        assets = [a.strip() for a in row['assets'].split('+')]
        for asset in assets:
            weight_col = f'weight_{asset}'
            if weight_col in row and pd.notna(row[weight_col]):
                print(f"    {asset}: {row[weight_col]:.3f}")
    
    # Summary by quality metrics
    if df['has_cointegration'].any():
        print("\n\nQUALITY METRICS FOR COINTEGRATED PORTFOLIOS:")
        print("-" * 80)
        
        coint_df = df[df['has_cointegration'] & df['half_life_days'].notna()]
        
        if len(coint_df) > 0:
            print(f"\nHalf-life distribution (days):")
            print(f"  Min: {coint_df['half_life_days'].min():.1f}")
            print(f"  Median: {coint_df['half_life_days'].median():.1f}")
            print(f"  Max: {coint_df['half_life_days'].max():.1f}")
            
            print(f"\nMax |Z-score| distribution:")
            print(f"  Min: {coint_df['max_abs_zscore'].min():.1f}")
            print(f"  Median: {coint_df['max_abs_zscore'].median():.1f}")
            print(f"  Max: {coint_df['max_abs_zscore'].max():.1f}")
            
            print(f"\n% within ±2σ distribution:")
            print(f"  Min: {coint_df['pct_within_2sigma'].min():.1f}%")
            print(f"  Median: {coint_df['pct_within_2sigma'].median():.1f}%")
            print(f"  Max: {coint_df['pct_within_2sigma'].max():.1f}%")
    
    # Recommendations
    print("\n\nRECOMMENDATIONS:")
    print("-" * 80)
    
    if df['has_cointegration'].any():
        print("\n1. VISUALIZE the best portfolios (highest rank):")
        # Sort by rank descending, but exclude rank = num_assets
        best_for_viz = df[df['has_cointegration']].nlargest(3, 'coint_rank')
        for _, row in best_for_viz.iterrows():
            assets = ' '.join([a.strip() for a in row['assets'].split('+')])
            print(f"   python visualize_portfolio_spread.py {assets} 2021 2024")
    
    no_coint_sectors = df[~df.groupby('sector')['has_cointegration'].transform('any')]['sector'].unique()
    if len(no_coint_sectors) > 0:
        print(f"\n2. SECTORS WITHOUT COINTEGRATION: {', '.join(no_coint_sectors)}")
        print("   Consider:")
        print("   - Using pairs trading instead")
        print("   - Trying 5-6 asset portfolios")
        print("   - Applying enhanced Kalman filter approach")
    
    print("\n3. FOCUS on sectors with successful cointegration for initial trading")
    
    # Add clarification about rank interpretation
    print("\n\nNOTE ON RANK INTERPRETATION:")
    print("-" * 80)
    print("Common confusion: In Johansen test, the 'rank' is the number of cointegrating vectors.")
    print("• Higher rank (up to N-1) = MORE cointegrating vectors = BETTER")
    print("• This is different from some other contexts where 'lower rank' might be better")
    print("• Think of it as: more cointegrating vectors = more stable relationships = more trading opportunities")
    
    return df


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'results/johansen_analysis_all_2021_summary.csv'
    analyze_johansen_results(csv_file)