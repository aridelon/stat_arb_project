import pandas as pd
import os
from pathlib import Path

LOGICAL_RELATIONSHIPS = {
    # Financials
    ('mega_banks', 'mega_banks'): 'same_business',
    ('mega_banks', 'regional_commercial_banks'): 'similar_banking',
    ('investment_banks_asset_mgmt', 'investment_banks_asset_mgmt'): 'same_business',
    ('insurance', 'insurance'): 'same_business',
    ('payments_fintech', 'payments_fintech'): 'same_business',
    
    # Energy
    ('oil_majors', 'oil_majors'): 'same_business',
    ('oil_majors', 'oil_exploration'): 'upstream_oil',
    ('oil_exploration', 'oil_exploration'): 'same_business',
    ('oil_services', 'oil_services'): 'same_business',
    ('refiners', 'refiners'): 'same_business',
    ('natural_gas_midstream', 'natural_gas_midstream'): 'same_business',
    
    # Technology
    ('semiconductors', 'semiconductors'): 'same_business',
    ('software_giants', 'software_giants'): 'same_business',
    ('hardware_devices', 'hardware_devices'): 'same_business',
    ('enterprise_software', 'enterprise_software'): 'same_business',
    ('payments_fintech', 'payments_fintech'): 'same_business',
    
    # Consumer Staples
    ('beverages', 'beverages'): 'same_business',
    ('food_products', 'food_products'): 'same_business',
    ('household_products', 'household_products'): 'same_business',
    ('retail_staples', 'retail_staples'): 'same_business',
    ('tobacco', 'tobacco'): 'same_business',
    
    # REITs
    ('retail_reits', 'retail_reits'): 'same_property_type',
    ('residential_reits', 'residential_reits'): 'same_property_type',
    ('industrial_reits', 'industrial_reits'): 'same_property_type',
    ('office_reits', 'office_reits'): 'same_property_type',
    ('healthcare_reits', 'healthcare_reits'): 'same_property_type',
    ('specialized_reits', 'specialized_reits'): 'same_property_type',
    
    # Industrials
    ('airlines', 'airlines'): 'same_business',
    ('defense_aerospace', 'defense_aerospace'): 'same_business',
    ('railroads', 'railroads'): 'same_business',
    ('logistics_delivery', 'logistics_delivery'): 'same_business',
    ('machinery_equipment', 'machinery_equipment'): 'same_business',
    
    # Materials
    ('metals_mining', 'metals_mining'): 'same_commodity',
    ('chemicals_specialty', 'chemicals_specialty'): 'same_business',
    ('construction_materials', 'construction_materials'): 'same_business',
    ('containers_packaging', 'containers_packaging'): 'same_business',
    
    # Utilities
    ('electric_utilities', 'electric_utilities'): 'same_business',
    ('multi_utilities', 'multi_utilities'): 'same_business',
    ('gas_utilities', 'gas_utilities'): 'same_business',
}

def load_sector_summaries(results_path):
    all_pairs = []
    
    sectors = [
        'financials', 'energy', 'technology', 'consumer_staples', 
        'healthcare', 'reits', 'industrials', 'utilities', 'materials'
    ]
    
    for sector in sectors:
        filename = f"{sector}_2021_summary.csv"
        filepath = os.path.join(results_path, filename)
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df['sector'] = sector
                all_pairs.append(df)
                print(f"✓ Loaded {filename}: {len(df)} pairs")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        else:
            print(f"✗ File not found: {filename}")
    
    if all_pairs:
        combined_df = pd.concat(all_pairs, ignore_index=True)
        print(f"\nTotal pairs loaded: {len(combined_df)}")
        return combined_df
    else:
        print("No data loaded!")
        return pd.DataFrame()

def assess_logical_relationship(row):
    subsector1 = row['subsector1']
    subsector2 = row['subsector2']
    
    if (subsector1, subsector2) in LOGICAL_RELATIONSHIPS:
        return LOGICAL_RELATIONSHIPS[(subsector1, subsector2)]
    elif (subsector2, subsector1) in LOGICAL_RELATIONSHIPS:
        return LOGICAL_RELATIONSHIPS[(subsector2, subsector1)]
    else:
        return None

def filter_logical_pairs(df):
    exclude_pairs = ['GOOGL-GOOG', 'GOOG-GOOGL']

    df['logical_relationship'] = df.apply(assess_logical_relationship, axis=1)
    
    # Convert half-life to days
    bars_per_day = 26
    df['halflife_days'] = df['halflife'] / bars_per_day
    
    # Filter for logical pairs with good stats
    logical_pairs = df[
        (~df['pair'].isin(exclude_pairs)) &
        (df['logical_relationship'].notna()) &  # Has logical relationship
        (df['coint_pvalue'] < 0.01) &          # Strong cointegration
        (df['halflife'] > 52) &                 # > 2 days
        (df['halflife'] < 520) &                # < 20 days
        (df['return_correlation'] > 0.4) &      # Higher correlation threshold
        (df['beta'].abs() < 3)                  # Reasonable hedge ratio
    ].copy()
    
    print(f"\nLogical pairs with good statistics: {len(logical_pairs)}")
    
    # Also get statistically strong pairs that might have indirect relationships
    strong_stats = df[
        (~df['pair'].isin(exclude_pairs)) &
        (df['coint_pvalue'] < 0.005) &         # Very strong cointegration
        (df['halflife'] > 78) &                 # > 3 days
        (df['halflife'] < 390) &                # < 15 days
        (df['return_correlation'] > 0.5) &      # High correlation
        (df['beta'].abs() < 2)                 
    ].copy()
    
    print(f"Very strong statistical pairs: {len(strong_stats)}")
    
    # Combine and remove duplicates
    combined = pd.concat([logical_pairs, strong_stats]).drop_duplicates(subset='pair')
    
    print(f"\nTotal filtered pairs: {len(combined)}")
    print("\nFilter criteria for logical pairs:")
    print("- Must have logical economic relationship (same/similar subsector)")
    print("- Cointegration p-value < 0.01")
    print("- Half-life between 2-20 days")
    print("- Return correlation > 0.4")
    print("- |Beta| < 3")
    
    return combined.sort_values('coint_pvalue')

def rank_pairs(df):
    # Score logical relationships
    relationship_scores = {
        'same_business': 0,
        'same_property_type': 0,
        'same_commodity': 0,
        'upstream_oil': 1,
        'similar_banking': 1,
    }
    
    df['relationship_score'] = df['logical_relationship'].map(
        lambda x: relationship_scores.get(x, 10) if pd.notna(x) else 10
    )
    
    # Optimal half-life around 5-10 days (130-260 bars)
    optimal_halflife = 195  # 7.5 days
    df['halflife_score'] = abs(df['halflife'] - optimal_halflife) / optimal_halflife
    
    # Composite score (lower is better)
    df['composite_score'] = (
        df['relationship_score'] * 100 +  # Heavily weight logical relationships
        df['coint_pvalue'] * 50 +
        df['halflife_score'] * 20 +
        (1 - df['return_correlation']) * 10
    )
    
    return df.sort_values('composite_score')

def display_results(df):
    top_pairs = df.head(25)
    
    for idx, (_, row) in enumerate(top_pairs.iterrows(), 1):
        relationship = row['logical_relationship'] if pd.notna(row['logical_relationship']) else 'indirect'
        print(f"\n{idx}. {row['pair']} ({row['sector']})")
        print(f"   Subsectors: {row['subsector1']} - {row['subsector2']}")
        print(f"   Relationship: {relationship}")
        print(f"   Coint p-value: {row['coint_pvalue']:.5f}")
        print(f"   Half-life: {row['halflife']:.0f} bars ({row['halflife_days']:.1f} days)")
        print(f"   Beta: {row['beta']:.3f}")
        print(f"   Correlation: {row['return_correlation']:.3f}")
    
    relationship_groups = top_pairs.groupby('logical_relationship').size()
    for rel, count in relationship_groups.items():
        if pd.notna(rel):
            print(f"{rel}: {count} pairs")
    
    if top_pairs['logical_relationship'].isna().any():
        print(f"indirect/statistical: {top_pairs['logical_relationship'].isna().sum()} pairs")
    
    return top_pairs

def save_results(df, output_path):
    output_file = os.path.join(output_path, 'logical_pairs_2021.csv')
    
    save_cols = [
        'pair', 'sector', 'subsector1', 'subsector2', 'logical_relationship',
        'coint_pvalue', 'adf_pvalue', 'halflife', 'halflife_days',
        'return_correlation', 'beta', 'composite_score'
    ]
    
    df[save_cols].to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

def main():
    project_path = "/Users/aridelondavidwinayu/Downloads/Project/stat_arb_project"
    results_path = os.path.join(project_path, "results")
    
    # Load data
    df_all = load_sector_summaries(results_path)
    
    if df_all.empty:
        return
    
    # Filter for logical pairs
    df_filtered = filter_logical_pairs(df_all)
    
    if df_filtered.empty:
        print("No pairs meet the criteria!")
        return
    
    # Rank pairs
    df_ranked = rank_pairs(df_filtered)
    
    # Display results
    top_pairs = display_results(df_ranked)
    
    # Save results
    save_results(df_ranked, results_path)
    
    # Specific recommendations
    recommendations = df_ranked[df_ranked['logical_relationship'].notna()].head(5)
    
    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
        print(f"\n{idx}. {row['pair']} ({row['sector']})")
        print(f"   Relationship: {row['logical_relationship']}")
        print(f"   Stats: p={row['coint_pvalue']:.5f}, hl={row['halflife_days']:.1f}d, corr={row['return_correlation']:.3f}")

if __name__ == "__main__":
    main()