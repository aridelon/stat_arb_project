# Statistical Arbitrage Research Project

A quantitative analysis of pairs trading strategies across multiple sectors using various spread modeling techniques.

## Overview

This project implements and compares different statistical arbitrage methodologies for equity pairs trading. The analysis covers static OLS, rolling OLS, Kalman filtering, and enhanced Kalman approaches across five sectors. The goal was to understand which methods work best under different market conditions and identify the most promising pair relationships.

Main findings:
- Rolling methodologies significantly outperform static approaches
- Sector selection matters more than initially expected
- Most pairs show structural breaks that static models can't handle
- REITs and utilities show the most stable relationships

## Results Summary

After testing 200+ pairs across 9 sectors, I focused on 5 pairs that showed strong cointegration relationships. All selected pairs have p-values below 0.01 and reasonable half-lives for intraday trading.

### Selected Pairs

| Sector | Pair | Cointegration p-value | Half-life (Days) | Static Beta | Notes |
|--------|------|----------------------|------------------|-------------|-------|
| Utilities | AEP-WEC | 0.0092 | 5.46 | 0.786 | Stable regulated utilities |
| Materials | CLF-SCCO | 0.0052 | 6.18 | -0.266 | Mining companies, commodity exposure |
| REITs | MAA-CPT | 0.0087 | 4.08 | 1.173 | Residential REITs, similar geography |
| Financials | MET-TRV | 0.0017 | 5.07 | 0.669 | Large cap insurers |
| Energy | VLO-PBF | 0.0009 | 3.66 | 2.334 | Independent refiners |

The energy pair (VLO-PBF) shows the strongest cointegration and fastest mean reversion, while the financials pair (MET-TRV) has the most significant cointegration test result.

## Key Findings

### Methodological Evolution: From Static to Enhanced Kalman

I tested four different approaches in sequence, each solving problems from the previous method. Here's what I found using AEP-WEC as an example:

| Method | Max |Z-Score| | % within ±2σ | % within ±1σ | Key Issues |
|--------|--------------|-------------|-------------|------------|
| **Static OLS** | 14.66 | 40.2% | 24.0% | Severe drift, non-stationary |
| **Rolling OLS** | 3.94 | 78.9% | 44.6% | Much better, but still jumpy |
| **Kalman Filter** | 17.24 | 95.5% | 78.6% | Excellent fit, but extreme outliers |
| **Enhanced Kalman** | 5.11 | 92.7% | 66.9% | Best balance of all methods |

#### Static OLS Problems
The static approach was a disaster. When I calibrated spreads using 2021 parameters and applied them through 2024, the z-scores exploded:
- AEP-WEC: Z-scores reached 14.7 standard deviations (should stay within ±3)
- MET-TRV: Even worse, hitting 24.4 standard deviations  
- Only 40% of observations stayed within ±2σ
- This makes any trading signals completely unreliable

#### Rolling OLS Improvement
The rolling approach (90-day estimation window, 45-day z-score window) fixed the major issues:
- Maximum z-scores dropped to reasonable levels (under 4.0)
- ~79% of observations now within ±2σ (nearly doubled!)
- Much more stable trading signals
- But still had some jumpiness when parameters updated

#### Kalman Filter Breakthrough  
The basic Kalman filter was a revelation:
- 95.5% of observations within ±2σ (excellent stationarity)
- Smooth, adaptive parameter updates
- Much better at tracking regime changes
- But occasionally had extreme outliers (17+ standard deviations) during market stress

#### Enhanced Kalman Optimization
The enhanced version added regime detection and outlier control:
- Reduced maximum z-score from 17.2 to 5.1 
- Maintained 92.7% within ±2σ (slight trade-off but much safer)
- Added delta parameter for volatility clustering
- Best overall balance between fit quality and robustness

### Beta Stability Analysis

One interesting finding was how much hedge ratios (betas) change over time:

| Pair | Static Beta | Rolling Beta Mean | Rolling Beta Std | Coefficient of Variation |
|------|-------------|-------------------|------------------|-------------------------|
| AEP-WEC | 0.786 | 0.869 | 0.265 | 30.5% |
| CLF-SCCO | -0.266 | 0.172 | 0.209 | 121.6% |
| MAA-CPT | 1.173 | 1.170 | 0.172 | 14.7% |
| MET-TRV | 0.669 | 0.282 | 0.271 | 96.1% |
| VLO-PBF | 2.334 | 2.046 | 0.507 | 24.8% |

MAA-CPT (REITs) shows the most stable relationship with only 14.7% variation in beta. CLF-SCCO (materials) is extremely unstable - the beta even changes sign from negative to positive.

## Sector Insights

**Utilities (AEP-WEC):** Pretty stable relationship, which makes sense for regulated utilities with similar business models. The beta drift is moderate (30% CV) but manageable.

**Materials (CLF-SCCO):** This was surprising - despite strong cointegration, the hedge ratio is all over the place. CLF is steel, SCCO is copper, so they're both commodity plays but apparently react differently to market conditions. Might be too risky for systematic trading.

**REITs (MAA-CPT):** Best performing pair by far. Both are residential REITs with similar geographic exposure. The relationship is very stable (14.7% CV) and mean-reverts quickly (4 days). This would be my top pick for actual trading.

**Financials (MET-TRV):** Strong cointegration but unstable hedge ratios. Both are large insurers, but they seem to have different sensitivities to interest rates or regulatory changes. The beta even goes negative sometimes.

**Energy (VLO-PBF):** Strongest statistical relationship and fastest mean reversion, but moderate beta instability. Both are independent refiners, so they should move together, but the hedge ratio varies quite a bit (25% CV).

## Portfolio-Based Extension: Beyond Pairs Trading

After completing the pairs analysis, I extended the research to portfolio-based statistical arbitrage - trading baskets of 3-5 stocks within sectors rather than just pairs. This is more challenging but potentially more profitable since you can capture multiple cointegrating relationships simultaneously.

### Portfolio Selection Methodology

The portfolio selection process was quite involved. I couldn't just pick random combinations - I needed a systematic way to find groups of stocks that move together long-term but have temporary mispricings.

**Step 1: Graph-Based Screening**
I treated each sector as a network where stocks are nodes and cointegrating relationships are edges:
- Started with all possible pairs within each sector
- Ran Johansen tests on every pair (p < 0.05 threshold)
- Built connectivity graphs showing which stocks are cointegrated with which others
- Looked for dense clusters - groups where most stocks are cointegrated with most other stocks

**Step 2: Connectivity Scoring**
For each potential portfolio, I calculated a connectivity score:
```
Connectivity = (# of cointegrated pairs) / (# of possible pairs)
```
- 4-stock portfolio: max 6 pairs, so connectivity = actual_pairs/6
- Only considered portfolios with connectivity > 60%
- This ensures most stocks in the portfolio move together

**Step 3: Diversification Within Sectors**
I also tracked the business model diversity within portfolios using sub-sector classifications:
- Energy: oil_exploration, refiners, natural_gas_midstream, alternative_energy, oil_services
- Financials: banks, insurers, asset_managers, payments
- Materials: mining, chemicals, construction, packaging
- This prevents over-concentration in one business type

### Johansen Cointegration Testing

For portfolios (vs pairs), I used the Johansen test which can detect multiple cointegrating relationships:
- **Null hypothesis:** No cointegration among the N stocks
- **Alternative:** At least one cointegrating relationship exists
- The test returns the number of cointegrating vectors (rank)
- Higher rank = more stable long-term relationships

### Portfolio Results Analysis

I found some interesting portfolios across sectors:

| Sector | Portfolio | Assets | Connectivity | Coint. Rank | Composition |
|--------|-----------|--------|-------------|-------------|-------------|
| Energy | ENE_001 | CTRA, NEE, NOV, PSX | 83% | 2 | Oil exploration, alt energy, services, refining |
| Financials | FIN_001 | JPM, MET, PGR, TRV | 83% | 1 | Bank + 3 insurers |
| Materials | MAT_001 | AMCR, CCK, CRH, X | 100% | 1 | Packaging + steel |
| REITs | REI_001 | BRX, IRM, SKT, WPC | 100% | 4 | Diversified REIT types |
| Health Care | HEA_001 | GILD, HCA, LH, SYK | 100% | 4 | Pharma, hospitals, devices |

### VECM Implementation and Spread Construction

Once I identified cointegrated portfolios, I used Vector Error Correction Models (VECM) to construct tradeable spreads:

1. **Estimate the cointegrating vector** using Johansen's method
2. **Calculate the error correction term** (the "spread")
3. **Model the short-term dynamics** of how stocks adjust back to equilibrium
4. **Generate z-scores** for trading signals

The spread for a 4-stock portfolio looks like:
```
Spread = w1*P1 + w2*P2 + w3*P3 + w4*P4
```
Where the weights (w) come from the first eigenvector of the Johansen test.

### Portfolio Spread Performance

The portfolio spreads generally showed better stationarity than pairs:

| Portfolio | Max |Z-Score| | % within ±2σ | Half-life (Days) | Notes |
|-----------|--------------|-------------|------------------|--------|
| ENE_001 | 2.21 | 63.1% | 1.40 | Very fast mean reversion |
| FIN_001 | 2.95 | 86.7% | 6.81 | Most stable portfolio |
| MAT_001 | 5.25 | 82.7% | 2.78 | Good balance |
| REI_001 | 4.01 | 83.4% | 3.10 | Strong relationships |
| HEA_001 | 5.39 | 85.0% | 3.83 | Healthcare complexity |

The portfolio approach showed some clear advantages:
- **Better diversification:** Less susceptible to individual stock news
- **Multiple relationships:** Can profit from several mispricings simultaneously  
- **Sector-specific insights:** Captures industry dynamics better than cross-sector pairs

However, the complexity is much higher - you need to track 4-5 positions instead of 2, and the transaction costs scale up significantly.

## Implementation Details

### Data
- 15-minute bars from 2021-2024 (about 26,000 observations per pair)
- Standard preprocessing: adjusted for splits and dividends
- Filtered for liquidity (minimum $50M daily volume)

### Methods
The code implements several approaches:

1. **Johansen Cointegration Test** - Standard implementation to identify cointegrated pairs
2. **Static OLS** - Traditional approach using fixed parameters from calibration period  
3. **Rolling OLS** - Adaptive approach with 90-day estimation window
4. **Kalman Filter** - State-space model for dynamic hedge ratios (partially implemented)

### Key Parameters
After testing different settings, I settled on:
- OLS estimation window: 90 days (seems optimal for capturing regime changes without too much noise)
- Z-score calculation window: 45 days (faster adaptation to changing volatility)
- Entry threshold: ±2.0 standard deviations
- Exit threshold: Zero crossing
- Stop loss: ±3.0 standard deviations

## File Structure

```
results/
├── pairs_based/
│   ├── cointegration_test_result/     # Johansen test outputs
│   └── spread_visualization/
│       ├── static_spread/             # Static OLS analysis  
│       ├── rolling_vs_static/         # Rolling OLS comparison
│       ├── kalman_analysis/           # Kalman filter results
│       └── enhanced_kalman/           # Advanced Kalman results
├── portfolio_based/
│   ├── portfolio_candidates/          # Selected portfolios by sector
│   └── static_johansen/              # Portfolio VECM analysis
core/                                  # Main analysis modules
config/                               # Sector definitions
```

Key scripts:
- `visualize_static_spread.py` - Static OLS analysis
- `visualize_rolling_spread.py` - Rolling OLS analysis  
- `find_best_pairs.py` - Pair selection process
- `run_portfolio_selection.py` - Portfolio construction
- `analyze_portfolio_candidates.py` - Portfolio evaluation

## Next Steps

Several areas I'd like to explore further:

1. **Dynamic Portfolio Rebalancing** - Currently portfolios are static, but member stocks could change based on evolving relationships
2. **Multi-Timeframe Analysis** - Test whether portfolio relationships hold across different frequencies (daily, hourly, etc.)
3. **Transaction Cost Integration** - Portfolio trading involves more positions, so costs become critical
4. **Risk Parity Weighting** - Instead of using Johansen weights, try risk-balanced portfolio construction
5. **Cross-Sector Portfolios** - Test whether portfolios spanning multiple sectors can work
6. **Machine Learning Integration** - Use ML to predict when cointegrating relationships might break down
7. **Live Implementation** - Build real-time system for portfolio monitoring and execution

## References

The implementation draws from several academic papers:
- Engle & Granger (1987) on cointegration
- Gatev, Goetzmann & Rouwenhorst (2006) on pairs trading performance  
- Avellaneda & Lee (2010) on statistical arbitrage
- Various papers on Kalman filtering in finance

## Notes

This is research code, not production-ready trading system. There are no transaction costs, market impact, or proper risk controls. The results look promising but would need significant additional work before real money deployment.

The data quality is also limited - using free/academic sources rather than institutional feeds. Real implementation would need higher frequency data and better execution infrastructure.
