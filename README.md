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

### Static vs Rolling OLS Comparison

The most important finding is that static OLS models perform poorly on this data. When I calibrated spreads using 2021 parameters and applied them through 2024, the z-scores became severely non-stationary.

**Static OLS Problems:**
- AEP-WEC: Z-scores reached 14.7 standard deviations (should stay within ±3)
- MET-TRV: Even worse, hitting 24.4 standard deviations  
- Most pairs had less than 50% of observations within 2 standard deviations
- This makes trading signals unreliable

**Rolling OLS Results:**
The rolling approach (90-day estimation window, 45-day z-score window) fixed most of these issues:
- All pairs now keep ~80% of observations within 2 standard deviations
- Maximum z-scores stay reasonable (under 5.0)
- Much more stable trading signals

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
│       └── enhanced_kalman/           # Advanced Kalman (in progress)
├── portfolio_based/                   # Portfolio-level analysis
core/                                  # Main analysis modules
config/                               # Sector definitions
```

The main analysis scripts are in the root directory:
- `visualize_static_spread.py` - Generate static OLS results
- `visualize_rolling_spread.py` - Rolling OLS analysis
- `find_best_pairs.py` - Pair selection process
- Various sector analysis scripts

## Next Steps

There are several areas I'd like to explore further:

1. **Kalman Filter Implementation** - The basic version is working, but I want to add regime switching capabilities
2. **Transaction Cost Analysis** - Current results don't include bid-ask spreads or market impact
3. **Portfolio Construction** - Instead of trading pairs individually, optimize across multiple pairs
4. **Risk Management** - Add proper position sizing and correlation monitoring
5. **Live Trading** - Test the strategies on more recent data and smaller timeframes

## References

The implementation draws from several academic papers:
- Engle & Granger (1987) on cointegration
- Gatev, Goetzmann & Rouwenhorst (2006) on pairs trading performance  
- Avellaneda & Lee (2010) on statistical arbitrage
- Various papers on Kalman filtering in finance

## Notes

This is research code, not production-ready trading system. There are no transaction costs, market impact, or proper risk controls. The results look promising but would need significant additional work before real money deployment.

The data quality is also limited - using free/academic sources rather than institutional feeds. Real implementation would need higher frequency data and better execution infrastructure.
