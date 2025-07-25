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

| Method | Max Z-Score | % within ±2σ | % within ±1σ | Key Issues |
|--------|-------------|-------------|-------------|------------|
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

| Portfolio | Max Z-Score | % within ±2σ | Half-life (Days) | Notes |
|-----------|-------------|-------------|------------------|-------|
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

### Data Infrastructure
- **Frequency:** 15-minute intraday bars (2021-2024)
- **Volume:** ~26,000 observations per pair over 4 years
- **Preprocessing:** Corporate actions adjustment, dividend reinvestment
- **Quality filters:** Minimum $50M daily volume, no penny stocks
- **Missing data:** Forward-fill method with maximum 3-period gaps

### Statistical Methods: Mathematical Foundations

#### 1. Johansen Cointegration Test

The Johansen test identifies long-term equilibrium relationships between non-stationary time series. For two price series P₁(t) and P₂(t):

**Vector Error Correction Model (VECM):**
```
ΔP₁(t) = α₁[β₁P₁(t-1) + β₂P₂(t-1)] + Σγ₁ᵢΔP₁(t-i) + Σδ₁ᵢΔP₂(t-i) + ε₁(t)
ΔP₂(t) = α₂[β₁P₁(t-1) + β₂P₂(t-1)] + Σγ₂ᵢΔP₁(t-i) + Σδ₂ᵢΔP₂(t-i) + ε₂(t)
```

**Key components:**
- **Cointegrating vector [β₁, β₂]:** Long-term relationship weights
- **Error correction terms α₁, α₂:** Speed of adjustment back to equilibrium
- **Lag structure:** Short-term dynamics (typically 1-5 lags)

**Test procedure:**
1. **Trace test:** H₀: rank ≤ r vs H₁: rank > r
2. **Maximum eigenvalue test:** H₀: rank = r vs H₁: rank = r+1
3. **Critical values:** Mackinnon-Haug-Michelis (1999) critical values

**Implementation details:**
- Lag selection via AIC/BIC information criteria
- Deterministic trend specification (constant vs trend)
- 5% significance level for cointegration acceptance

#### 2. Static OLS Approach

Traditional pairs trading using fixed parameters from calibration period.

**Model specification:**
```
P₁(t) = α + β·P₂(t) + ε(t)
```

**Spread construction:**
```
S(t) = P₁(t) - β̂·P₂(t) - α̂
```

**Z-score normalization:**
```
Z(t) = [S(t) - μₛ] / σₛ
```
Where μₛ and σₛ are calibrated on 2021 data.

**Critical limitations:**
- Parameter estimates β̂ fixed over entire period
- Assumes stable relationship (violated during regime changes)
- No adaptation to changing volatility or correlation structure

#### 3. Rolling OLS Implementation

Adaptive approach with sliding window parameter estimation.

**Rolling regression:**
```
β̂(t) = [Σᵢ₌₍ₜ₋ᵩ₊₁₎ᵗ P₂(i)²]⁻¹ · [Σᵢ₌₍ₜ₋ᵩ₊₁₎ᵗ P₁(i)·P₂(i)]
```
Where W = 90 days (estimation window)

**Adaptive spread:**
```
S(t) = P₁(t) - β̂(t)·P₂(t)
```

**Rolling z-score:**
```
Z(t) = [S(t) - μₛ(t)] / σₛ(t)
```
Where μₛ(t) and σₛ(t) computed over 45-day rolling window.

**Advantages:**
- Adapts to regime changes and structural breaks
- Captures time-varying hedge ratios
- Better handling of volatility clustering

#### 4. Kalman Filter State-Space Model

Dynamic hedge ratio estimation using state-space framework.

**State equation (unobserved):**
```
β(t) = β(t-1) + w(t),  w(t) ~ N(0, Q)
```

**Observation equation:**
```
P₁(t) = β(t)·P₂(t) + v(t),  v(t) ~ N(0, R)
```

**Kalman filter recursions:**

*Prediction step:*
```
β̂(t|t-1) = β̂(t-1|t-1)
P(t|t-1) = P(t-1|t-1) + Q
```

*Update step:*
```
K(t) = P(t|t-1)·P₂(t) / [P₂(t)²·P(t|t-1) + R]
β̂(t|t) = β̂(t|t-1) + K(t)·[P₁(t) - β̂(t|t-1)·P₂(t)]
P(t|t) = [1 - K(t)·P₂(t)]·P(t|t-1)
```

**Parameter estimation:**
- **Process noise Q:** Controls hedge ratio adaptability (Q = 1e-4)
- **Observation noise R:** Measurement error variance (estimated via EM)
- **Initial conditions:** β̂(0) from first 30 observations

#### 5. Enhanced Kalman Filter

Extension with regime detection and outlier robustness.

**State space model:**
```
μ(t) = μ(t-1) + η₁(t)     [mean level]
γ(t) = γ(t-1) + η₂(t)     [hedge ratio]
δ(t) = ρδ(t-1) + η₃(t)    [volatility factor]
```

**Observation equation:**
```
S(t) = μ(t) + γ(t)·[P₂(t) - P₂(t-1)] + δ(t)·ε(t)
```

**Regime detection:**
- **Innovation sequence:** ν(t) = S(t) - Ŝ(t|t-1)
- **Outlier threshold:** |ν(t)| > 3σᵥ triggers robustification
- **Discount factor:** Reduces Kalman gain during outliers

**Enhanced features:**
- **Volatility clustering:** δ(t) captures time-varying spread volatility
- **Robust updating:** Down-weights observations during market stress
- **Adaptive learning:** Process noise adapts to market regime

#### 6. Half-Life Estimation

Mean reversion speed calculation using Ornstein-Uhlenbeck framework.

**OU process:**
```
dS(t) = -λ[S(t) - μ]dt + σdW(t)
```

**Discrete approximation:**
```
S(t) - S(t-1) = -λ[S(t-1) - μ] + ε(t)
```

**Half-life formula:**
```
τ₁/₂ = ln(2) / λ
```

**Estimation via AR(1):**
```
ΔS(t) = α + β·S(t-1) + ε(t)
λ̂ = -β̂,  τ̂₁/₂ = ln(2) / (-β̂)
```

#### 7. Portfolio Extension: Multivariate Cointegration

**Johansen test for n > 2 assets:**

**VAR(p) representation:**
```
ΔX(t) = Π·X(t-1) + Σᵢ₌₁ᵖ⁻¹ Γᵢ·ΔX(t-i) + ε(t)
```

**Granger representation:**
```
Π = αβ'
```
Where α = adjustment coefficients, β = cointegrating vectors

**VECM for portfolio:**
```
ΔPᵢ(t) = αᵢ·[β'·P(t-1)] + Σⱼ γᵢⱼ·ΔPⱼ(t-1) + εᵢ(t)
```

**Portfolio spread construction:**
```
S(t) = β'·P(t) = β₁P₁(t) + β₂P₂(t) + ... + βₙPₙ(t)
```

**Connectivity scoring:**
```
C = |{(i,j): p-value(i,j) < 0.05}| / (n choose 2)
```

### Key Parameters
After extensive backtesting, I optimized the following parameters:

**Rolling OLS specifications:**
- **Estimation window:** 90 days (optimal balance: responsiveness vs noise)
- **Z-score window:** 45 days (faster volatility adaptation)
- **Minimum observations:** 50 (data quality threshold)

**Trading signal thresholds:**
- **Entry threshold:** ±2.0σ (based on historical false positive analysis)
- **Exit threshold:** Zero crossing (profit-taking at mean reversion)
- **Stop loss:** ±3.0σ (tail risk protection)

**Kalman filter parameters:**
- **Process noise Q:** 1e-4 (moderate adaptation speed)
- **Observation noise R:** Estimated via EM algorithm
- **Outlier threshold:** 3σ (robust estimation during market stress)

### Performance Metrics and Statistical Tests

#### Stationarity Assessment
**Augmented Dickey-Fuller Test:**
```
H₀: Z(t) has unit root (non-stationary)
H₁: Z(t) is stationary
```
- Critical value: -2.86 (5% level)
- All enhanced methods achieve stationarity (p < 0.01)

**Ljung-Box Test for Autocorrelation:**
```
H₀: No serial correlation in residuals
LB = n(n+2)·Σₖ₌₁ʰ ρ̂ₖ²/(n-k)
```
- Target: p-value > 0.05 (white noise residuals)
- Rolling/Kalman methods achieve LB test acceptance

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
