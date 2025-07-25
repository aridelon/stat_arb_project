# Statistical Arbitrage Trading System
## Advanced Pairs Trading and Portfolio Analysis Framework

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat-square&logo=github)](https://github.com/aridelon/stat_arb_project)
[![Python](https://img.shields.io/badge/Python-3.8+-green?style=flat-square&logo=python)](https://python.org)
[![Data Science](https://img.shields.io/badge/Focus-Quantitative%20Finance-orange?style=flat-square)](https://github.com/aridelon/stat_arb_project)

---

## 🎯 **Executive Summary**

This repository contains a comprehensive **statistical arbitrage trading system** designed for institutional sell-side trading desks. The framework implements sophisticated pairs trading strategies using multiple methodologies including **Static OLS**, **Rolling OLS**, **Kalman Filtering**, and **Enhanced Kalman approaches**. 

**Key Value Proposition:**
- Systematic identification of cointegrated equity pairs across 9 sectors
- Multi-method spread modeling with robust statistical validation  
- Comprehensive performance analysis and risk metrics
- Production-ready code structure for institutional deployment

---

## 📊 **Core Results & Analysis**

### **Portfolio Performance Overview**

The system has been tested on **5 high-quality pairs** selected from rigorous cointegration analysis across **200+ equity pairs**:

| **Sector** | **Pair** | **Cointegration p-value** | **Half-life (Days)** | **Static β** | **Status** |
|------------|----------|---------------------------|---------------------|--------------|------------|
| **Utilities** | AEP-WEC | 0.0092 ✅ | 5.46 | 0.786 | **Excellent** |
| **Materials** | CLF-SCCO | 0.0052 ✅ | 6.18 | -0.266 | **Excellent** |
| **REITs** | MAA-CPT | 0.0087 ✅ | 4.08 | 1.173 | **Excellent** |
| **Financials** | MET-TRV | 0.0017 ✅ | 5.07 | 0.669 | **Outstanding** |
| **Energy** | VLO-PBF | 0.0009 ✅ | 3.66 | 2.334 | **Outstanding** |

*All pairs show strong cointegration (p < 0.01) with optimal mean-reversion speeds (3-7 days)*

---

## 🔬 **Detailed Methodology Comparison**

### **1. Static OLS vs Rolling OLS Analysis**

Our analysis reveals critical insights about spread stationarity and the superiority of adaptive methods:

#### **Static OLS Performance Issues**
```
⚠️  CRITICAL FINDING: Static OLS exhibits severe non-stationarity
```

| **Pair** | **Static Z-Score Range** | **% Within 2σ** | **Max Deviation** | **Assessment** |
|----------|--------------------------|----------------|-------------------|----------------|
| AEP-WEC | -3.5 to **14.7** | 40.2% | **14.7σ** | ❌ Non-stationary |
| CLF-SCCO | -6.9 to 6.9 | 66.5% | 6.9σ | ⚠️ Marginal |
| MAA-CPT | -7.7 to 7.7 | 72.7% | 7.7σ | ⚠️ Marginal |
| MET-TRV | -24.4 to **24.4** | 36.0% | **24.4σ** | ❌ Non-stationary |
| VLO-PBF | -14.9 to **14.9** | 41.2% | **14.9σ** | ❌ Non-stationary |

**Key Insight:** Static parameters calibrated on 2021 data fail to capture regime changes, leading to extreme z-score deviations and poor trading signals.

#### **Rolling OLS Superior Performance**
```
✅ SOLUTION: Rolling OLS demonstrates robust stationarity
```

| **Pair** | **Rolling Z-Score Range** | **% Within 2σ** | **Improvement** | **Assessment** |
|----------|---------------------------|----------------|-----------------|----------------|
| AEP-WEC | -3.9 to 3.9 | **78.9%** | +38.7pp | ✅ **Excellent** |
| CLF-SCCO | -4.4 to 4.4 | **79.3%** | +12.8pp | ✅ **Excellent** |
| MAA-CPT | -4.8 to 4.8 | **78.7%** | +6.0pp | ✅ **Excellent** |
| MET-TRV | -4.0 to 4.0 | **81.9%** | +45.9pp | ✅ **Outstanding** |
| VLO-PBF | -4.9 to 4.9 | **82.6%** | +41.4pp | ✅ **Outstanding** |

**Trading Implication:** Rolling OLS maintains spread stationarity with ~80% of observations within 2σ, enabling reliable statistical arbitrage signals.

---

### **2. Beta Coefficient Analysis**

#### **Static vs Dynamic Beta Evolution**

The analysis reveals significant beta instability over the 4-year period:

| **Pair** | **Static β** | **Rolling β (Mean±Std)** | **β Range** | **Coefficient of Variation** |
|----------|--------------|---------------------------|-------------|----------------------------|
| AEP-WEC | 0.786 | 0.869 ± 0.265 | [-0.03, 1.59] | **30.5%** |
| CLF-SCCO | -0.266 | 0.172 ± 0.209 | [-0.20, 0.68] | **121.6%** |
| MAA-CPT | 1.173 | 1.170 ± 0.172 | [0.68, 1.64] | **14.7%** |
| MET-TRV | 0.669 | 0.282 ± 0.271 | [-0.59, 0.81] | **96.1%** |
| VLO-PBF | 2.334 | 2.046 ± 0.507 | [0.82, 2.88] | **24.8%** |

**Key Insights:**
- **MAA-CPT** shows the most stable relationship (CV = 14.7%)
- **CLF-SCCO** exhibits extreme beta instability (CV = 121.6%)
- **Energy pairs** demonstrate moderate but significant beta drift
- **Financial pairs** show structural breaks in hedge ratios

---

## 🏛️ **Sector-Specific Performance Analysis**

### **Utilities Sector (AEP-WEC)**
```
🔵 SECTOR: Electric Utilities
📈 PERFORMANCE: Strong cointegration with moderate beta stability
⚡ CHARACTERISTICS: Regulated utilities with similar business models
```
- **Cointegration Strength:** Excellent (p = 0.0092)
- **Mean Reversion Speed:** Moderate (5.46 days)
- **Beta Stability:** Good (CV = 30.5%)
- **Trading Suitability:** ✅ **Recommended** for stat-arb strategies

### **Materials Sector (CLF-SCCO)**
```
🟤 SECTOR: Steel & Copper Mining
📈 PERFORMANCE: Excellent cointegration but high beta volatility
🏭 CHARACTERISTICS: Commodity-exposed industrials with cyclical nature
```
- **Cointegration Strength:** Excellent (p = 0.0052)
- **Mean Reversion Speed:** Moderate (6.18 days)
- **Beta Stability:** Poor (CV = 121.6%)
- **Trading Suitability:** ⚠️ **Caution** - requires sophisticated hedging

### **REITs Sector (MAA-CPT)**
```
🏢 SECTOR: Residential REITs
📈 PERFORMANCE: Outstanding stability and cointegration
🏠 CHARACTERISTICS: Apartment-focused REITs with similar geographic exposure
```
- **Cointegration Strength:** Excellent (p = 0.0087)
- **Mean Reversion Speed:** Fast (4.08 days)
- **Beta Stability:** Excellent (CV = 14.7%)
- **Trading Suitability:** ✅ **Highest Recommended** - ideal for stat-arb

### **Financials Sector (MET-TRV)**
```
🏦 SECTOR: Property & Casualty Insurance
📈 PERFORMANCE: Strongest cointegration but structural breaks
💼 CHARACTERISTICS: Large-cap insurers with similar business models
```
- **Cointegration Strength:** Outstanding (p = 0.0017)
- **Mean Reversion Speed:** Moderate (5.07 days)
- **Beta Stability:** Poor (CV = 96.1%)
- **Trading Suitability:** ⚠️ **Advanced** - requires regime-aware models

### **Energy Sector (VLO-PBF)**
```
⚫ SECTOR: Oil Refining
📈 PERFORMANCE: Strongest cointegration with fastest mean reversion
🛢️ CHARACTERISTICS: Independent refiners with crack spread exposure
```
- **Cointegration Strength:** Outstanding (p = 0.0009)
- **Mean Reversion Speed:** Very Fast (3.66 days)
- **Beta Stability:** Moderate (CV = 24.8%)
- **Trading Suitability:** ✅ **Recommended** - high-frequency opportunities

---

## 🛠️ **Technical Implementation**

### **Data Infrastructure**
- **Frequency:** 15-minute intraday bars
- **Period:** 4 years (2021-2024)
- **Coverage:** 26,000+ observations per pair
- **Quality:** Professional-grade data with robust preprocessing

### **Statistical Methods**

#### **1. Johansen Cointegration Test**
```python
# Null Hypothesis: No cointegration
# Alternative: At least one cointegrating relationship
# Confidence Level: 99% (α = 0.01)
```

#### **2. Rolling OLS Implementation**
```python
# Parameters:
# - OLS Window: 90 days (optimal for beta estimation)
# - Z-Score Window: 45 days (optimal for mean reversion)
# - Minimum Observations: 50 (data quality threshold)
```

#### **3. Half-Life Calculation**
```python
# Ornstein-Uhlenbeck Process: dX = -λ(X - μ)dt + σdW
# Half-Life = ln(2) / λ
# Optimal Range: 1-10 days for intraday trading
```

---

## 📈 **Trading Signal Framework**

### **Entry/Exit Logic**
```
📍 ENTRY SIGNALS:
- Z-Score ≤ -2.0: Enter LONG spread position
- Z-Score ≥ +2.0: Enter SHORT spread position

🎯 EXIT SIGNALS:
- Z-Score crosses zero: Take profit
- Z-Score ≥ ±3.0: Stop loss (risk management)

⏱️ POSITION MANAGEMENT:
- Maximum holding period: 20 days
- No overnight gaps during earnings
- Liquidity filters: Min $50M daily volume
```

### **Risk Management**
- **Position Sizing:** Kelly Criterion with 25% of optimal
- **Portfolio Heat:** Maximum 5% capital per pair
- **Sector Limits:** Maximum 3 pairs per sector
- **Correlation Monitoring:** Cross-pair correlation ≤ 0.3

---

## 📁 **Repository Structure**

```
stat_arb_project/
├── 📊 results/
│   ├── pairs_based/
│   │   ├── cointegration_test_result/     # Johansen test outputs
│   │   └── spread_visualization/
│   │       ├── static_spread/             # Static OLS results
│   │       ├── rolling_vs_static/         # Rolling OLS comparison
│   │       ├── kalman_analysis/           # Kalman filter results
│   │       └── enhanced_kalman/           # Advanced Kalman results
│   └── portfolio_based/
│       ├── portfolio_candidates/          # Selected pair portfolios
│       └── static_johansen/              # Portfolio cointegration
├── 🧠 core/
│   ├── data_loader.py                    # Data ingestion & preprocessing
│   ├── johansen_analyzer.py              # Cointegration testing
│   ├── kalman_filter.py                  # Basic Kalman implementation
│   ├── kalman_filter_enhanced.py         # Advanced Kalman with regime detection
│   ├── pair_analyzer.py                  # Pairs trading logic
│   ├── portfolio_selector.py             # Portfolio optimization
│   └── rolling_ols_zscore.py            # Rolling regression engine
├── ⚙️ config/
│   ├── utilities.py                      # Utility sector tickers
│   ├── financials.py                     # Financial sector tickers
│   ├── materials.py                      # Materials sector tickers
│   ├── reits.py                         # REIT sector tickers
│   └── energy.py                        # Energy sector tickers
└── 🔬 analysis/
    ├── visualize_static_spread.py        # Static OLS visualization
    ├── visualize_rolling_spread.py       # Rolling OLS visualization
    ├── find_best_pairs.py               # Pair selection optimization
    └── run_sector_analysis.py           # Sector-wide analysis
```

---

## 🎓 **Academic & Professional Rigor**

### **Publications & References**
This implementation draws from seminal academic literature:

1. **Engle & Granger (1987)** - Cointegration and Error Correction
2. **Johansen (1991)** - Estimation and Hypothesis Testing of Cointegration Vectors
3. **Gatev, Goetzmann & Rouwenhorst (2006)** - Pairs Trading: Performance of a Relative-Value Arbitrage Rule
4. **Avellaneda & Lee (2010)** - Statistical Arbitrage in the U.S. Equities Market
5. **Kalman (1960)** - A New Approach to Linear Filtering and Prediction Problems

### **Risk Disclosures**
```
⚠️  RISK WARNINGS:
• Past performance does not guarantee future results
• Cointegration relationships can break down during market stress
• Model assumes normal market conditions and adequate liquidity
• Regulatory changes may impact pair relationships
• Transaction costs and market impact not included in analysis
```

---

## 🚀 **Production Deployment Considerations**

### **Infrastructure Requirements**
- **Low-latency data feed** (Thomson Reuters, Bloomberg)
- **Prime brokerage** with DMA capabilities
- **Risk management system** with real-time P&L monitoring
- **Execution algorithms** with TWAP/VWAP implementation

### **Compliance & Risk Controls**
- **Market risk limits** aligned with VaR models
- **Operational risk** controls for model validation
- **Regulatory reporting** (MiFID II, Dodd-Frank compliance)
- **Audit trail** for all trading decisions

---

## 📞 **Contact & Professional Background**

**Developed by:** [Your Name]  
**LinkedIn:** [Your LinkedIn Profile]  
**Email:** [Your Professional Email]  

```
🎯 CAREER OBJECTIVE:
Quantitative Analyst / Trader role at top-tier sell-side institution
Specializing in systematic trading strategies and risk management

💼 CORE COMPETENCIES:
• Statistical Arbitrage & Pairs Trading
• Quantitative Risk Management  
• Python/R Programming for Finance
• Machine Learning in Trading
• Portfolio Optimization
```

---

## 📜 **License & Usage**

This project is developed for **educational and professional demonstration purposes**. 

```
⚖️  USAGE TERMS:
• Academic research: ✅ Permitted
• Professional interviews: ✅ Permitted  
• Commercial deployment: ❌ Requires explicit permission
• Code modification: ✅ Permitted with attribution
```

---

*"In markets, the only certainty is uncertainty. Statistical arbitrage seeks to profit from the temporary dislocations while acknowledging the fundamental unpredictability of financial markets."*

**Last Updated:** July 2025 | **Version:** 2.0 | **Status:** Production Ready
