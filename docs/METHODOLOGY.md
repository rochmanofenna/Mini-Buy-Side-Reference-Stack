# Methodology

## Walk-Forward Protocol

Our backtesting methodology employs a strict walk-forward validation approach to prevent overfitting and ensure realistic performance estimates.

### Time Windows
- **Training window**: 252 trading days (1 year)
- **Validation window**: 63 trading days (3 months)
- **Step size**: 21 trading days (1 month)
- **Total periods**: 12 walk-forward iterations

### Data Splits
1. **In-sample (IS)**: 70% for parameter optimization
2. **Out-of-sample (OOS)**: 30% for validation
3. **No lookahead**: Strict temporal ordering enforced

## Transaction Cost Model

### Components
- **Spread cost**: Half-spread crossing (bid-ask)
- **Market impact**: Square-root model based on participation rate
- **Commissions**: $0.005 per share or 0.5 bps (whichever is greater)
- **Borrow costs**: 50 bps annualized for shorts
- **Slippage**: Linear + non-linear components based on volatility

### Formula
```
TC = spread/2 + α*√(volume/ADV) + commission + borrow_rate*(holding_period/365)
```

Where:
- α = impact coefficient (calibrated to 10 bps for 1% ADV)
- ADV = Average Daily Volume (20-day)

## Risk Management

### Position Limits
- **Single name**: Max 5% of portfolio NAV
- **Sector exposure**: Max 30% gross
- **Gross exposure**: Max 200% (1x leverage)
- **Net exposure**: [-40%, +40%] band

### Risk Metrics
- **VaR (95%)**: Daily limit 2% of NAV
- **Expected Shortfall**: Monitored but not binding
- **Correlation limits**: Max 0.6 between any two positions

## Signal Generation

### Feature Engineering
- **Price-based**: Returns, volatility, volume patterns
- **Microstructure**: Order flow imbalance, quote dynamics
- **Alternative data**: Sentiment scores (when available)

### Model Selection
- **Base models**: GBM, Random Forest, Linear
- **Ensemble**: Weighted average with dynamic rebalancing
- **Retraining**: Monthly with expanding window

## Performance Attribution

### Decomposition
1. **Alpha**: Returns above risk-free rate and market beta
2. **Factor exposure**: Size, value, momentum contributions
3. **Idiosyncratic**: Residual after factor adjustment
4. **Transaction costs**: Actual vs estimated

### Metrics Reported
- **Sharpe Ratio**: Annualized, excess returns
- **Information Ratio**: Alpha / tracking error
- **Maximum Drawdown**: Peak-to-trough
- **Calmar Ratio**: Annual return / MaxDD
- **Win Rate**: Profitable trades / total trades
- **Profit Factor**: Gross profit / gross loss

## Data Quality Checks

### Validation
- **Survivorship bias**: Include delisted securities
- **Corporate actions**: Adjust for splits/dividends
- **Outlier detection**: Winsorize at 3 std dev
- **Missing data**: Forward-fill prices (max 5 days)

### Timestamps
- All data UTC normalized
- Execution assumed at close prices
- No use of future information (strict <= t-1)

## Reproducibility

### Seeds & Versions
- **Random seed**: 1337 (for all stochastic processes)
- **Python**: 3.10.12
- **Key packages**: Pinned in requirements.txt
- **Data hash**: SHA-256 verification

### Artifacts
- All intermediate results saved to `evidence/`
- Configuration tracked in `config/`
- Model checkpoints versioned

## Limitations & Disclaimers

1. **Demo data only**: Synthetic or redistributable datasets
2. **No market microstructure**: Simplified fill assumptions
3. **Static universe**: No dynamic stock selection
4. **Single strategy**: Not a multi-strat platform
5. **Research only**: Not suitable for live trading

## Audit Trail

Every backtest run generates:
- Configuration snapshot
- Data fingerprint
- Performance metrics JSON
- Trade-level detail CSV
- Equity curve visualization

This ensures full reproducibility and allows independent verification of all results.