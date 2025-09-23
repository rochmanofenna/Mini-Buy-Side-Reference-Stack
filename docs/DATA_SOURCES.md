# Data Sources & Licensing

## Overview

This reference stack uses exclusively **demo, synthetic, or redistributable** data to ensure full reproducibility and avoid licensing issues.

## Primary Data Sources

### 1. Synthetic Market Data
- **Generator**: Internal GBM/Monte-Carlo engine
- **Parameters**: Calibrated to realistic market dynamics
- **Symbols**: 100 synthetic tickers (SYN001-SYN100)
- **Frequency**: 1-minute bars aggregated from tick simulation
- **License**: MIT (generated data)

### 2. Demo Dataset
- **Source**: Generated from public market statistics
- **Coverage**: 2020-2024 daily OHLCV
- **Universe**: 50 liquid names (anonymized)
- **Purpose**: Backtesting and strategy development
- **License**: CC0 (public domain equivalent)

### 3. Reference Data
- **Static data**: Sector mappings, market holidays
- **Source**: Public sources (exchanges, Wikipedia)
- **Updates**: Quarterly refresh
- **License**: Various open licenses

## Data Schema

### OHLCV Format
```python
{
    "timestamp": "2024-01-15T09:30:00Z",
    "symbol": "SYN001",
    "open": 100.25,
    "high": 101.30,
    "low": 99.80,
    "close": 100.90,
    "volume": 1250000,
    "vwap": 100.85,
    "trade_count": 3421
}
```

### Order Book Snapshot
```python
{
    "timestamp": "2024-01-15T09:30:00.123Z",
    "symbol": "SYN001",
    "bids": [[100.89, 500], [100.88, 1000], ...],
    "asks": [[100.91, 300], [100.92, 800], ...],
    "mid": 100.90,
    "spread": 0.02
}
```

## Data Quality

### Synthetic Data Properties
- **Correlation structure**: Realistic sector correlations
- **Volatility clustering**: GARCH(1,1) dynamics
- **Jump component**: Poisson arrivals for gaps
- **Microstructure**: Tick size, lot size constraints

### Validation Checks
- No forward-looking bias
- Monotonic timestamps
- Price continuity (adjusted for splits)
- Volume profile consistency

## Storage Format

### File Structure
```
data/
├── raw/
│   ├── ohlcv/
│   │   └── {YYYY-MM-DD}/
│   │       └── {symbol}.parquet
│   └── orderbook/
│       └── {YYYY-MM-DD}/
│           └── {symbol}.parquet
├── processed/
│   ├── features/
│   └── signals/
└── reference/
    ├── symbols.csv
    ├── holidays.csv
    └── sectors.csv
```

### Parquet Schema
- **Compression**: Snappy
- **Row group size**: 100MB
- **Partitioning**: By date and symbol
- **Statistics**: Min/max for efficient filtering

## Data Pipeline

### Ingestion Flow
1. **Generation/Download**: Create synthetic or fetch demo data
2. **Validation**: Schema checks, outlier detection
3. **Transformation**: Normalize, adjust for corporate actions
4. **Storage**: Write to Parquet with versioning
5. **Indexing**: Update metadata catalog

### Update Frequency
- **Real-time sim**: 100ms for synthetic feed
- **Historical**: Daily batch after market close
- **Reference**: Quarterly updates

## Compliance & Legal

### Usage Restrictions
1. **Demo purposes only**: Not real market data
2. **No redistribution**: Of any external data sources
3. **No warranties**: Data provided as-is
4. **Research only**: Not for production trading

### Attribution
- Synthetic data generation methodology inspired by academic literature
- No proprietary data sources used
- All external data properly attributed

## Data Access

### Reading Data
```python
import pandas as pd
import pyarrow.parquet as pq

# Read OHLCV
df = pd.read_parquet('data/raw/ohlcv/2024-01-15/SYN001.parquet')

# Read with filtering
table = pq.read_table(
    'data/raw/ohlcv/',
    filters=[('timestamp', '>=', '2024-01-01')]
)
```

### Data Catalog
Access via `data/catalog.json`:
```json
{
    "datasets": {
        "ohlcv_demo": {
            "path": "data/raw/ohlcv/",
            "format": "parquet",
            "schema_version": "1.0.0",
            "row_count": 1250000,
            "date_range": ["2020-01-01", "2024-01-15"],
            "checksum": "sha256:abcd1234..."
        }
    }
}
```

## Version History

### v1.0.0 (2024-01)
- Initial synthetic dataset
- 100 symbols, 4 years history
- Basic order book simulation

### Future Additions
- Options chain (synthetic)
- News sentiment scores (generated)
- Fundamental ratios (randomized)

## Contact

For data questions or access to enhanced demo datasets:
- GitHub Issues: [repo]/issues
- Documentation: See README.md

---

**Note**: This is a reference implementation. Production systems require proper data licensing, vendor agreements, and compliance review.