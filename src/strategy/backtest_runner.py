"""
Backtest runner for strategy evaluation.
"""

import argparse
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import sys


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_synthetic_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Generate synthetic market data for backtesting."""
    np.random.seed(config['backtest']['seed'])

    start_date = config['backtest']['start_date']
    end_date = config['backtest']['end_date']

    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)

    # Generate returns with some structure
    returns = np.random.normal(0.0002, 0.01, n_days)
    returns = returns + 0.0001 * np.sin(np.arange(n_days) * 2 * np.pi / 252)

    df = pd.DataFrame({
        'date': dates,
        'returns': returns,
        'portfolio_value': (1 + returns).cumprod() * config['backtest']['initial_capital']
    })

    return df


def run_backtest(config: Dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
    """Run the backtest simulation."""
    # Simple momentum strategy simulation
    data['signal'] = data['returns'].rolling(20).mean()
    data['position'] = np.where(data['signal'] > 0, 1, -1)
    data['strategy_returns'] = data['position'].shift(1) * data['returns']

    # Apply transaction costs
    data['trade'] = data['position'].diff().abs()
    data['tc'] = data['trade'] * config['execution']['commission']
    data['net_returns'] = data['strategy_returns'] - data['tc']

    return data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run backtest')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--output', type=str, required=True, help='Output file path')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config['backtest']['seed'] = args.seed

    # Generate data
    data = generate_synthetic_data(config)

    # Run backtest
    results = run_backtest(config, data)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_path)

    print(f"Backtest complete. Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())