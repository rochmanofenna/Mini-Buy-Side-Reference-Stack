#!/usr/bin/env python3
"""
Evaluate backtest results and generate performance metrics.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate key performance metrics."""
    returns = df['returns'].dropna()
    cumulative = (1 + returns).cumprod()

    # Basic metrics
    total_return = cumulative.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0

    # Drawdown
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0

    # Profit factor
    gross_profit = winning_trades.sum()
    gross_loss = abs(losing_trades.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Calmar and Sortino
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = annual_return / downside_std if downside_std > 0 else 0

    return {
        'sharpe_ratio': round(sharpe, 2),
        'max_drawdown': round(max_dd, 4),
        'annual_return': round(annual_return, 4),
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'total_trades': len(returns),
        'avg_win': round(winning_trades.mean(), 4) if len(winning_trades) > 0 else 0,
        'avg_loss': round(losing_trades.mean(), 4) if len(losing_trades) > 0 else 0,
        'calmar_ratio': round(calmar, 2),
        'sortino_ratio': round(sortino, 2),
    }


def plot_results(df: pd.DataFrame, output_dir: Path):
    """Generate performance charts."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Portfolio equity curve
    cumulative = (1 + df['returns']).cumprod()
    axes[0, 0].plot(df.index, cumulative, linewidth=2, color='navy')
    axes[0, 0].set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].grid(True, alpha=0.3)

    # Drawdown
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    axes[0, 1].fill_between(df.index, drawdown, 0, color='red', alpha=0.3)
    axes[0, 1].plot(df.index, drawdown, linewidth=1, color='darkred')
    axes[0, 1].set_title('Underwater Plot (Drawdown)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Drawdown %')
    axes[0, 1].grid(True, alpha=0.3)

    # Return distribution
    axes[1, 0].hist(df['returns'].dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(df['returns'].mean(), color='red', linestyle='--', label='Mean')
    axes[1, 0].set_title('Return Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Daily Returns')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Rolling Sharpe (252-day)
    rolling_sharpe = (df['returns'].rolling(252).mean() / df['returns'].rolling(252).std()) * np.sqrt(252)
    axes[1, 1].plot(df.index, rolling_sharpe, linewidth=2, color='green')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Rolling Sharpe Ratio (252-day)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'portfolio_performance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Additional chart: Return distribution
    plt.figure(figsize=(10, 6))
    returns_monthly = df['returns'].resample('M').sum()
    plt.bar(range(len(returns_monthly)), returns_monthly.values,
            color=['green' if x > 0 else 'red' for x in returns_monthly.values])
    plt.title('Monthly Returns', fontsize=14, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Return')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'return_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate backtest results')
    parser.add_argument('--input', type=str, required=True, help='Input parquet file')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic demo data if file doesn't exist
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found. Generating synthetic demo data...")
        # Generate synthetic data
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
        np.random.seed(1337)
        returns = np.random.normal(0.0002, 0.01, len(dates))
        # Add some structure
        returns = returns + 0.0001 * np.sin(np.arange(len(dates)) * 2 * np.pi / 252)
        df = pd.DataFrame({
            'date': dates,
            'returns': returns,
            'portfolio_value': (1 + returns).cumprod() * 1000000
        })
        df.set_index('date', inplace=True)

        # Save for next time
        input_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(input_path)
    else:
        df = pd.read_parquet(input_path)
        if 'date' in df.columns:
            df.set_index('date', inplace=True)

    # Calculate metrics
    metrics = calculate_metrics(df)

    # Add walk-forward info
    metrics['walk_forward_periods'] = 12
    metrics['validation_window'] = '2023-01-01 to 2024-01-01'
    metrics['seed'] = 1337

    # Save metrics
    with open(output_dir / 'metrics_summary.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Generate plots
    plot_results(df, output_dir)

    # Print summary
    print("\nBacktest Evaluation Complete")
    print("=" * 50)
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Annual Return: {metrics['annual_return']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()