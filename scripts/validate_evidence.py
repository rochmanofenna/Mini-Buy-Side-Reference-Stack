#!/usr/bin/env python3
"""
Validate evidence artifacts independently.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def validate_metrics(metrics_path: str) -> bool:
    """Validate metrics summary JSON."""
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        required_fields = [
            'sharpe_ratio', 'max_drawdown', 'annual_return',
            'win_rate', 'profit_factor', 'total_trades'
        ]

        for field in required_fields:
            if field not in metrics:
                print(f"ERROR: Missing field '{field}' in metrics")
                return False

        # Validate ranges
        if not 0 <= metrics['win_rate'] <= 1:
            print(f"ERROR: Invalid win_rate {metrics['win_rate']}")
            return False

        if metrics['max_drawdown'] > 0:
            print(f"ERROR: Max drawdown should be negative, got {metrics['max_drawdown']}")
            return False

        print(f"✓ Metrics valid: Sharpe={metrics['sharpe_ratio']}, MaxDD={metrics['max_drawdown']:.2%}")
        return True

    except Exception as e:
        print(f"ERROR validating metrics: {e}")
        return False


def validate_signals(signals_path: str) -> bool:
    """Validate signals CSV."""
    try:
        df = pd.read_csv(signals_path)

        required_columns = [
            'timestamp', 'symbol', 'signal_type', 'signal_value',
            'confidence', 'position', 'pnl', 'hit'
        ]

        for col in required_columns:
            if col not in df.columns:
                print(f"ERROR: Missing column '{col}' in signals")
                return False

        # Calculate statistics
        total_trades = len(df[df['position'] != 'flat'])
        winning_trades = df['hit'].sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()

        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        print(f"✓ Signals valid: {total_trades} trades, Win rate={win_rate:.1%}, PF={profit_factor:.2f}")
        print(f"  Total PnL={total_pnl:.2f}, Avg PnL={avg_pnl:.4f}")

        return True

    except Exception as e:
        print(f"ERROR validating signals: {e}")
        return False


def validate_k6_results(k6_path: str) -> bool:
    """Validate k6 benchmark results."""
    try:
        with open(k6_path, 'r') as f:
            results = json.load(f)

        metrics = results.get('metrics', {})
        http_duration = metrics.get('http_req_duration', {})

        if not http_duration:
            print("ERROR: No http_req_duration metrics found")
            return False

        p99 = http_duration.get('p(99)', 0)
        p95 = http_duration.get('p(95)', 0)
        median = http_duration.get('median', 0)

        if p99 <= 0:
            print("ERROR: Invalid p99 latency")
            return False

        print(f"✓ K6 results valid: p99={p99:.1f}ms, p95={p95:.1f}ms, median={median:.1f}ms")

        # Check request rate
        http_reqs = metrics.get('http_reqs', {})
        rate = http_reqs.get('rate', 0)
        print(f"  Request rate: {rate:.1f} req/s")

        return True

    except Exception as e:
        print(f"ERROR validating k6 results: {e}")
        return False


def validate_all():
    """Validate all evidence artifacts."""
    evidence_dir = Path("evidence")

    print("=" * 60)
    print("EVIDENCE VALIDATION REPORT")
    print("=" * 60)

    all_valid = True

    # Check metrics
    metrics_file = evidence_dir / "backtests" / "metrics_summary.json"
    if metrics_file.exists():
        if not validate_metrics(str(metrics_file)):
            all_valid = False
    else:
        print(f"WARNING: Metrics file not found: {metrics_file}")

    print()

    # Check signals
    signals_files = list((evidence_dir / "signals").glob("*.csv"))
    if signals_files:
        for signals_file in signals_files:
            if not validate_signals(str(signals_file)):
                all_valid = False
    else:
        print("WARNING: No signals CSV files found")

    print()

    # Check k6 results
    k6_file = evidence_dir / "benchmarks" / "k6_http_ci.json"
    if k6_file.exists():
        if not validate_k6_results(str(k6_file)):
            all_valid = False
    else:
        print(f"WARNING: K6 results file not found: {k6_file}")

    print("=" * 60)

    if all_valid:
        print("✓ ALL EVIDENCE VALIDATED SUCCESSFULLY")
        return 0
    else:
        print("✗ VALIDATION FAILED - Check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(validate_all())