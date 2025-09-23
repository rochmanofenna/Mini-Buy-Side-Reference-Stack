#!/usr/bin/env python3
"""
Generate comprehensive performance tearsheet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from datetime import datetime

sns.set_style("whitegrid")


class TearsheetGenerator:
    """Generate professional trading tearsheet."""

    def __init__(self, results_path: str):
        self.results_path = results_path
        self.metrics = {}

    def calculate_all_metrics(self, returns: pd.Series) -> dict:
        """Calculate comprehensive performance metrics."""

        # Basic stats
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Find drawdown duration
        dd_start = drawdown[drawdown == max_dd].index[0]
        dd_recovery = drawdown[drawdown.index > dd_start]
        dd_recovery = dd_recovery[dd_recovery >= 0]
        if len(dd_recovery) > 0:
            dd_duration = (dd_recovery.index[0] - dd_start).days
        else:
            dd_duration = (drawdown.index[-1] - dd_start).days

        # Risk metrics
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = annual_return / downside_std if downside_std > 0 else 0
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        # Win/loss analysis
        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]
        win_rate = len(winning_days) / len(returns) if len(returns) > 0 else 0

        avg_win = winning_days.mean() if len(winning_days) > 0 else 0
        avg_loss = losing_days.mean() if len(losing_days) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        gross_profit = winning_days.sum()
        gross_loss = abs(losing_days.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Risk-adjusted metrics
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'max_dd_duration': dd_duration,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'total_days': len(returns),
            'winning_days': len(winning_days),
            'losing_days': len(losing_days),
            'flat_days': len(returns[returns == 0])
        }

    def generate_plots(self, df: pd.DataFrame, output_dir: Path):
        """Generate tearsheet plots."""

        fig = plt.figure(figsize=(16, 20))

        # 1. Cumulative Returns
        ax1 = plt.subplot(6, 2, 1)
        cumulative = (1 + df['returns']).cumprod()
        ax1.plot(df.index, cumulative, linewidth=2, color='navy')
        ax1.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown
        ax2 = plt.subplot(6, 2, 2)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        ax2.fill_between(df.index, drawdown * 100, 0, color='red', alpha=0.3)
        ax2.plot(df.index, drawdown * 100, linewidth=1, color='darkred')
        ax2.set_title('Underwater Plot (Drawdown)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True, alpha=0.3)

        # 3. Daily Returns
        ax3 = plt.subplot(6, 2, 3)
        ax3.bar(df.index, df['returns'] * 100,
               color=['green' if x > 0 else 'red' for x in df['returns']],
               width=1, alpha=0.6)
        ax3.set_title('Daily Returns', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Return %')
        ax3.grid(True, alpha=0.3)

        # 4. Return Distribution
        ax4 = plt.subplot(6, 2, 4)
        ax4.hist(df['returns'] * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(df['returns'].mean() * 100, color='red', linestyle='--', label='Mean')
        ax4.set_title('Return Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Return %')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Rolling Sharpe (60-day)
        ax5 = plt.subplot(6, 2, 5)
        rolling_sharpe = (df['returns'].rolling(60).mean() /
                         df['returns'].rolling(60).std()) * np.sqrt(252)
        ax5.plot(df.index, rolling_sharpe, linewidth=2, color='green')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Sharpe=1')
        ax5.set_title('Rolling Sharpe Ratio (60-day)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Rolling Volatility
        ax6 = plt.subplot(6, 2, 6)
        rolling_vol = df['returns'].rolling(30).std() * np.sqrt(252) * 100
        ax6.plot(df.index, rolling_vol, linewidth=2, color='orange')
        ax6.set_title('Rolling Volatility (30-day)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Volatility %')
        ax6.grid(True, alpha=0.3)

        # 7. Monthly Returns Heatmap
        ax7 = plt.subplot(6, 2, 7)
        monthly_returns = df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_pivot = pd.DataFrame(monthly_returns)
        monthly_pivot['Year'] = monthly_pivot.index.year
        monthly_pivot['Month'] = monthly_pivot.index.month
        monthly_matrix = monthly_pivot.pivot_table(values='returns', index='Year', columns='Month')

        sns.heatmap(monthly_matrix * 100, annot=True, fmt='.1f',
                   cmap='RdYlGn', center=0, ax=ax7, cbar_kws={'label': 'Return %'})
        ax7.set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')

        # 8. Q-Q Plot
        ax8 = plt.subplot(6, 2, 8)
        from scipy import stats
        stats.probplot(df['returns'], dist="norm", plot=ax8)
        ax8.set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)

        # 9. Cumulative Returns by Year
        ax9 = plt.subplot(6, 2, 9)
        yearly_returns = df['returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
        colors = ['green' if x > 0 else 'red' for x in yearly_returns]
        ax9.bar(range(len(yearly_returns)), yearly_returns * 100, color=colors)
        ax9.set_title('Annual Returns', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Return %')
        ax9.set_xticks(range(len(yearly_returns)))
        ax9.set_xticklabels([str(y.year) for y in yearly_returns.index], rotation=45)
        ax9.grid(True, alpha=0.3)

        # 10. Win Rate by Month
        ax10 = plt.subplot(6, 2, 10)
        monthly_winrate = df['returns'].resample('M').apply(lambda x: (x > 0).mean())
        ax10.plot(monthly_winrate.index, monthly_winrate * 100, linewidth=2, marker='o')
        ax10.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax10.set_title('Monthly Win Rate', fontsize=12, fontweight='bold')
        ax10.set_ylabel('Win Rate %')
        ax10.grid(True, alpha=0.3)

        # 11. Return vs Volatility Scatter
        ax11 = plt.subplot(6, 2, 11)
        rolling_returns = df['returns'].rolling(30).mean() * 252 * 100
        rolling_vol_scatter = df['returns'].rolling(30).std() * np.sqrt(252) * 100
        ax11.scatter(rolling_vol_scatter, rolling_returns, alpha=0.5, s=10)
        ax11.set_title('Return vs Volatility (30-day rolling)', fontsize=12, fontweight='bold')
        ax11.set_xlabel('Volatility %')
        ax11.set_ylabel('Annualized Return %')
        ax11.grid(True, alpha=0.3)

        # 12. Performance Summary Text
        ax12 = plt.subplot(6, 2, 12)
        ax12.axis('off')

        summary_text = f"""
        PERFORMANCE SUMMARY
        {'='*30}
        Total Return: {self.metrics['total_return']:.2%}
        Annual Return: {self.metrics['annual_return']:.2%}
        Volatility: {self.metrics['volatility']:.2%}
        Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}
        Sortino Ratio: {self.metrics['sortino_ratio']:.2f}
        Calmar Ratio: {self.metrics['calmar_ratio']:.2f}

        Max Drawdown: {self.metrics['max_drawdown']:.2%}
        DD Duration: {self.metrics['max_dd_duration']} days

        Win Rate: {self.metrics['win_rate']:.2%}
        Profit Factor: {self.metrics['profit_factor']:.2f}
        Win/Loss Ratio: {self.metrics['win_loss_ratio']:.2f}

        VaR (95%): {self.metrics['var_95']:.2%}
        CVaR (95%): {self.metrics['cvar_95']:.2%}

        Skewness: {self.metrics['skewness']:.2f}
        Kurtosis: {self.metrics['kurtosis']:.2f}
        """

        ax12.text(0.1, 0.9, summary_text, fontsize=10, family='monospace',
                 verticalalignment='top', transform=ax12.transAxes)

        plt.suptitle('Trading Strategy Performance Tearsheet', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_path = output_dir / 'tearsheet.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Tearsheet saved to {output_path}")

    def generate_tearsheet(self, output_dir: str):
        """Generate complete tearsheet."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load or generate data
        if Path(self.results_path).exists():
            df = pd.read_parquet(self.results_path)
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
        else:
            # Generate synthetic data for demo
            print("Generating synthetic data for demo...")
            dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
            np.random.seed(1337)
            returns = np.random.normal(0.0002, 0.01, len(dates))
            df = pd.DataFrame({'returns': returns}, index=dates)

        # Calculate metrics
        self.metrics = self.calculate_all_metrics(df['returns'])

        # Save metrics
        metrics_path = output_dir / 'tearsheet_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

        # Generate plots
        self.generate_plots(df, output_dir)

        # Print summary
        print("\n" + "="*60)
        print("TEARSHEET GENERATION COMPLETE")
        print("="*60)
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {self.metrics['max_drawdown']:.2%}")
        print(f"Annual Return: {self.metrics['annual_return']:.2%}")
        print(f"Win Rate: {self.metrics['win_rate']:.2%}")
        print(f"\nFiles saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate performance tearsheet')
    parser.add_argument('--input', type=str, default='out/backtest_results.parquet',
                       help='Input results file')
    parser.add_argument('--output', type=str, default='out/tearsheet',
                       help='Output directory')

    args = parser.parse_args()

    generator = TearsheetGenerator(args.input)
    generator.generate_tearsheet(args.output)


if __name__ == "__main__":
    main()