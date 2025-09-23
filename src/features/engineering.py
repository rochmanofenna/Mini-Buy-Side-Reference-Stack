"""
Feature engineering for trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats


class FeatureEngineer:
    """Create trading features from market data."""

    def __init__(self):
        self.feature_names = []

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based technical features."""
        features = pd.DataFrame(index=df.index)

        # Returns at multiple horizons
        for period in [1, 5, 10, 20]:
            features[f'returns_{period}d'] = df['close'].pct_change(period)
            features[f'log_returns_{period}d'] = np.log(df['close'] / df['close'].shift(period))

        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'sma_ratio_{period}'] = df['close'] / features[f'sma_{period}']

        # Exponential moving averages
        for period in [12, 26]:
            features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # MACD
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_diff'] = features['macd'] - features['macd_signal']

        # Bollinger Bands
        for period in [20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = sma + (std * 2)
            features[f'bb_lower_{period}'] = sma - (std * 2)
            features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
            features[f'bb_position_{period}'] = (df['close'] - features[f'bb_lower_{period}']) / features[f'bb_width_{period}']

        # Price relative to high/low
        for period in [20, 50]:
            features[f'high_ratio_{period}'] = df['close'] / df['high'].rolling(period).max()
            features[f'low_ratio_{period}'] = df['close'] / df['low'].rolling(period).min()

        return features

    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        features = pd.DataFrame(index=df.index)

        # Volume moving averages
        for period in [5, 10, 20]:
            features[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / features[f'volume_sma_{period}']

        # VWAP
        features['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        features['vwap_ratio'] = df['close'] / features['vwap']

        # On-Balance Volume (OBV)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_sma'] = obv.rolling(20).mean()
        features['obv_diff'] = obv - features['obv_sma']

        # Volume-Price Trend (VPT)
        features['vpt'] = ((df['close'].pct_change() * df['volume'])).fillna(0).cumsum()

        # Money Flow Index components
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        features['money_flow_14'] = money_flow.rolling(14).sum()

        return features

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features."""
        features = pd.DataFrame(index=df.index)

        # Historical volatility
        returns = df['close'].pct_change()
        for period in [10, 20, 30]:
            features[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)

        # Parkinson volatility (using high-low)
        hl_ratio = np.log(df['high'] / df['low'])
        for period in [10, 20]:
            features[f'parkinson_vol_{period}'] = hl_ratio.rolling(period).apply(
                lambda x: np.sqrt(np.sum(x**2) / (4 * period * np.log(2)))
            ) * np.sqrt(252)

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        for period in [14, 20]:
            features[f'atr_{period}'] = true_range.rolling(period).mean()
            features[f'atr_ratio_{period}'] = features[f'atr_{period}'] / df['close']

        # Volatility ratios
        features['vol_ratio_20_10'] = features['volatility_20d'] / features['volatility_10d']
        features['vol_ratio_30_10'] = features['volatility_30d'] / features['volatility_10d']

        return features

    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum indicators."""
        features = pd.DataFrame(index=df.index)

        # RSI (Relative Strength Index)
        for period in [14, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        for period in [14]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            features[f'stoch_k_{period}'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()

        # Williams %R
        for period in [14]:
            features[f'williams_r_{period}'] = -100 * ((high_max - df['close']) / (high_max - low_min))

        # Rate of Change (ROC)
        for period in [10, 20]:
            features[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

        # Commodity Channel Index (CCI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        for period in [20]:
            sma_tp = typical_price.rolling(period).mean()
            mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            features[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)

        return features

    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features."""
        features = pd.DataFrame(index=df.index)

        # Spread metrics (if bid/ask available)
        if 'bid' in df.columns and 'ask' in df.columns:
            features['spread'] = df['ask'] - df['bid']
            features['spread_pct'] = features['spread'] / ((df['ask'] + df['bid']) / 2)
            features['spread_sma'] = features['spread'].rolling(20).mean()

        # High-Low spread
        features['hl_spread'] = (df['high'] - df['low']) / df['close']
        features['hl_spread_sma'] = features['hl_spread'].rolling(20).mean()

        # Close position in daily range
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # Volume clock (volume time)
        cumvol = df['volume'].cumsum()
        avg_daily_vol = df['volume'].rolling(20).mean()
        features['volume_clock'] = cumvol / (avg_daily_vol * 20)

        # Order flow imbalance proxy
        features['ofi_proxy'] = np.sign(df['close'] - df['open']) * df['volume']
        features['ofi_cumsum'] = features['ofi_proxy'].rolling(20).sum()

        # Amihud illiquidity
        features['amihud_illiq'] = np.abs(df['close'].pct_change()) / df['volume']
        features['amihud_illiq_sma'] = features['amihud_illiq'].rolling(20).mean()

        return features

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        features = pd.DataFrame(index=df.index)

        returns = df['close'].pct_change()

        # Rolling statistics
        for period in [20, 60]:
            features[f'skew_{period}'] = returns.rolling(period).skew()
            features[f'kurtosis_{period}'] = returns.rolling(period).kurt()
            features[f'zscore_{period}'] = (df['close'] - df['close'].rolling(period).mean()) / df['close'].rolling(period).std()

        # Autocorrelation
        for lag in [1, 5, 10]:
            features[f'autocorr_{lag}'] = returns.rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )

        # Hurst exponent (simplified)
        def hurst(ts):
            lags = range(2, min(20, len(ts) // 2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0

        features['hurst_20'] = returns.rolling(20).apply(hurst)

        return features

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all feature groups."""
        print("Creating price features...")
        price_features = self.create_price_features(df)

        print("Creating volume features...")
        volume_features = self.create_volume_features(df)

        print("Creating volatility features...")
        volatility_features = self.create_volatility_features(df)

        print("Creating momentum features...")
        momentum_features = self.create_momentum_features(df)

        print("Creating microstructure features...")
        micro_features = self.create_microstructure_features(df)

        print("Creating statistical features...")
        stat_features = self.create_statistical_features(df)

        # Combine all features
        all_features = pd.concat([
            price_features,
            volume_features,
            volatility_features,
            momentum_features,
            micro_features,
            stat_features
        ], axis=1)

        # Remove initial NaN rows
        all_features = all_features.dropna(thresh=len(all_features.columns) * 0.5)

        self.feature_names = all_features.columns.tolist()
        print(f"Created {len(self.feature_names)} features")

        return all_features

    def select_features(self, features: pd.DataFrame, target: pd.Series,
                       n_features: int = 20) -> List[str]:
        """Select top features using statistical tests."""
        from sklearn.feature_selection import f_regression, mutual_info_regression

        # Remove NaN
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx].values
        y = target[valid_idx].values

        # F-statistic
        f_scores, _ = f_regression(X, y)

        # Mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)

        # Combine scores
        scores = pd.DataFrame({
            'feature': features.columns,
            'f_score': f_scores,
            'mi_score': mi_scores,
            'combined': (f_scores / f_scores.max() + mi_scores / mi_scores.max()) / 2
        })

        # Select top features
        top_features = scores.nlargest(n_features, 'combined')['feature'].tolist()

        print(f"Selected top {n_features} features:")
        for i, feat in enumerate(top_features[:10], 1):
            score = scores[scores['feature'] == feat]['combined'].values[0]
            print(f"  {i}. {feat}: {score:.3f}")

        return top_features


def demo():
    """Demo feature engineering."""
    # Generate sample data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'open': 100 + np.random.randn(len(dates)).cumsum(),
        'high': 102 + np.random.randn(len(dates)).cumsum(),
        'low': 98 + np.random.randn(len(dates)).cumsum(),
        'close': 100 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.lognormal(14, 1, len(dates))
    }, index=dates)

    # Ensure high >= close >= low
    df['high'] = df[['high', 'close']].max(axis=1)
    df['low'] = df[['low', 'close']].min(axis=1)

    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_all_features(df)

    # Save sample features
    features.iloc[:100].to_csv('evidence/features_sample.csv')
    print(f"\nSaved sample features to evidence/features_sample.csv")

    return features


if __name__ == "__main__":
    demo()