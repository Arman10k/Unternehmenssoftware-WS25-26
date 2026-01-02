"""
Event Feature Builder for News-Driven Trading Model

This module builds features from the pre-window ([-20, 0] minutes before news).
Features capture market state and momentum BEFORE the news is published.

Key principle: NO look-ahead bias - only use data from before event time.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List
from datetime import timedelta


class EventFeatureBuilder:
    """
    Builds aggregated features from pre-window data for each event.
    """
    
    def __init__(self, features_df: pd.DataFrame, pre_window_minutes: int = 20):
        """
        Initialize feature builder.
        
        Args:
            features_df: Full features dataframe (with timestamp index)
            pre_window_minutes: Size of pre-window in minutes
        """
        self.df = features_df.copy()
        self.pre_window_minutes = pre_window_minutes
        
        # Ensure timestamp index
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], utc=True)
            self.df = self.df.set_index('timestamp')
        else:
            self.df.index = pd.to_datetime(self.df.index, utc=True)
        
        self.df = self.df.sort_index()
    
    def extract_pre_window(self, event_time: pd.Timestamp) -> pd.DataFrame:
        """
        Extract pre-window data for a single event.
        
        Args:
            event_time: Event timestamp (news publication time)
            
        Returns:
            DataFrame with pre-window bars
        """
        pre_start = event_time - timedelta(minutes=self.pre_window_minutes)
        pre_end = event_time
        
        # Get pre-window, excluding the event_time itself
        pre_window = self.df.loc[pre_start:pre_end]
        pre_window = pre_window[pre_window.index < event_time]
        
        return pre_window
    
    def build_price_momentum_features(self, pre_window: pd.DataFrame) -> Dict:
        """
        Build price momentum features from pre-window.
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        if len(pre_window) < 2:
            return {
                'pre_return_5m': np.nan,
                'pre_return_10m': np.nan,
                'pre_return_20m': np.nan,
                'pre_price_trend': np.nan,
                'pre_price_std': np.nan,
            }
        
        price_col = 'open'  # Using open price as reference
        prices = pre_window[price_col]
        
        # Returns over different windows
        if len(pre_window) >= 5:
            features['pre_return_5m'] = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100
        else:
            features['pre_return_5m'] = np.nan
        
        if len(pre_window) >= 10:
            features['pre_return_10m'] = (prices.iloc[-1] / prices.iloc[-10] - 1) * 100
        else:
            features['pre_return_10m'] = np.nan
        
        # Full window return
        features['pre_return_20m'] = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        
        # Price trend (linear regression slope)
        x = np.arange(len(prices))
        if len(x) > 1:
            slope, _ = np.polyfit(x, prices.values, 1)
            features['pre_price_trend'] = slope
        else:
            features['pre_price_trend'] = 0.0
        
        # Price volatility
        features['pre_price_std'] = prices.std()
        
        return features
    
    def build_volume_features(self, pre_window: pd.DataFrame) -> Dict:
        """
        Build volume-based features from pre-window.
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        if len(pre_window) == 0 or 'volume' not in pre_window.columns:
            return {
                'pre_volume_mean': np.nan,
                'pre_volume_std': np.nan,
                'pre_volume_spike': np.nan,
                'pre_volume_trend': np.nan,
            }
        
        volumes = pre_window['volume']
        
        features['pre_volume_mean'] = volumes.mean()
        features['pre_volume_std'] = volumes.std()
        
        # Volume spike: max / mean
        if features['pre_volume_mean'] > 0:
            features['pre_volume_spike'] = volumes.max() / features['pre_volume_mean']
        else:
            features['pre_volume_spike'] = 1.0
        
        # Volume trend
        x = np.arange(len(volumes))
        if len(x) > 1:
            slope, _ = np.polyfit(x, volumes.values, 1)
            features['pre_volume_trend'] = slope
        else:
            features['pre_volume_trend'] = 0.0
        
        return features
    
    def build_trade_activity_features(self, pre_window: pd.DataFrame) -> Dict:
        """
        Build trade activity features from pre-window.
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        if len(pre_window) == 0:
            return {
                'pre_trade_count_mean': np.nan,
                'pre_avg_trade_size': np.nan,
            }
        
        # Trade count
        if 'trade_count' in pre_window.columns:
            features['pre_trade_count_mean'] = pre_window['trade_count'].mean()
            
            # Average trade size = volume / trade_count
            if 'volume' in pre_window.columns:
                valid_trades = pre_window[pre_window['trade_count'] > 0]
                if len(valid_trades) > 0:
                    avg_sizes = valid_trades['volume'] / valid_trades['trade_count']
                    features['pre_avg_trade_size'] = avg_sizes.mean()
                else:
                    features['pre_avg_trade_size'] = np.nan
            else:
                features['pre_avg_trade_size'] = np.nan
        else:
            features['pre_trade_count_mean'] = np.nan
            features['pre_avg_trade_size'] = np.nan
        
        return features
    
    def build_volatility_features(self, pre_window: pd.DataFrame) -> Dict:
        """
        Build volatility features from pre-window.
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        if len(pre_window) < 2:
            return {
                'pre_realized_vol': np.nan,
                'pre_hl_span_mean': np.nan,
            }
        
        price_col = 'open'
        
        # Realized volatility (std of log returns)
        log_returns = np.log(pre_window[price_col] / pre_window[price_col].shift(1))
        features['pre_realized_vol'] = log_returns.std()
        
        # High-Low span
        if 'high' in pre_window.columns and 'low' in pre_window.columns:
            hl_spans = pre_window['high'] - pre_window['low']
            features['pre_hl_span_mean'] = hl_spans.mean()
        else:
            features['pre_hl_span_mean'] = np.nan
        
        return features
    
    def build_time_features(self, event_time: pd.Timestamp) -> Dict:
        """
        Build time-based features.
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Hour of day
        features['news_hour'] = event_time.hour
        
        # Minute of hour
        features['news_minute'] = event_time.minute
        
        # Day of week (0=Monday, 6=Sunday)
        features['news_day_of_week'] = event_time.dayofweek
        
        # Minutes since market open (assuming 9:30 ET = 14:30 UTC)
        market_open_utc = event_time.replace(hour=14, minute=30, second=0, microsecond=0)
        if event_time >= market_open_utc:
            features['minutes_since_market_open'] = (event_time - market_open_utc).total_seconds() / 60.0
        else:
            features['minutes_since_market_open'] = 0.0
        
        return features
    
    def build_all_features(self, event_time: pd.Timestamp, news_sentiment: float = None) -> Dict:
        """
        Build all features for a single event.
        
        Args:
            event_time: Event timestamp
            news_sentiment: Sentiment score of the news (if available)
            
        Returns:
            Dictionary with all features
        """
        # Extract pre-window
        pre_window = self.extract_pre_window(event_time)
        
        # Build feature groups
        features = {}
        
        # Price momentum
        features.update(self.build_price_momentum_features(pre_window))
        
        # Volume
        features.update(self.build_volume_features(pre_window))
        
        # Trade activity
        features.update(self.build_trade_activity_features(pre_window))
        
        # Volatility
        features.update(self.build_volatility_features(pre_window))
        
        # Time features
        features.update(self.build_time_features(event_time))
        
        # News sentiment (if available)
        if news_sentiment is not None:
            features['news_sentiment'] = news_sentiment
        else:
            features['news_sentiment'] = 0.0
        
        # Coverage metadata
        features['pre_bars_count'] = len(pre_window)
        features['pre_coverage_pct'] = len(pre_window) / self.pre_window_minutes * 100
        
        return features


def main():
    """Test the feature builder."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    
    features_file = os.path.join(data_dir, 'nasdaq100_index_1m_features.csv')
    events_file = os.path.join(data_dir, 'news_events_metadata.csv')
    
    print("Loading data...")
    features_df = pd.read_csv(features_file)
    events_df = pd.read_csv(events_file)
    
    print(f"Loaded {len(features_df):,} feature rows")
    print(f"Loaded {len(events_df):,} events")
    
    # Initialize feature builder
    builder = EventFeatureBuilder(features_df, pre_window_minutes=20)
    
    # Test on first event
    first_event = events_df.iloc[0]
    event_time = pd.to_datetime(first_event['event_time'], utc=True)
    news_sentiment = first_event.get('news_sentiment', None)
    
    print(f"\nTesting on first event: {event_time}")
    
    features = builder.build_all_features(event_time, news_sentiment)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
    
    print("\nFeature builder test complete!")


if __name__ == "__main__":
    main()
