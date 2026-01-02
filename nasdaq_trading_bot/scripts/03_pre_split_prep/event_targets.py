"""
Event Target Builder for News-Driven Trading Model

This module calculates targets from the post-window ([0, +20] minutes after news).
Primary target: VWAP Return as decided by user.

Key principle: Targets represent what happens AFTER the news is published.
"""

import pandas as pd
import numpy as np
from typing import Dict
from datetime import timedelta


class EventTargetBuilder:
    """
    Calculates targets from post-window data for each event.
    """
    
    def __init__(self, features_df: pd.DataFrame, post_window_minutes: int = 20):
        """
        Initialize target builder.
        
        Args:
            features_df: Full features dataframe (with timestamp index)
            post_window_minutes: Size of post-window in minutes
        """
        self.df = features_df.copy()
        self.post_window_minutes = post_window_minutes
        
        # Ensure timestamp index
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], utc=True)
            self.df = self.df.set_index('timestamp')
        else:
            self.df.index = pd.to_datetime(self.df.index, utc=True)
        
        self.df = self.df.sort_index()
    
    def extract_post_window(self, event_time: pd.Timestamp) -> pd.DataFrame:
        """
        Extract post-window data for a single event.
        
        Args:
            event_time: Event timestamp (news publication time)
            
        Returns:
            DataFrame with post-window bars
        """
        post_start = event_time
        post_end = event_time + timedelta(minutes=self.post_window_minutes)
        
        post_window = self.df.loc[post_start:post_end]
        
        return post_window
    
    def get_base_price(self, post_window: pd.DataFrame) -> float:
        """
        Calculate robust base price from first 1-2 minutes.
        This is more robust to spikes than using just first bar.
        
        Args:
            post_window: Post-window dataframe
            
        Returns:
            Base price (VWAP of first 1-2 bars)
        """
        if len(post_window) == 0:
            return np.nan
        
        # Use first 1-2 bars for robust base price
        n_bars = min(2, len(post_window))
        first_bars = post_window.iloc[:n_bars]
        
        price_col = 'vwap' if 'vwap' in first_bars.columns else 'close'
        
        # VWAP of first bars
        total_pv = (first_bars[price_col] * first_bars['volume']).sum()
        total_volume = first_bars['volume'].sum()
        
        if total_volume > 0:
            return total_pv / total_volume
        else:
            return first_bars[price_col].iloc[0]
    
    def calculate_vwap_return_horizon(self, post_window: pd.DataFrame, horizon_minutes: int, base_price: float = None) -> float:
        """
        Calculate VWAP return for a specific time horizon.
        
        Args:
            post_window: Full post-window dataframe
            horizon_minutes: Time horizon in minutes (e.g., 20, 60, 240)
            base_price: Base price for return calculation (if None, auto-calculate)
            
        Returns:
            VWAP return in percent for the specified horizon
        """
        if len(post_window) == 0:
            return np.nan
        
        # Get base price if not provided
        if base_price is None:
            base_price = self.get_base_price(post_window)
        
        # Extract window up to horizon
        event_time = post_window.index[0]
        horizon_end = event_time + timedelta(minutes=horizon_minutes)
        window = post_window[post_window.index <= horizon_end]
        
        if len(window) == 0:
            return np.nan
        
        # Calculate VWAP in horizon window
        price_col = 'vwap' if 'vwap' in window.columns else 'close'
        
        total_pv = (window[price_col] * window['volume']).sum()
        total_volume = window['volume'].sum()
        
        if total_volume > 0:
            vwap_horizon = total_pv / total_volume
        else:
            return np.nan
        
        # Calculate return in percent
        return (vwap_horizon - base_price) / base_price * 100
    
    def calculate_vwap_return(self, event_time: pd.Timestamp, post_window: pd.DataFrame) -> float:
        """
        Calculate VWAP return: (VWAP_post - Price_t0) / Price_t0 * 100
        DEPRECATED: Use calculate_vwap_return_horizon instead.
        Kept for backward compatibility.
        """
        return self.calculate_vwap_return_horizon(post_window, self.post_window_minutes)
    
    def calculate_max_return(self, event_time: pd.Timestamp, post_window: pd.DataFrame) -> float:
        """
        Calculate maximum positive return in post-window.
        
        Args:
            event_time: Event timestamp
            post_window: Post-window dataframe
            
        Returns:
            Maximum return in percent
        """
        if len(post_window) == 0:
            return np.nan
        
        # Reference price at t0
        if event_time in post_window.index:
            reference_price = post_window.loc[event_time, 'open']
        else:
            reference_price = post_window['open'].iloc[0]
        
        # Find maximum price in post-window
        if 'high' in post_window.columns:
            max_price = post_window['high'].max()
        else:
            max_price = post_window['open'].max()
        
        max_return = (max_price - reference_price) / reference_price * 100
        
        return max_return
    
    def calculate_min_return(self, event_time: pd.Timestamp, post_window: pd.DataFrame) -> float:
        """
        Calculate maximum negative return (drawdown) in post-window.
        
        Args:
            event_time: Event timestamp
            post_window: Post-window dataframe
            
        Returns:
            Minimum return in percent (will be negative for drawdown)
        """
        if len(post_window) == 0:
            return np.nan
        
        # Reference price at t0
        if event_time in post_window.index:
            reference_price = post_window.loc[event_time, 'open']
        else:
            reference_price = post_window['open'].iloc[0]
        
        # Find minimum price in post-window
        if 'low' in post_window.columns:
            min_price = post_window['low'].min()
        else:
            min_price = post_window['open'].min()
        
        min_return = (min_price - reference_price) / reference_price * 100
        
        return min_return
    
    def calculate_end_return(self, event_time: pd.Timestamp, post_window: pd.DataFrame) -> float:
        """
        Calculate return at end of post-window (after exactly 20 minutes).
        
        Args:
            event_time: Event timestamp
            post_window: Post-window dataframe
            
        Returns:
            End return in percent
        """
        if len(post_window) == 0:
            return np.nan
        
        # Reference price at t0
        if event_time in post_window.index:
            reference_price = post_window.loc[event_time, 'open']
        else:
            reference_price = post_window['open'].iloc[0]
        
        # End price (last price in post-window)
        end_price = post_window['open'].iloc[-1]
        
        end_return = (end_price - reference_price) / reference_price * 100
        
        return end_return
    
    def calculate_direction_label(self, vwap_return: float, threshold: float = 0.1) -> int:
        """
        Calculate directional label based on VWAP return.
        
        Args:
            vwap_return: VWAP return in percent
            threshold: Threshold for neutral classification (default 0.1%)
            
        Returns:
            -1 (bearish), 0 (neutral), or 1 (bullish)
        """
        if np.isnan(vwap_return):
            return 0
        
        if vwap_return > threshold:
            return 1  # Bullish
        elif vwap_return < -threshold:
            return -1  # Bearish
        else:
            return 0  # Neutral
    
    def calculate_profitable_labels(self, max_return: float) -> Dict:
        """
        Calculate binary profitable labels.
        
        Args:
            max_return: Maximum return in percent
            
        Returns:
            Dictionary with binary labels for different thresholds
        """
        return {
            'target_profitable_05pct': 1 if max_return > 0.5 else 0,
            'target_profitable_1pct': 1 if max_return > 1.0 else 0,
        }
    
    def build_all_targets(self, event_time: pd.Timestamp) -> Dict:
        """
        Build all targets for a single event with multiple time horizons.
        
        Args:
            event_time: Event timestamp
            
        Returns:
            Dictionary with all targets (20m, 60m, 4h horizons)
        """
        # Extract post-window (needs to be large enough for longest horizon)
        post_window = self.extract_post_window(event_time)
        
        targets = {}
        
        if len(post_window) == 0:
            # Return all NaN targets if no data
            return {
                'target_vwap_return_20m': np.nan,
                'target_vwap_return_60m': np.nan,
                'target_vwap_return_4h': np.nan,
                'target_max_return_20m': np.nan,
                'target_min_return_20m': np.nan,
                'target_end_return_20m': np.nan,
                'target_direction': 0,
                'target_profitable_05pct': 0,
                'target_profitable_1pct': 0,
                'target_max_drawdown_20m': np.nan,
                'post_bars_count': 0,
                'post_coverage_pct': 0.0
            }
        
        # Get consistent base price for ALL horizons (robust to spikes)
        base_price = self.get_base_price(post_window)
        
        # MULTIPLE TIME HORIZONS: All use same base price
        # 20min (original)
        targets['target_vwap_return_20m'] = self.calculate_vwap_return_horizon(
            post_window, 20, base_price
        )
        
        # 60min (NEW - for deployment)
        targets['target_vwap_return_60m'] = self.calculate_vwap_return_horizon(
            post_window, 60, base_price
        )
        
        # 4h (NEW - for longer-term)
        targets['target_vwap_return_4h'] = self.calculate_vwap_return_horizon(
            post_window, 240, base_price
        )
        
        # Use 20m for backward compatibility with secondary targets
        vwap_return_20m = targets['target_vwap_return_20m']
        
        # Secondary targets (for analysis and alternative strategies)
        targets['target_max_return_20m'] = self.calculate_max_return(event_time, post_window)
        targets['target_min_return_20m'] = self.calculate_min_return(event_time, post_window)
        targets['target_end_return_20m'] = self.calculate_end_return(event_time, post_window)
        
        # Classification labels (based on 20m for now)
        targets['target_direction'] = self.calculate_direction_label(vwap_return_20m)
        
        # Profitable labels
        if not np.isnan(targets['target_max_return_20m']):
            targets.update(self.calculate_profitable_labels(targets['target_max_return_20m']))
        else:
            targets['target_profitable_05pct'] = 0
            targets['target_profitable_1pct'] = 0
        
        # Risk metric: max drawdown
        targets['target_max_drawdown_20m'] = targets['target_min_return_20m']
        
        # Coverage metadata
        targets['post_bars_count'] = len(post_window)
        targets['post_coverage_pct'] = len(post_window) / self.post_window_minutes * 100
        
        return targets


def main():
    """Test the target builder."""
    import os
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    
    features_file = os.path.join(data_dir, 'nasdaq100_index_1m_features.csv')
    events_file = os.path.join(data_dir, 'news_events_metadata.csv')
    
    print("Loading data...")
    features_df = pd.read_csv(features_file)
    events_df = pd.read_csv(events_file)
    
    print(f"Loaded {len(features_df):,} feature rows")
    print(f"Loaded {len(events_df):,} events")
    
    # Initialize target builder with 60min post-window
    builder = EventTargetBuilder(features_df, post_window_minutes=60)
    
    # Test on first event
    first_event = events_df.iloc[0]
    event_time = pd.to_datetime(first_event['event_time'], utc=True)
    
    print(f"\nTesting on first event: {event_time}")
    
    targets = builder.build_all_targets(event_time)
    
    print("\nCalculated Targets:")
    for key, value in targets.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}%")
        else:
            print(f"  {key}: {value}")
    
    print("\nTarget builder test complete!")


if __name__ == "__main__":
    main()
