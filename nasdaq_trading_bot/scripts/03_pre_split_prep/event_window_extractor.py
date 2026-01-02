"""
Event Window Extractor for News-Driven Trading Model

This script extracts event windows around news publications:
- Pre-window: [-20, 0] minutes before news (for features)
- Post-window: [0, +20] minutes after news (for targets)

User Decisions:
- Minimum 15/20 bars required in each window
- First-news-wins strategy (40-min lockout for overlapping events)
- VWAP return as primary target
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta
from typing import Tuple, List, Dict


class EventWindowExtractor:
    """
    Extracts event windows from the features dataset around news publications.
    """
    
    def __init__(
        self,
        features_file: str,
        pre_window_minutes: int = 20,
        post_window_minutes: int = 20,
        min_bars_required: int = 15,
        lockout_minutes: int = 40
    ):
        """
        Initialize the extractor.
        
        Args:
            features_file: Path to the features CSV file
            pre_window_minutes: Minutes before news to include
            post_window_minutes: Minutes after news to include
            min_bars_required: Minimum number of bars required in each window
            lockout_minutes: Lockout period to avoid overlapping events
        """
        self.features_file = features_file
        self.pre_window_minutes = pre_window_minutes
        self.post_window_minutes = post_window_minutes
        self.min_bars_required = min_bars_required
        self.lockout_minutes = lockout_minutes
        
        self.df = None
        self.events = []
        
    def load_data(self) -> None:
        """Load the features dataset."""
        print(f"Loading data from {self.features_file}...")
        self.df = pd.read_csv(self.features_file)
        
        # Ensure timestamp is datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], utc=True)
            self.df = self.df.set_index('timestamp')
        else:
            self.df.index = pd.to_datetime(self.df.index, utc=True)
        
        self.df = self.df.sort_index()
        
        print(f"Loaded {len(self.df):,} rows")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        
    def identify_news_events(self) -> pd.DataFrame:
        """
        Identify news events where news_age_minutes < 0 (news just published).
        
        Returns:
            DataFrame with news events
        """
        print("\nIdentifying news events...")
        
        # Check if news_age_minutes column exists
        if 'news_age_minutes' not in self.df.columns:
            raise ValueError("Column 'news_age_minutes' not found in dataset")
        
        # Filter for fresh news (age < 0 means news just appeared within this minute)
        # We use < 1.0 to catch news that appeared in the current minute
        fresh_news_mask = self.df['news_age_minutes'] < 1.0
        
        news_events = self.df[fresh_news_mask].copy()
        
        print(f"Found {len(news_events):,} rows with fresh news")
        
        # Get unique news IDs to avoid counting the same news multiple times
        if 'news_id' in news_events.columns:
            unique_news = news_events['news_id'].nunique()
            print(f"Unique news items: {unique_news:,}")
        
        return news_events
    
    def extract_event_window(
        self, 
        event_time: pd.Timestamp
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Extract pre- and post-window for a single event.
        
        Args:
            event_time: Timestamp of the news event
            
        Returns:
            Tuple of (pre_window_df, post_window_df, metadata_dict)
        """
        # Define time windows
        pre_start = event_time - timedelta(minutes=self.pre_window_minutes)
        pre_end = event_time
        
        post_start = event_time
        post_end = event_time + timedelta(minutes=self.post_window_minutes)
        
        # Extract windows
        pre_window = self.df.loc[pre_start:pre_end]
        post_window = self.df.loc[post_start:post_end]
        
        # Exclude the event_time itself from pre_window (it's in post_window)
        pre_window = pre_window[pre_window.index < event_time]
        
        # Calculate metadata
        metadata = {
            'event_time': event_time,
            'pre_bars': len(pre_window),
            'post_bars': len(post_window),
            'pre_coverage_pct': len(pre_window) / self.pre_window_minutes * 100,
            'post_coverage_pct': len(post_window) / self.post_window_minutes * 100,
            'pre_start': pre_window.index.min() if len(pre_window) > 0 else None,
            'pre_end': pre_window.index.max() if len(pre_window) > 0 else None,
            'post_start': post_window.index.min() if len(post_window) > 0 else None,
            'post_end': post_window.index.max() if len(post_window) > 0 else None,
        }
        
        # Check for gaps in the windows
        if len(pre_window) > 1:
            pre_gaps = pre_window.index.to_series().diff()
            max_pre_gap = pre_gaps.max().total_seconds() / 60.0
            metadata['max_pre_gap_minutes'] = max_pre_gap
        else:
            metadata['max_pre_gap_minutes'] = 0
            
        if len(post_window) > 1:
            post_gaps = post_window.index.to_series().diff()
            max_post_gap = post_gaps.max().total_seconds() / 60.0
            metadata['max_post_gap_minutes'] = max_post_gap
        else:
            metadata['max_post_gap_minutes'] = 0
        
        return pre_window, post_window, metadata
    
    def apply_lockout(self, news_events: pd.DataFrame) -> pd.DataFrame:
        """
        Apply lockout strategy: only keep first news if multiple occur within lockout period.
        
        Args:
            news_events: DataFrame with all news events
            
        Returns:
            Filtered DataFrame with lockout applied
        """
        print(f"\nApplying {self.lockout_minutes}-minute lockout strategy...")
        
        filtered_events = []
        last_event_time = None
        
        for timestamp, row in news_events.iterrows():
            if last_event_time is None:
                # First event
                filtered_events.append(timestamp)
                last_event_time = timestamp
            else:
                # Check if enough time has passed
                time_since_last = (timestamp - last_event_time).total_seconds() / 60.0
                
                if time_since_last >= self.lockout_minutes:
                    filtered_events.append(timestamp)
                    last_event_time = timestamp
        
        result = news_events.loc[filtered_events]
        
        print(f"Events before lockout: {len(news_events):,}")
        print(f"Events after lockout: {len(result):,}")
        print(f"Events removed: {len(news_events) - len(result):,}")
        
        return result
    
    def extract_all_events(self) -> List[Dict]:
        """
        Extract event windows for all news events.
        
        Returns:
            List of event dictionaries with metadata
        """
        # Identify news events
        news_events = self.identify_news_events()
        
        # Apply lockout strategy
        news_events = self.apply_lockout(news_events)
        
        print(f"\nExtracting event windows for {len(news_events):,} events...")
        
        valid_events = []
        invalid_events = []
        
        for i, (event_time, event_row) in enumerate(news_events.iterrows()):
            # Extract windows
            pre_window, post_window, metadata = self.extract_event_window(event_time)
            
            # Check data quality with SEPARATE requirements
            # Pre-window: 20min → need ~15 bars (75%)
            # Post-window: 60min → need ~36 bars (60%)
            min_pre_bars = int(self.pre_window_minutes * 0.75)  # ~15 for 20min
            min_post_bars = self.min_bars_required  # ~36 for 60min
            
            is_valid = (
                metadata['pre_bars'] >= min_pre_bars and
                metadata['post_bars'] >= min_post_bars
            )
            
            # Add event metadata
            event_data = {
                'event_id': i,
                'event_time': event_time,
                'news_id': event_row.get('news_id', None),
                'news_headline': event_row.get('news_headline', None),
                'news_sentiment': event_row.get('last_news_sentiment', None),
                'is_valid': is_valid,
                **metadata
            }
            
            if is_valid:
                valid_events.append(event_data)
            else:
                invalid_events.append(event_data)
            
            # Progress
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1:,} / {len(news_events):,} events")
        
        print(f"\nValid events: {len(valid_events):,}")
        print(f"Invalid events (insufficient data): {len(invalid_events):,}")
        
        self.events = valid_events
        return valid_events
    
    def save_event_metadata(self, output_file: str) -> None:
        """
        Save event metadata to CSV.
        
        Args:
            output_file: Path to output CSV file
        """
        if not self.events:
            print("No events to save. Run extract_all_events() first.")
            return
        
        events_df = pd.DataFrame(self.events)
        events_df.to_csv(output_file, index=False)
        
        print(f"\nSaved {len(events_df)} events to {output_file}")
        
        # Print summary statistics
        print("\nEvent Summary:")
        print(f"Average pre-window coverage: {events_df['pre_coverage_pct'].mean():.1f}%")
        print(f"Average post-window coverage: {events_df['post_coverage_pct'].mean():.1f}%")
        print(f"Average pre-window bars: {events_df['pre_bars'].mean():.1f}")
        print(f"Average post-window bars: {events_df['post_bars'].mean():.1f}")
        print(f"Max pre-window gap: {events_df['max_pre_gap_minutes'].max():.1f} minutes")
        print(f"Max post-window gap: {events_df['max_post_gap_minutes'].max():.1f} minutes")


def main():
    """Main execution function."""
    # Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    
    features_file = os.path.join(data_dir, 'nasdaq100_index_1m_features.csv')
    output_file = os.path.join(data_dir, 'news_events_metadata.csv')
    
    # Create extractor
    extractor = EventWindowExtractor(
        features_file=features_file,
        pre_window_minutes=20,  # Keep pre-window same (features are pre-event)
        post_window_minutes=60,  # CHANGED: 20 → 60 for longer horizon
        min_bars_required=36,    # CHANGED: 45 → 36 (~60% coverage for realistic market gaps)
        lockout_minutes=40      # Keep lockout same
    )
    
    # Extract events
    extractor.load_data()
    extractor.extract_all_events()
    extractor.save_event_metadata(output_file)
    
    print("\nEvent extraction complete!")


if __name__ == "__main__":
    main()
