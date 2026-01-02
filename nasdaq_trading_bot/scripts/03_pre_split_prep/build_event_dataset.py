"""
Event Dataset Builder - Combines Features and Targets

This script:
1. Loads event metadata
2. Builds features from pre-windows
3. Calculates targets from post-windows
4. Creates final event-based dataset for ML training

Output: event_dataset.csv with one row per news event
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from event_features import EventFeatureBuilder
from event_targets import EventTargetBuilder


def build_event_dataset(
    features_file: str,
    events_file: str,
    output_file: str,
    pre_window_minutes: int = 20,
    post_window_minutes: int = 20
):
    """
    Build complete event dataset with features and targets.
    
    Args:
        features_file: Path to full features CSV
        events_file: Path to event metadata CSV
        output_file: Path to output event dataset CSV
        pre_window_minutes: Pre-window size
        post_window_minutes: Post-window size
    """
    print("="*80)
    print("EVENT DATASET BUILDER")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    features_df = pd.read_csv(features_file)
    events_df = pd.read_csv(events_file)
    
    print(f"  Loaded {len(features_df):,} feature rows")
    print(f"  Loaded {len(events_df):,} events")
    
    # Initialize builders
    print("\nInitializing feature and target builders...")
    feature_builder = EventFeatureBuilder(features_df, pre_window_minutes)
    target_builder = EventTargetBuilder(features_df, post_window_minutes)
    
    # Process each event
    print(f"\nProcessing {len(events_df):,} events...")
    
    event_data = []
    errors = []
    
    for idx, event_row in tqdm(events_df.iterrows(), total=len(events_df), desc="Building dataset"):
        try:
            event_time = pd.to_datetime(event_row['event_time'], utc=True)
            
            # Build features
            features = feature_builder.build_all_features(
                event_time=event_time,
                news_sentiment=event_row.get('news_sentiment', None)
            )
            
            # Build targets
            targets = target_builder.build_all_targets(event_time=event_time)
            
            # Combine into single record
            record = {
                'event_id': event_row['event_id'],
                'event_time': event_time,
                'news_id': event_row.get('news_id', None),
                'news_headline': event_row.get('news_headline', None),
                **features,
                **targets
            }
            
            event_data.append(record)
            
        except Exception as e:
            errors.append({
                'event_id': event_row.get('event_id', idx),
                'error': str(e)
            })
    
    # Create dataframe
    print(f"\nCreating dataset...")
    dataset_df = pd.DataFrame(event_data)
    
    print(f"  Successfully processed: {len(dataset_df):,} events")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print("\nSome events had errors:")
        for err in errors[:5]:  # Show first 5 errors
            print(f"  Event {err['event_id']}: {err['error']}")
    
    # Data quality check
    print(f"\nDataset Quality Check:")
    print(f"  Total events: {len(dataset_df):,}")
    print(f"  Total features + targets: {len(dataset_df.columns)}")
    
    # Check for NaN in primary target
    primary_target = 'target_vwap_return_20m'
    nan_targets = dataset_df[primary_target].isna().sum()
    print(f"  NaN in primary target ({primary_target}): {nan_targets} ({nan_targets/len(dataset_df)*100:.1f}%)")
    
    # Target distribution
    if nan_targets < len(dataset_df):
        target_stats = dataset_df[primary_target].describe()
        print(f"\nPrimary Target Distribution ({primary_target}):")
        print(f"  Mean: {target_stats['mean']:.4f}%")
        print(f"  Std: {target_stats['std']:.4f}%")
        print(f"  Min: {target_stats['min']:.4f}%")
        print(f"  Max: {target_stats['max']:.4f}%")
        print(f"  Median: {target_stats['50%']:.4f}%")
    
    # Save dataset
    print(f"\nSaving dataset to {output_file}...")
    dataset_df.to_csv(output_file, index=False)
    
    print(f"\nDataset saved with {len(dataset_df)} rows and {len(dataset_df.columns)} columns")
    
    # Print column summary
    print(f"\nColumn Summary:")
    
    feature_cols = [col for col in dataset_df.columns if col.startswith('pre_') or col in ['news_sentiment', 'news_hour', 'news_minute', 'news_day_of_week', 'minutes_since_market_open']]
    target_cols = [col for col in dataset_df.columns if col.startswith('target_')]
    metadata_cols = ['event_id', 'event_time', 'news_id', 'news_headline']
    
    print(f"  Metadata columns: {len(metadata_cols)}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Target columns: {len(target_cols)}")
    
    # Show feature names
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"    - {col}")
    
    print(f"\nTarget columns ({len(target_cols)}):")
    for col in target_cols:
        print(f"    - {col}")
    
    print("\n" + "="*80)
    print("EVENT DATASET BUILD COMPLETE!")
    print("="*80)
    
    return dataset_df


def main():
    """Main execution."""
    # Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    
    # Use raw OHLCV data file instead of features file
    features_file = os.path.join(data_dir, 'nasdaq100_index_1m.csv')  # CHANGED: use raw data
    events_file = os.path.join(data_dir, 'news_events_metadata.csv')
    output_file = os.path.join(data_dir, 'event_dataset.csv')
    
    print(f"Using raw data file: {features_file}")
    
    # Build dataset
    dataset_df = build_event_dataset(
        features_file=features_file,
        events_file=events_file,
        output_file=output_file,
        pre_window_minutes=20,
        post_window_minutes=60  # CHANGED: 20 → 60 for longer horizon
    )
    
    # Additional analysis
    print(f"\nAdditional Dataset Statistics:")
    
    # Directional distribution
    if 'target_direction' in dataset_df.columns:
        direction_counts = dataset_df['target_direction'].value_counts()
        print(f"\nDirection Distribution:")
        print(f"  Bullish (1): {direction_counts.get(1, 0)} ({direction_counts.get(1, 0)/len(dataset_df)*100:.1f}%)")
        print(f"  Neutral (0): {direction_counts.get(0, 0)} ({direction_counts.get(0, 0)/len(dataset_df)*100:.1f}%)")
        print(f"  Bearish (-1): {direction_counts.get(-1, 0)} ({direction_counts.get(-1, 0)/len(dataset_df)*100:.1f}%)")
    
    # Profitable events
    if 'target_profitable_1pct' in dataset_df.columns:
        profitable = dataset_df['target_profitable_1pct'].sum()
        print(f"\nProfitable Events (>1%):")
        print(f"  Count: {profitable} ({profitable/len(dataset_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
