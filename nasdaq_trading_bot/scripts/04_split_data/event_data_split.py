"""
Step 4: Event Data Split
----------------------------------------
Splits the event dataset temporally (70/15/15)
Saves: event_train.csv, event_validation.csv, event_test.csv
"""

import os
import pandas as pd


def split_event_data(event_dataset_file, data_dir, train_ratio=0.70, validation_ratio=0.15):
    """
    Split event dataset temporally.
    
    Args:
        event_dataset_file: Path to event_dataset.csv
        data_dir: Output directory
        train_ratio: Training set ratio (default 70%)
        validation_ratio: Validation set ratio (default 15%)
    """
    print("="*80)
    print("EVENT DATA TEMPORAL SPLIT")
    print("="*80)
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Load event dataset
    print(f"\nLoading {event_dataset_file}...")
    df = pd.read_csv(event_dataset_file)
    print(f"  Loaded {len(df):,} events")
    
    # Convert event_time to datetime and sort
    df['event_time'] = pd.to_datetime(df['event_time'], utc=True)
    df = df.sort_values('event_time').reset_index(drop=True)
    
    print(f"  Date range: {df['event_time'].min()} to {df['event_time'].max()}")
    
    # Calculate split indices
    df_length = len(df)
    train_end = int(train_ratio * df_length)
    validation_end = int((train_ratio + validation_ratio) * df_length)
    
    # Temporal split (no shuffling!)
    train = df.iloc[:train_end]
    validation = df.iloc[train_end:validation_end]
    test = df.iloc[validation_end:]
    
    print(f"\nTemporal Split (Chronological):")
    print(f"  Train: {len(train):,} events ({len(train)/df_length*100:.1f}%)")
    print(f"    Date range: {train['event_time'].min()} to {train['event_time'].max()}")
    
    print(f"  Validation: {len(validation):,} events ({len(validation)/df_length*100:.1f}%)")
    print(f"    Date range: {validation['event_time'].min()} to {validation['event_time'].max()}")
    
    print(f"  Test: {len(test):,} events ({len(test)/df_length*100:.1f}%)")
    print(f"    Date range: {test['event_time'].min()} to {test['event_time'].max()}")
    
    # Save splits with unique event names
    train_file = os.path.join(data_dir, "event_train.csv")
    validation_file = os.path.join(data_dir, "event_validation.csv")
    test_file = os.path.join(data_dir, "event_test.csv")
    
    train.to_csv(train_file, index=False)
    validation.to_csv(validation_file, index=False)
    test.to_csv(test_file, index=False)
    
    print(f"\nSaved splits:")
    print(f"  {train_file}")
    print(f"  {validation_file}")
    print(f"  {test_file}")
    
    print("\n" + "="*80)
    print("EVENT DATA SPLIT COMPLETE!")
    print("="*80)


def main():
    """Main execution."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    event_dataset_file = os.path.join(data_dir, 'event_dataset.csv')
    
    if not os.path.exists(event_dataset_file):
        print(f"Error: {event_dataset_file} not found!")
        print("   Run build_event_dataset.py first.")
        return
    
    split_event_data(
        event_dataset_file=event_dataset_file,
        data_dir=data_dir,
        train_ratio=0.70,
        validation_ratio=0.15
    )


if __name__ == "__main__":
    main()
