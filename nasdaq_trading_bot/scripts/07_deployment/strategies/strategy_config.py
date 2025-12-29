"""
Trading Strategy Configuration System
======================================
Defines base classes and interfaces for modular trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
import json
import os


@dataclass
class StrategyConfig:
    """Base configuration for all strategies"""
    name: str
    strategy_type: str  # "long_only", "short_only", "long_short", "cfd_leveraged"
    api_type: str  # "alpaca" or "oanda"
    ticker: str  # Trading symbol (e.g., "QQQ" for Alpaca, "NAS100_USD" for OANDA)

    # Entry/Exit thresholds
    entry_threshold: float
    exit_threshold: float

    # Position sizing
    position_size_pct: float
    max_positions: int

    # Risk management
    stop_loss_pct: float
    take_profit_pct: float

    # Timing
    min_hold_minutes: int
    max_hold_minutes: int
    cooldown_minutes: int

    # Leverage (for CFDs)
    leverage: float = 1.0

    # Additional params
    use_trailing_stop: bool = False
    trailing_stop_pct: Optional[float] = None

    # Market conditions
    min_volume: Optional[float] = None
    min_volatility: Optional[float] = None
    max_volatility: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'StrategyConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, filepath: str) -> 'StrategyConfig':
        """Load strategy config from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, filepath: str):
        """Save strategy config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


class TradingStrategy(ABC):
    """Abstract base class for all trading strategies"""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    def calc_signal(self, prediction: dict, market_data: dict) -> Tuple[float, dict]:
        """
        Calculate trading signal from model prediction and market data.

        Returns:
            signal (float): Signal strength (-1 to 1, where >0 = buy, <0 = sell)
            metadata (dict): Additional signal information
        """
        pass

    @abstractmethod
    def can_enter(self, symbol: str, signal: float, metadata: dict,
                  broker_interface) -> Tuple[bool, str]:
        """
        Check if entry conditions are met.

        Returns:
            can_enter (bool): Whether to enter position
            reason (str): Explanation
        """
        pass

    @abstractmethod
    def should_exit(self, symbol: str, position: dict, signal: float,
                    metadata: dict, broker_interface) -> Tuple[bool, str]:
        """
        Check if exit conditions are met.

        Returns:
            should_exit (bool): Whether to exit position
            reason (str): Explanation
        """
        pass

    @abstractmethod
    def calculate_position_size(self, symbol: str, signal: float,
                                account_equity: float, current_price: float) -> int:
        """Calculate position size based on strategy rules"""
        pass

    @abstractmethod
    def get_order_params(self, symbol: str, qty: int, side: str,
                         current_price: float) -> dict:
        """
        Get broker-specific order parameters.

        Returns:
            dict with order details (type, limit_price, stop_loss, take_profit, etc.)
        """
        pass


class BrokerInterface(ABC):
    """Abstract interface for broker interactions"""

    @abstractmethod
    def get_account_info(self) -> dict:
        pass

    @abstractmethod
    def get_positions(self) -> List[dict]:
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[dict]:
        pass

    @abstractmethod
    def submit_order(self, order_params: dict) -> Optional[dict]:
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> bool:
        pass

    @abstractmethod
    def get_last_fill_time(self, symbol: str, side: str) -> Optional[datetime]:
        pass


def load_all_strategies(config_path: str = None) -> Dict[str, StrategyConfig]:
    """
    Load all strategies from strategies.json
    
    Returns:
        Dict mapping strategy_id -> StrategyConfig
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "strategies.json")
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    strategies = {}
    for strategy_id, strategy_data in data.items():
        try:
            strategies[strategy_id] = StrategyConfig.from_dict(strategy_data)
        except Exception as e:
            print(f"[WARN] Could not load strategy '{strategy_id}': {e}")
    
    return strategies


def list_available_strategies(config_path: str = None) -> None:
    """Print all available strategies"""
    strategies = load_all_strategies(config_path)
    
    print("\n" + "=" * 60)
    print("AVAILABLE STRATEGIES")
    print("=" * 60)
    
    for sid, cfg in strategies.items():
        api_badge = "[ALPACA]" if cfg.api_type == "alpaca" else "[OANDA]"
        print(f"\n  [{sid}]")
        print(f"    Name:     {cfg.name}")
        print(f"    Type:     {cfg.strategy_type}")
        print(f"    API:      {api_badge}")
        print(f"    Ticker:   {cfg.ticker}")
        print(f"    Leverage: {cfg.leverage}x")
    
    print("\n" + "=" * 60)