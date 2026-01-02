"""
Trading Strategies Module
==========================
Exports all strategy components for use in deployment scripts.
"""

from .strategy_config import (
    StrategyConfig,
    TradingStrategy,
    BrokerInterface,
    load_all_strategies,
    list_available_strategies,
)

from .strategies import (
    LongOnlyMomentumStrategy,
    ShortOnlyStrategy,
    CFDLeveragedStrategy,
)

from .broker_adapters import (
    AlpacaBroker,
    OANDABroker,
    create_broker,
)

__all__ = [
    # Config
    "StrategyConfig",
    "TradingStrategy", 
    "BrokerInterface",
    "load_all_strategies",
    "list_available_strategies",
    # Strategies
    "LongOnlyMomentumStrategy",
    "ShortOnlyStrategy",
    "CFDLeveragedStrategy",
    # Brokers
    "AlpacaBroker",
    "OANDABroker",
    "create_broker",
]
