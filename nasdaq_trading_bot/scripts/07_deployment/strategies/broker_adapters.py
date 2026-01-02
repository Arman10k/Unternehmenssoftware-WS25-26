"""
Broker Adapters for Trading System
===================================
Provides unified interface for different brokers (Alpaca for stocks/ETFs, OANDA for CFDs).
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import os
import requests

from .strategy_config import BrokerInterface


class AlpacaBroker(BrokerInterface):
    """Alpaca broker adapter for stocks and ETFs (Paper + Live trading)"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = None, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        
        if base_url:
            self.base_url = base_url
        elif paper:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        
        self.paper = paper
        print(f"[ALPACA] Initialized {'Paper' if paper else 'Live'} trading")
    
    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    
    def get_account_info(self) -> dict:
        r = requests.get(f"{self.base_url}/v2/account", headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    
    def get_positions(self) -> List[dict]:
        r = requests.get(f"{self.base_url}/v2/positions", headers=self._headers(), timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()
    
    def get_position(self, symbol: str) -> Optional[dict]:
        r = requests.get(f"{self.base_url}/v2/positions/{symbol}", headers=self._headers(), timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    
    def submit_order(self, order_params: dict) -> Optional[dict]:
        try:
            r = requests.post(f"{self.base_url}/v2/orders", headers=self._headers(), 
                            json=order_params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ALPACA ERROR] submit_order failed: {e}")
            return None
    
    def submit_bracket_order(self, symbol: str, qty: int, side: str, 
                            sl_price: float, tp_price: float) -> Optional[dict]:
        """Submit a bracket order (market order with stop-loss and take-profit)"""
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "order_class": "bracket",
            "take_profit": {"limit_price": f"{tp_price:.2f}"},
            "stop_loss": {"stop_price": f"{sl_price:.2f}"},
        }
        result = self.submit_order(payload)
        if result:
            print(f"[ALPACA ORDER] {side.upper()} {qty} {symbol} | SL={sl_price:.2f} TP={tp_price:.2f}")
        return result
    
    def close_position(self, symbol: str) -> bool:
        try:
            r = requests.delete(f"{self.base_url}/v2/positions/{symbol}", 
                              headers=self._headers(), timeout=30)
            r.raise_for_status()
            print(f"[ALPACA] Closed position: {symbol}")
            return True
        except Exception as e:
            print(f"[ALPACA ERROR] close_position failed: {e}")
            return False
    
    def get_last_fill_time(self, symbol: str, side: str) -> Optional[datetime]:
        params = {"status": "closed", "limit": "200", "direction": "desc", "nested": "false"}
        try:
            r = requests.get(f"{self.base_url}/v2/orders", headers=self._headers(), 
                           params=params, timeout=30)
            r.raise_for_status()
            orders = r.json()
        except Exception as e:
            print(f"[ALPACA WARN] get_last_fill_time failed: {e}")
            return None
        
        last_dt = None
        for o in orders:
            if o.get("status", "").lower() != "filled":
                continue
            if o.get("symbol", "").upper() != symbol.upper():
                continue
            if o.get("side", "").lower() != side.lower():
                continue
            
            filled_at = o.get("filled_at")
            if not filled_at:
                continue
            
            try:
                dt = datetime.fromisoformat(str(filled_at).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                    
                if last_dt is None or dt > last_dt:
                    last_dt = dt
            except Exception:
                continue
        
        return last_dt
    
    def get_calendar(self, start_date: str, end_date: str) -> List[dict]:
        """Get trading calendar"""
        params = {"start": start_date, "end": end_date}
        r = requests.get(f"{self.base_url}/v2/calendar", headers=self._headers(), 
                        params=params, timeout=30)
        r.raise_for_status()
        return r.json()


class OANDABroker(BrokerInterface):
    """OANDA broker adapter for CFD trading (Practice + Live)"""
    
    def __init__(self, api_key: str, account_id: str, practice: bool = True):
        self.api_key = api_key
        self.account_id = account_id
        
        if practice:
            self.base_url = "https://api-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
        
        self.practice = practice
        print(f"[OANDA] Initialized {'Practice' if practice else 'Live'} trading")
        print(f"[OANDA] Account ID: {account_id}")
    
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    
    def get_account_info(self) -> dict:
        r = requests.get(f"{self.base_url}/v3/accounts/{self.account_id}", 
                        headers=self._headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        account = data.get("account", {})
        
        # Map to common format
        return {
            "equity": float(account.get("NAV", 0)),
            "cash": float(account.get("balance", 0)),
            "margin_available": float(account.get("marginAvailable", 0)),
            "margin_used": float(account.get("marginUsed", 0)),
            "unrealized_pl": float(account.get("unrealizedPL", 0)),
        }
    
    def get_positions(self) -> List[dict]:
        r = requests.get(f"{self.base_url}/v3/accounts/{self.account_id}/openPositions", 
                        headers=self._headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        
        positions = []
        for pos in data.get("positions", []):
            # OANDA has long and short separately
            long_units = int(pos.get("long", {}).get("units", 0))
            short_units = int(pos.get("short", {}).get("units", 0))
            
            if long_units != 0:
                positions.append({
                    "symbol": pos.get("instrument"),
                    "qty": long_units,
                    "side": "long",
                    "avg_entry_price": float(pos.get("long", {}).get("averagePrice", 0)),
                    "unrealized_pl": float(pos.get("long", {}).get("unrealizedPL", 0)),
                })
            if short_units != 0:
                positions.append({
                    "symbol": pos.get("instrument"),
                    "qty": abs(short_units),
                    "side": "short",
                    "avg_entry_price": float(pos.get("short", {}).get("averagePrice", 0)),
                    "unrealized_pl": float(pos.get("short", {}).get("unrealizedPL", 0)),
                })
        
        return positions
    
    def get_position(self, symbol: str) -> Optional[dict]:
        positions = self.get_positions()
        for pos in positions:
            if pos.get("symbol") == symbol:
                return pos
        return None
    
    def submit_order(self, order_params: dict) -> Optional[dict]:
        """Submit an order to OANDA"""
        symbol = order_params.get("symbol")
        units = order_params.get("qty", order_params.get("units", 0))
        side = order_params.get("side", "buy")
        
        # OANDA uses negative units for sell/short
        if side in ("sell", "short"):
            units = -abs(units)
        
        oanda_order = {
            "order": {
                "type": "MARKET",
                "instrument": symbol,
                "units": str(units),
                "timeInForce": "FOK",  # Fill or Kill
            }
        }
        
        # Add stop-loss if provided
        if "stop_loss" in order_params:
            sl = order_params["stop_loss"]
            oanda_order["order"]["stopLossOnFill"] = {
                "price": str(sl.get("stop_price", sl.get("price")))
            }
        
        # Add take-profit if provided
        if "take_profit" in order_params:
            tp = order_params["take_profit"]
            oanda_order["order"]["takeProfitOnFill"] = {
                "price": str(tp.get("limit_price", tp.get("price")))
            }
        
        try:
            r = requests.post(f"{self.base_url}/v3/accounts/{self.account_id}/orders",
                            headers=self._headers(), json=oanda_order, timeout=30)
            r.raise_for_status()
            result = r.json()
            print(f"[OANDA ORDER] {side.upper()} {abs(units)} {symbol}")
            return result
        except Exception as e:
            print(f"[OANDA ERROR] submit_order failed: {e}")
            return None
    
    def close_position(self, symbol: str) -> bool:
        """Close all positions for a symbol"""
        try:
            # Close long positions
            r = requests.put(
                f"{self.base_url}/v3/accounts/{self.account_id}/positions/{symbol}/close",
                headers=self._headers(),
                json={"longUnits": "ALL"},
                timeout=30
            )
            
            # Close short positions
            r2 = requests.put(
                f"{self.base_url}/v3/accounts/{self.account_id}/positions/{symbol}/close",
                headers=self._headers(),
                json={"shortUnits": "ALL"},
                timeout=30
            )
            
            print(f"[OANDA] Closed position: {symbol}")
            return True
        except Exception as e:
            print(f"[OANDA ERROR] close_position failed: {e}")
            return False
    
    def get_last_fill_time(self, symbol: str, side: str) -> Optional[datetime]:
        """Get the last fill time for a symbol and side"""
        try:
            r = requests.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/trades",
                headers=self._headers(),
                params={"instrument": symbol, "state": "ALL", "count": 50},
                timeout=30
            )
            r.raise_for_status()
            trades = r.json().get("trades", [])
            
            for trade in trades:
                trade_side = "buy" if int(trade.get("currentUnits", 0)) > 0 else "sell"
                if trade_side == side:
                    open_time = trade.get("openTime")
                    if open_time:
                        return datetime.fromisoformat(open_time.replace("Z", "+00:00"))
            
            return None
        except Exception as e:
            print(f"[OANDA WARN] get_last_fill_time failed: {e}")
            return None
    
    def get_candles(self, symbol: str, granularity: str = "M1", count: int = 500) -> List[dict]:
        """Get candlestick data from OANDA"""
        r = requests.get(
            f"{self.base_url}/v3/instruments/{symbol}/candles",
            headers=self._headers(),
            params={"granularity": granularity, "count": count, "price": "M"},
            timeout=30
        )
        r.raise_for_status()
        return r.json().get("candles", [])


def create_broker(api_type: str, keys: dict) -> BrokerInterface:
    """
    Factory function to create the appropriate broker based on api_type.
    
    Args:
        api_type: "alpaca" or "oanda"
        keys: Dictionary containing API keys
        
    Returns:
        BrokerInterface instance
    """
    if api_type == "alpaca":
        api_key = keys.get("APCA-API-KEY-ID-Paper") or keys.get("ALPACA_KEY_ID")
        secret = keys.get("APCA-API-SECRET-KEY-Paper") or keys.get("ALPACA_SECRET")
        base_url = keys.get("ALPACA_BASE", "https://paper-api.alpaca.markets")
        
        if not api_key or not secret:
            raise ValueError("Missing Alpaca API keys. Set in conf/keys.yaml")
        
        return AlpacaBroker(api_key, secret, base_url, paper=True)
    
    elif api_type == "oanda":
        api_key = keys.get("OANDA_API_KEY")
        account_id = keys.get("OANDA_ACCOUNT_ID")
        
        if not api_key or not account_id:
            raise ValueError(
                "Missing OANDA API keys. Add to conf/keys.yaml:\n"
                "  OANDA_API_KEY: your-api-key\n"
                "  OANDA_ACCOUNT_ID: your-account-id\n\n"
                "Get your Practice account at: https://www.oanda.com/demo-account/"
            )
        
        return OANDABroker(api_key, account_id, practice=True)
    
    else:
        raise ValueError(f"Unknown api_type: {api_type}. Supported: alpaca, oanda")
