"""Strategy interface. A strategy maps a price history to target weights on rebalance dates."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Strategy(ABC):
    """Subclass and implement ``target_weights``.

    ``params`` holds every tunable knob so the robustness tools can vary them generically.
    """

    name: str = "strategy"
    params: dict = field(default_factory=dict)

    @abstractmethod
    def target_weights(self, prices: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Return a DataFrame indexed by rebalance_dates, columns = tickers, values = target weight.

        Only information available at or before each rebalance date may be used.
        Rows must sum to <= 1.0; the remainder is cash earning ``cash_rate``.
        """

    @property
    def tickers(self) -> list[str]:
        """Every ticker the strategy may trade or read (override if different)."""
        return list(self.params.get("universe", []))

    def with_params(self, **overrides) -> "Strategy":
        new = {**self.params, **overrides}
        return type(self)(name=self.name, params=new)

    def describe(self) -> dict:
        return {"name": self.name, "params": self.params}
