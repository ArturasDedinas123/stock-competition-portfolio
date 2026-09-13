"""Competition rules and model settings shared by the search and the backtest."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategySettings:
    """Rules of the competition and the knobs of the simulation model.

    Attributes:
        horizon: Trading days between the purchase close and the final close.
        min_weight: Smallest allowed weight per stock.
        max_weight: Largest allowed weight per stock.
        n_rivals: Number of rival portfolios in the competition.
        block_days: Length of the consecutive-day blocks used by the bootstrap.
        equity_premium: Expected annual return of the market above the risk-free rate.
        momentum_tilt: Extra annual expected return per standard deviation of momentum.
        risk_free_fallback: Annual risk-free rate used when the T-bill yield is unavailable.
        half_life_days: Weight recent history more (half-life in trading days); None weights all days equally.
    """

    horizon: int
    min_weight: float = 0.05
    max_weight: float = 0.20
    n_rivals: int = 20
    block_days: int = 21
    equity_premium: float = 0.05
    momentum_tilt: float = 0.04
    risk_free_fallback: float = 0.04
    half_life_days: float | None = None
