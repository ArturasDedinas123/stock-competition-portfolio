"""Simulation-based portfolio construction for a winner-take-all stock competition.

Modules
-------
data            Prices, analyst targets and earnings dates from Yahoo Finance (cached per day)
stats           Descriptive statistics: beta, drawdown, momentum, correlation
scenarios       Block-bootstrap simulation of horizon returns under several expected-return views
rivals          Simulated rival portfolios and the return it takes to beat them
search          Exhaustive weight-grid search, fine-tuning and portfolio evaluation
backtest        Walk-forward test of the strategy on past quarters
charts          Shared chart style
"""

__version__ = "1.0.0"
