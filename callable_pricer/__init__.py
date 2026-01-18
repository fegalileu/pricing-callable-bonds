"""Callable bond pricer package (QuantLib).

This package provides:
- Market loaders (curve + volatility surfaces)
- Model calibrations (Hull-White, CIR, Black-Karasinski)
- Manual pricing engines (HW LSMC, CIR PDE, KWF-style tree)
- Orchestrator for price + effective duration/convexity

The implementation is designed to be reproducible and easy to adapt for academic work.
"""

from .config import AppConfig
from .instruments import CallableBondSpec
from .market import MarketLoader
from .calibration import Calibrator
from .pricer import MasterPricer
