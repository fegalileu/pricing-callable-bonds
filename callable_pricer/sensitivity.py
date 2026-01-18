"""Sensitivity sweeps for the callable bond price.

These utilities are meant for *reporting* / dissertation figures (Chapter 4).
They run deterministic sweeps while keeping the pricing logic identical to the
validated prototype.

The three standard sweeps implemented here are:

1) Price vs volatility (model parameter sigma)
2) Price vs risk-free parallel shift
3) Price vs credit spread (OAS)

The functions return ``pandas.DataFrame`` objects in a *wide* format: the first
column is the x-axis, and each additional column is a method label.
"""

import pandas as pd
import QuantLib as ql


def _scale_sigma(params, multiplier):
    """Return a copy of params with sigma scaled, if sigma exists."""
    if params is None:
        return None
    p = dict(params)
    if "sigma" in p and p["sigma"] is not None:
        p["sigma"] = float(params["sigma"]) * float(multiplier)
    return p


def price_vs_volatility(pricer, scenarios, oas_decimal, vol_multipliers):
    """Compute callable bond price sensitivity to the vol parameter.

    Parameters
    ----------
    pricer : callable_pricer.pricer.MasterPricer
    scenarios : list[tuple]
        List of tuples: (label, params_dict_or_None, method_code).
    oas_decimal : float
        Credit spread (OAS) in decimal terms (e.g. 90bps -> 0.0090).
    vol_multipliers : iterable[float]
        Multiplicative bumps applied to the model sigma parameter.
    """
    rows = []
    for m in vol_multipliers:
        row = {"vol_multiplier": float(m)}
        for label, params, method in scenarios:
            p = _scale_sigma(params, m)
            price, _ = pricer.calculate(p, method, oas_decimal, state_cache=None)
            row[label] = float(price)
        rows.append(row)
    return pd.DataFrame(rows)


def price_vs_rate_shift(pricer, scenarios, oas_decimal, rate_shifts_bps):
    """Compute price sensitivity to a parallel shift of the risk-free curve.

    The shift is applied to the *risk-free* curve used as the base term
    structure. The OAS is then applied on top of the shifted curve.
    """
    base_ptr = pricer.ts_base.currentLink()
    rows = []
    try:
        for bps in rate_shifts_bps:
            shift = float(bps) / 10000.0
            # Build a shifted curve on top of the current base_ptr
            ts_shift = ql.ZeroSpreadedTermStructure(
                ql.YieldTermStructureHandle(base_ptr),
                ql.QuoteHandle(ql.SimpleQuote(shift)),
            )
            ts_shift.enableExtrapolation()
            pricer.ts_base.linkTo(ts_shift)

            row = {"rate_shift_bps": float(bps)}
            for label, params, method in scenarios:
                price, _ = pricer.calculate(params, method, oas_decimal, state_cache=None)
                row[label] = float(price)
            rows.append(row)
    finally:
        pricer.ts_base.linkTo(base_ptr)

    return pd.DataFrame(rows)


def price_vs_oas(pricer, scenarios, oas_grid_bps):
    """Compute price sensitivity to the credit spread (OAS)."""
    rows = []
    for bps in oas_grid_bps:
        oas_decimal = float(bps) / 10000.0
        row = {"oas_bps": float(bps)}
        for label, params, method in scenarios:
            price, _ = pricer.calculate(params, method, oas_decimal, state_cache=None)
            row[label] = float(price)
        rows.append(row)
    return pd.DataFrame(rows)
