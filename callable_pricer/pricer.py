import QuantLib as ql

from .engines import CIRPDEEngine, HullWhiteLSMCEngine, KWFManualTreeEngine


class MasterPricer:
    """High-level orchestrator.

    Responsibilities
    ----------------
    - Apply OAS (as a static zero-spread shift) consistently for all pricing engines
    - Build QuantLib reference instruments (straight bond and callable bond)
    - Provide: price(), metrics() (effective duration/convexity via bump-and-reprice)

    Notes
    -----
    The effective risk metrics use the bump configured in ``cfg.risk_bump_bps``
    (default: same as ``cfg.bump_bps``).
    """

    def __init__(self, ts_base, bond_spec, cfg):
        """Create a pricer.

        Parameters
        ----------
        ts_base : QuantLib.RelinkableYieldTermStructureHandle
            Base risk-free curve (no OAS). This handle will be relinked during
            bump-and-reprice to build effective duration/convexity.
        bond_spec : CallableBondSpec
            Instrument definition.
        cfg : AppConfig
            Numerical knobs.
        """
        self.ts_base = ts_base
        self.bond_spec = bond_spec
        self.cfg = cfg
        self._keep_alive = []

        # QuantLib reference instruments
        self.ql_straight = bond_spec.ql_straight_bond()
        self.ql_callable = bond_spec.ql_callable_bond()

    def _make_ts_with_oas(self, oas_decimal):
        """Return a YieldTermStructureHandle with OAS applied."""
        if abs(oas_decimal) <= 1e-12:
            return ql.YieldTermStructureHandle(self.ts_base.currentLink())

        risky_obj = ql.ZeroSpreadedTermStructure(
            self.ts_base,
            ql.QuoteHandle(ql.SimpleQuote(float(oas_decimal))),
        )
        risky_obj.enableExtrapolation()

        # Keep object alive to avoid Python/QL lifetime issues.
        self._keep_alive = [risky_obj]
        return ql.YieldTermStructureHandle(risky_obj)

    def calculate(self, params, method, oas_decimal, state_cache=None):
        """Price (clean) a bond using a given method.

        Returns
        -------
        (clean_price, state_cache)
        """
        ts_use = self._make_ts_with_oas(oas_decimal)

        if method == "STRAIGHT BOND":
            self.ql_straight.setPricingEngine(ql.DiscountingBondEngine(ts_use))
            return float(self.ql_straight.cleanPrice()), None

        accrued = float(self.ql_straight.accruedAmount())

        if method == "HW_LSMC":
            dirty, sc = HullWhiteLSMCEngine(self.cfg).price(
                ts_use, self.bond_spec.to_engine_bond_data(), params, state_cache
            )
            return float(dirty - accrued), sc

        if method == "CIR_PDE":
            dirty, sc = CIRPDEEngine(self.cfg).price(
                ts_use, self.bond_spec.to_engine_bond_data(), params, state_cache
            )
            return float(dirty - accrued), sc

        if method == "KWF_MANUAL":
            dirty, sc = KWFManualTreeEngine(self.cfg).price(
                ts_use, self.bond_spec.to_engine_bond_data(), params, state_cache
            )
            return float(dirty - accrued), sc

        # QuantLib tree references
        if "QL_TREE" in method:
            try:
                if method == "HW_QL_TREE":
                    model = ql.HullWhite(ts_use, params["a"], params["sigma"])
                elif method == "CIR_QL_TREE":
                    r0 = ts_use.currentLink().zeroRate(0.001, ql.Continuous).rate()
                    model = ql.ExtendedCoxIngersollRoss(
                        ts_use, params["theta"], params["k"], params["sigma"], r0
                    )
                elif method == "KWF_QL_TREE":
                    model = ql.BlackKarasinski(ts_use, params["a"], params["sigma"])
                else:
                    return 0.0, None

                self.ql_callable.setPricingEngine(
                    ql.TreeCallableFixedRateBondEngine(model, int(self.cfg.ql_grid_size))
                )
                return float(self.ql_callable.cleanPrice()), None
            except Exception:
                return 0.0, None

        return 0.0, None

    def metrics(self, params, method, oas_decimal, state_cache=None, return_state=False):
        """Return (price, effective duration, effective convexity).

        Parameters
        ----------
        params : dict or None
            Model parameters for the chosen engine.
        method : str
            Pricing method key.
        oas_decimal : float
            OAS in decimal (e.g., 90 bps -> 0.0090).
        state_cache : dict or None
            Optional cache to reuse CRN/regression coefficients and avoid
            introducing MC noise into the bump runs.
        return_state : bool
            If True, also returns the base-run cache.
        """
        P0, state = self.calculate(params, method, oas_decimal, state_cache)
        if P0 <= 1e-8:
            if return_state:
                return 0.0, 0.0, 0.0, state
            return 0.0, 0.0, 0.0

        # Parallel bump size used for risk metrics (default 1bp, see AppConfig).
        dy = float(getattr(self.cfg, "risk_bump_bps", self.cfg.bump_bps)) / 10000.0
        if abs(dy) < 1e-12:
            if return_state:
                return float(P0), 0.0, 0.0, state
            return float(P0), 0.0, 0.0

        base_ptr = self.ts_base.currentLink()
        try:
            ts_up = ql.ZeroSpreadedTermStructure(
                ql.YieldTermStructureHandle(base_ptr),
                ql.QuoteHandle(ql.SimpleQuote(dy)),
            )
            ts_up.enableExtrapolation()
            self.ts_base.linkTo(ts_up)
            Pup, _ = self.calculate(params, method, oas_decimal, state)

            ts_dn = ql.ZeroSpreadedTermStructure(
                ql.YieldTermStructureHandle(base_ptr),
                ql.QuoteHandle(ql.SimpleQuote(-dy)),
            )
            ts_dn.enableExtrapolation()
            self.ts_base.linkTo(ts_dn)
            Pdn, _ = self.calculate(params, method, oas_decimal, state)
        finally:
            self.ts_base.linkTo(base_ptr)

        dur = (Pdn - Pup) / (2.0 * P0 * dy)
        conv = (Pup + Pdn - 2.0 * P0) / (P0 * (dy ** 2))

        if return_state:
            return float(P0), float(dur), float(conv), state
        return float(P0), float(dur), float(conv)
