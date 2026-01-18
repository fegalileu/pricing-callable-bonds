import QuantLib as ql
from scipy import optimize


class Calibrator:
    """Model calibrations used in the empirical comparison.

    The calibration routines are intentionally pragmatic:
    - Hull-White is calibrated to a set of swaptions with NORMAL vols.
    - CIR is calibrated to a subset of cap vols (normal) via differential evolution.
    - KWF proxy uses Black-Karasinski calibrated to a small subset of swaptions
      after converting normal vol -> lognormal vol via a simple ATM conversion.

    These choices are meant to keep the project runnable and aligned with the validated
    prototype, while still being justifiable academically.
    """

    def __init__(self, ts_handle):
        self.ts = ts_handle
        self.idx = ql.USDLibor(ql.Period('3M'), self.ts)

    def calibrate_hw(self, data):
        """Calibrate Hull-White (a, sigma) to NORMAL swaption vols.

        Parameters
        ----------
        data : list
            List of (expiry_period, tenor_period, normal_vol).

        Returns
        -------
        dict
            {'a': ..., 'sigma': ...}
        """
        if not data:
            return {'a': 0.01, 'sigma': 0.01}

        model = ql.HullWhite(self.ts)
        eng = ql.JamshidianSwaptionEngine(model)

        helpers = []
        for exp, ten, vol in data:
            # Keep a sane subset (optional). The condition below matches the prototype.
            try:
                if exp.length() + ten.length() > 25:
                    continue
                h = ql.SwaptionHelper(
                    exp,
                    ten,
                    ql.QuoteHandle(ql.SimpleQuote(float(vol))),
                    self.idx,
                    ql.Period('1Y'),
                    ql.Thirty360(ql.Thirty360.USA),
                    ql.Actual360(),
                    self.ts,
                    ql.SwaptionHelper.RelativePriceError,
                    ql.nullDouble(),
                    1.0,
                    ql.Normal,
                    0.0,
                )
                h.setPricingEngine(eng)
                helpers.append(h)
            except Exception:
                continue

        if helpers:
            model.calibrate(
                helpers,
                ql.LevenbergMarquardt(),
                ql.EndCriteria(1000, 200, 1e-8, 1e-8, 1e-8),
            )

        a, sigma = model.params()[0], model.params()[1]
        return {'a': float(a), 'sigma': float(sigma)}

    def calibrate_cir(self, data):
        """Calibrate CIR (theta, kappa, sigma) to cap vols.

        Notes
        -----
        This follows the validated prototype: we build a set of cap helpers and
        fit (theta, kappa, sigma) with a mild Feller penalty.

        Returns
        -------
        dict
            {'theta': ..., 'k': ..., 'sigma': ..., 'r0': ...}
        """
        r0 = self.ts.currentLink().zeroRate(0.001, ql.Continuous).rate()
        if not data:
            return {'theta': 0.05, 'k': 0.1, 'sigma': 0.02, 'r0': float(r0)}

        helpers = []
        for ten, stk, vol in data:
            try:
                # Keep short tenors (fast + stable) as in the prototype
                if ten.length() > 5:
                    continue

                # NOTE: prototype ignores strike. We keep the same behavior for consistency.
                h = ql.CapHelper(
                    ten,
                    ql.QuoteHandle(ql.SimpleQuote(float(vol))),
                    self.idx,
                    ql.Annual,
                    ql.Thirty360(ql.Thirty360.USA),
                    True,
                    self.ts,
                    ql.CapHelper.RelativePriceError,
                    ql.Normal,
                )
                helpers.append(h)
            except Exception:
                continue

        def loss(x):
            theta, kappa, sigma = x
            if kappa <= 1e-3 or sigma <= 1e-3 or theta <= 1e-3:
                return 1e9

            # Feller penalty (soft)
            penal = 0.0 if 2.0 * kappa * theta > sigma ** 2 else 1.0

            try:
                m = ql.CoxIngersollRoss(float(r0), float(theta), float(kappa), float(sigma))
                e = ql.AnalyticCapFloorEngine(m, self.ts)

                err = 0.0
                for h in helpers:
                    h.setPricingEngine(e)
                    mkt = h.marketValue()
                    if mkt > 1e-6:
                        err += ((h.modelValue() - mkt) / mkt) ** 2
                return float(err + penal)
            except Exception:
                return 1e9

        # Differential evolution (global-ish). Keep it light to stay runnable.
        res = optimize.differential_evolution(
            loss,
            [(0.01, 0.15), (0.01, 1.0), (0.01, 0.2)],
            seed=42,
            maxiter=5,
        )

        return {
            'theta': float(res.x[0]),
            'k': float(res.x[1]),
            'sigma': float(res.x[2]),
            'r0': float(r0),
        }

    def calibrate_kwf(self, data):
        """Calibrate a KWF proxy via Black-Karasinski + tree swaption engine.

        The prototype uses a very simple conversion from NORMAL vol to lognormal vol:
            vol_lognormal ≈ vol_normal / ATM

        This is not a full-fledged model conversion; it is good enough for a
        comparative study where the objective is consistency across methods.
        """
        if not data:
            return {'a': 0.1, 'sigma': 0.2}

        model = ql.BlackKarasinski(self.ts)
        eng = ql.TreeSwaptionEngine(model, 10)
        swp_engine = ql.DiscountingSwapEngine(self.ts)

        helpers = []
        for exp, ten, vol in data:
            try:
                # Keep a small anchor set (as in the prototype)
                if exp.length() not in [1, 5]:
                    continue
                if ten.length() != 5:
                    continue

                swap = ql.MakeVanillaSwap(ten, self.idx, 0.03, exp)
                swap.setPricingEngine(swp_engine)
                atm = max(swap.fairRate(), 0.01)
                vol_ln = float(vol) / float(atm)

                h = ql.SwaptionHelper(
                    exp,
                    ten,
                    ql.QuoteHandle(ql.SimpleQuote(vol_ln)),
                    self.idx,
                    ql.Period('1Y'),
                    ql.Thirty360(ql.Thirty360.USA),
                    ql.Actual360(),
                    self.ts,
                    ql.SwaptionHelper.RelativePriceError,
                    ql.nullDouble(),
                    1.0,
                    ql.ShiftedLognormal,
                    0.0,
                )
                h.setPricingEngine(eng)
                helpers.append(h)
            except Exception:
                continue

        if helpers:
            model.calibrate(
                helpers,
                ql.LevenbergMarquardt(),
                ql.EndCriteria(200, 50, 1e-6, 1e-6, 1e-6),
            )

        a, sigma = model.params()[0], model.params()[1]
        return {'a': float(a), 'sigma': float(sigma)}
