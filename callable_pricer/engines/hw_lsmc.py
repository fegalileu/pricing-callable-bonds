import numpy as np
import QuantLib as ql

from ..utils import DateUtils


class HullWhiteLSMCEngine:
    """Hull-White (1F) Monte Carlo with Longstaff-Schwartz regression.

    Important
    ---------
    This implementation intentionally mirrors a validated monolithic prototype
    (the one used to generate the baseline results in the dissertation). The
    refactor keeps the *same numerical logic* so outputs remain comparable.

    Notes on modelling choices (consistent with the prototype)
    ----------------------------------------------------------
    - State variable: OU process X(t) with mean reversion 'a' and vol 'sigma'.
    - Short rate is constructed via a deterministic shift ``alpha(t)`` so that
      instantaneous forwards from the input curve are matched.
    - Coupon amounts are treated as fixed per period:

        coupon = face * coupon_rate / payments_per_year

      (standard market convention for regular coupons; stubs are ignored).
    - Call is treated as Bermudan at the provided call dates.

    The engine returns a *dirty* PV at t=0 (valuation date). The orchestrator
    converts to clean by subtracting accrued.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def _payments_per_year(self, coupon_frequency):
        return DateUtils.payments_per_year(coupon_frequency)

    def price(self, ts, bond_data, params, state_cache=None):
        a = float(params['a'])
        sigma = float(params['sigma'])

        ts_obj = ts.currentLink()
        today = ts.referenceDate()
        mat = DateUtils.to_ql_date(bond_data['maturity_date'])

        # Time axis: use the bond day count (e.g., 30/360) so cashflow times and
        # exercise times line up with the instrument conventions.
        #
        # Note: the SOFR curve itself is Act/360; the time grid here is an
        # approximation consistent with the validated prototype used in the
        # dissertation.
        time_dc = bond_data.get('bond_day_count', ql.Thirty360(ql.Thirty360.USA))
        T = float(time_dc.yearFraction(today, mat))

        N = int(T * int(self.cfg.mc_steps_year))
        if N <= 1:
            N = 2
        dt = T / N

        # -----------------------------
        # 1) Common Random Numbers (CRN)
        # -----------------------------
        if state_cache is None:
            rng = np.random.RandomState(int(self.cfg.mc_seed)).normal(0.0, 1.0, (int(self.cfg.mc_paths), N))
            state_cache = {
                'rng': rng,
                'regressions': {},
                'exercise_prob': {},  # populated in base run (optional diagnostics)
            }
        rng = state_cache['rng']

        n_paths = rng.shape[0]

        # -----------------------------
        # 2) Simulate OU state X
        # -----------------------------
        X = np.zeros((n_paths, N + 1))
        if abs(a) < 1e-12:
            # a ~ 0 degenerates to Brownian driftless short-rate model
            for i in range(N):
                X[:, i + 1] = X[:, i] + sigma * np.sqrt(dt) * rng[:, i]
        else:
            for i in range(N):
                X[:, i + 1] = X[:, i] * (1.0 - a * dt) + sigma * np.sqrt(dt) * rng[:, i]

        # -----------------------------
        # 3) Shift alpha(t) to match the input curve
        # -----------------------------
        t_grid = np.linspace(0.0, T, N + 1)
        alpha = np.zeros(N + 1)
        for i, t in enumerate(t_grid):
            # Numerical forward rate from the term structure
            fwd = ts_obj.forwardRate(float(t), float(t) + 0.001, ql.Continuous, ql.NoFrequency).rate()

            if abs(a) > 1e-5:
                var = (sigma ** 2 / (2.0 * a ** 2)) * (1.0 - np.exp(-a * t)) ** 2
            else:
                var = 0.5 * (sigma ** 2) * (t ** 2)
            alpha[i] = float(fwd) + float(var)

        # Short rate along each path
        R = X + alpha

        # -----------------------------
        # 4) Build cashflow and call times
        # -----------------------------
        face = float(bond_data['face'])
        coupon_rate = float(bond_data['coupon_rate'])
        payments_per_year = float(self._payments_per_year(bond_data['coupon_frequency']))
        coupon_amt = face * coupon_rate / payments_per_year

        issue = DateUtils.to_ql_date(bond_data['issue_date'])
        period = DateUtils.ensure_period(bond_data['coupon_frequency'])
        sch = ql.Schedule(
            issue,
            mat,
            period,
            bond_data['calendar'],
            bond_data['business_convention'],
            bond_data['business_convention'],
            bond_data['date_generation'],
            bool(bond_data['end_of_month']),
        )

        # Cashflows list: (time, amount). Amount includes principal at maturity.
        cfs = []
        for d in sch:
            if d > today:
                amt = coupon_amt
                if d == mat:
                    amt += face
                cfs.append((float(time_dc.yearFraction(today, d)), float(amt)))

        # Map call events to steps using the same discretization spirit as the prototype.
        call_step_to_price = {}
        call_step_to_date = {}
        for cd, price in bond_data.get('call_schedule', []):
            qd = DateUtils.to_ql_date(cd)
            if qd <= today:
                continue
            t_call = float(time_dc.yearFraction(today, qd))
            step = int(round(t_call / dt))
            step = max(1, min(step, N))
            call_step_to_price[step] = float(price)
            call_step_to_date[step] = qd

        # -----------------------------
        # 5) Backward induction with LSMC
        # -----------------------------
        V = np.zeros(n_paths)
        is_base_run = (len(state_cache.get('regressions', {})) == 0)

        for i in range(N - 1, -1, -1):
            t = t_grid[i]
            t_next = t_grid[i + 1]

            # Discount one step using the short rate at the beginning of the interval
            V *= np.exp(-R[:, i] * dt)

            # Add any cashflows that happen in (t, t_next]
            curr_amt = 0.0
            for ct, amt in cfs:
                if t < ct <= t_next:
                    V += amt
                    curr_amt += amt

            # Call decision at t_next (Bermudan)
            call_step = i + 1
            if call_step in call_step_to_price:
                strike = call_step_to_price[call_step]

                # Quadratic basis in the *state* X (same as prototype)
                basis = np.vstack([np.ones(n_paths), X[:, i + 1], X[:, i + 1] ** 2]).T

                if is_base_run:
                    try:
                        betas = np.linalg.lstsq(basis, V, rcond=None)[0]
                    except Exception:
                        betas = np.zeros(3)
                    state_cache['regressions'][i] = betas
                else:
                    betas = state_cache['regressions'].get(i, np.zeros(3))

                cont_val = basis @ betas
                exercise = (cont_val - curr_amt) > strike

                # Optional diagnostics (does not affect the price)
                if is_base_run:
                    state_cache['exercise_prob'][str(call_step_to_date[call_step])] = float(np.mean(exercise))

                V = np.where(exercise, strike + curr_amt, V)

        # Return Standard Error of the Mean (SEM) = std(V) / sqrt(N)
        # This represents the statistical uncertainty of the estimated price.
        return float(np.mean(V)), float(np.std(V) / np.sqrt(n_paths)), state_cache
