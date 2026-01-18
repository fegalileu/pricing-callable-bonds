import numpy as np
import QuantLib as ql
from scipy import sparse
from scipy.sparse.linalg import splu

from ..utils import DateUtils


class CIRPDEEngine:
    """CIR PDE engine (manual finite-difference) for callable bonds.

    This implementation is intentionally close to the validated prototype used
    in the dissertation. In particular:

    - Time is measured using the bond day count (default: 30/360 USA), so the
      PDE time grid is aligned with the instrument convention.
    - Coupon amount is fixed per period: face * coupon_rate / payments_per_year
      (no stub handling).
    - The call is applied as an obstacle (min with call payoff) at each call date.

    Notes on OAS handling
    ---------------------
    The orchestrator applies the OAS by shifting the input curve handle. This
    affects the curve-implied r0 read here. The PDE itself discounts by the
    state rate r (i.e., it does not explicitly add a spread in the PDE term),
    which matches the validated prototype.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def _payments_per_year(self, coupon_frequency):
        return DateUtils.payments_per_year(coupon_frequency)

    def price(self, ts, bond_data, params, state_cache=None):
        theta_base = float(params['theta'])
        k = float(params['k'])
        sigma = float(params['sigma'])

        ts_obj = ts.currentLink()

        # r0 is read from the *current* curve (base / bumped)
        r0 = float(ts_obj.zeroRate(0.001, ql.Continuous).rate())

        # Optional: shift theta with the curve bump (prototype trick for stable risk metrics)
        if state_cache is None:
            state_cache = {'r0_base': r0}
        if bool(getattr(self.cfg, 'pde_shift_theta_with_curve', True)):
            r0_base = float(state_cache.get('r0_base', r0))
            spread_shock = r0 - r0_base
            theta = theta_base + spread_shock
        else:
            theta = theta_base

        # Spatial grid
        r = np.linspace(float(self.cfg.pde_r_min), float(self.cfg.pde_r_max), int(self.cfg.pde_grid_size))
        dr = float(r[1] - r[0])

        today = ts.referenceDate()
        mat = DateUtils.to_ql_date(bond_data['maturity_date'])
        time_dc = bond_data.get('bond_day_count', ql.Thirty360(ql.Thirty360.USA))
        T = float(time_dc.yearFraction(today, mat))

        Nt = int(T * int(self.cfg.pde_steps_year))
        if Nt <= 2:
            Nt = 3
        dt = T / Nt

        # CIR coefficients
        drift = k * (theta - r)
        diff = 0.5 * (sigma ** 2) * r

        # Crank-Nicolson discretization (prototype form)
        A = -0.25 * dt * (diff / dr ** 2 - drift / (2.0 * dr))
        B = 1.0 + 0.5 * dt * (r + diff / dr ** 2)
        C = -0.25 * dt * (diff / dr ** 2 + drift / (2.0 * dr))

        R_sub = 0.25 * dt * (diff / dr ** 2 - drift / (2.0 * dr))
        R_main = 1.0 - 0.5 * dt * (r + diff / dr ** 2)
        R_sup = 0.25 * dt * (diff / dr ** 2 + drift / (2.0 * dr))

        # Boundary at r_min: reflecting-like adjustment (prototype)
        B[0] += 2.0 * A[0]
        C[0] -= A[0]
        A[0] = 0.0

        M_L = sparse.diags([A[1:], B, C[:-1]], [-1, 0, 1], format='csc')
        M_R = sparse.diags([R_sub[1:], R_main, R_sup[:-1]], [-1, 0, 1], format='csc')
        solver = splu(M_L)

        # -----------------------------
        # Cashflows and call schedule
        # -----------------------------
        face = float(bond_data['face'])
        coupon_rate = float(bond_data['coupon_rate'])
        payments_per_year = float(self._payments_per_year(bond_data['coupon_frequency']))
        coupon_amt = face * coupon_rate / payments_per_year

        # Terminal condition: principal + last coupon
        V = np.full_like(r, face + coupon_amt, dtype=float)

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

        # Coupons before maturity (coupon-only)
        cfs = []
        for d in sch:
            if d > today and d < mat:
                cfs.append((float(time_dc.yearFraction(today, d)), float(coupon_amt)))

        # Bermudan call dates
        call_events = []
        for cd, price in bond_data.get('call_schedule', []):
            qd = DateUtils.to_ql_date(cd)
            if qd <= today:
                continue
            call_events.append((float(time_dc.yearFraction(today, qd)), float(price)))

        # -----------------------------
        # Backward solve in time
        # -----------------------------
        curr_t = T
        for _ in range(Nt):
            prev_t = curr_t
            curr_t -= dt

            V = solver.solve(M_R @ V)

            # Coupon jumps
            for ct, amt in cfs:
                if curr_t <= ct < prev_t:
                    V += amt

            # Call obstacle (applied only on call dates)
            for ct, call_price in call_events:
                if curr_t <= ct < prev_t:
                    # coupon at the call date (if aligned)
                    cpn = 0.0
                    for cft, amt in cfs:
                        if abs(cft - ct) < dt * 2.0:
                            cpn = amt
                            break
                    V = np.minimum(V, call_price + cpn)

        price = float(np.interp(r0, r, V))
        return price, state_cache
