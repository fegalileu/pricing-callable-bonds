import numpy as np
import QuantLib as ql
from scipy import optimize

from ..utils import DateUtils


class BKManualTreeEngine:
    """Manual BK-style recombining tree for callable bonds.

    The forward calibration (solving for alpha_t each step) is kept close to the
    validated prototype: alpha is chosen so the tree matches the discount factor
    curve at each step.

    Notes
    -----
    - Coupon amount is fixed per period: face * coupon_rate / payments_per_year.
    - Time is measured using the bond day count (default: 30/360 USA).
    - The tree is lognormal in the short rate: r = exp(alpha + j*dx).
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

        time_dc = bond_data.get('bond_day_count') or ql.Thirty360(ql.Thirty360.USA)
        T = float(time_dc.yearFraction(today, mat))

        # Cache geometry between bumps
        if state_cache is None:
            dt = 1.0 / float(self.cfg.bk_steps_year)
            dx = sigma * np.sqrt(3.0 * dt)
            N = int(T / dt) + 2
            j_max = min(int(6.0 / (a * dt)) + 1, 150) if a > 1e-12 else 150
            state_cache = {'dt': dt, 'dx': dx, 'N': N, 'j_max': j_max}

        dt = float(state_cache['dt'])
        dx = float(state_cache['dx'])
        N = int(state_cache['N'])
        j_max = int(state_cache['j_max'])

        # Forward calibration: solve alpha[i] so that the tree matches discount(t)
        Q = np.zeros(2 * j_max + 1)
        Q[j_max] = 1.0

        alpha = np.zeros(N)
        js = np.arange(-j_max, j_max + 1)
        zc = [float(ts_obj.discount((i + 1) * dt)) for i in range(N)]

        tree = []
        for i in range(N - 1):
            mu = -a * js * dx
            k = np.round(mu * dt / dx).astype(int)
            pu = 1.0 / 6.0 + 0.5 * ((mu * dt / dx - k) ** 2 + (mu * dt / dx - k))
            pd = 1.0 / 6.0 + 0.5 * ((mu * dt / dx - k) ** 2 - (mu * dt / dx - k))
            pm = 1.0 - pu - pd
            tree.append((k, pu, pm, pd))

        for i in range(N - 1):
            mask = Q > 1e-16
            if not np.any(mask):
                break

            def obj(val):
                return np.sum(Q[mask] * np.exp(-np.exp(val + js[mask] * dx) * dt)) - zc[i]

            try:
                alpha[i] = optimize.brentq(obj, -12.0, 12.0, xtol=1e-12)
            except Exception:
                alpha[i] = alpha[i - 1] if i > 0 else np.log(0.05)

            k, pu, pm, pd = tree[i]
            Q_n = np.zeros_like(Q)
            w = Q * np.exp(-np.exp(alpha[i] + js * dx) * dt)

            idx = np.where(mask)[0]
            for j in idx:
                kk = k[j]
                if 0 <= j + kk < len(Q):
                    Q_n[j + kk] += w[j] * pm[j]
                if 0 <= j + kk + 1 < len(Q):
                    Q_n[j + kk + 1] += w[j] * pu[j]
                if 0 <= j + kk - 1 < len(Q):
                    Q_n[j + kk - 1] += w[j] * pd[j]
            Q = Q_n

        # -----------------------------
        # Cashflows and call schedule
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

        cfs = {}
        for d in sch:
            if d > today:
                idx = int(round(float(time_dc.yearFraction(today, d)) / dt))
                amt = coupon_amt + (face if d == mat else 0.0)
                cfs[idx] = cfs.get(idx, 0.0) + float(amt)

        calls = {}
        for cd, price in bond_data.get('call_schedule', []):
            qd = DateUtils.to_ql_date(cd)
            if qd <= today:
                continue
            idx = int(round(float(time_dc.yearFraction(today, qd)) / dt))
            calls[idx] = float(price)

        # -----------------------------
        # Backward induction
        # -----------------------------
        V = np.zeros(2 * j_max + 1)
        if (N - 1) in cfs:
            V[:] = cfs[N - 1]

        for i in range(len(tree) - 1, -1, -1):
            k, pu, pm, pd = tree[i]
            V_n = np.zeros_like(V)

            # Loop version is kept for safety (variable k per node)
            for j in range(1, 2 * j_max):
                kk = k[j]
                if 0 < j + kk < 2 * j_max:
                    ev = pu[j] * V[j + kk + 1] + pm[j] * V[j + kk] + pd[j] * V[j + kk - 1]
                    disc = np.exp(-np.exp(alpha[i] + (j - j_max) * dx) * dt)
                    val = ev * disc

                    if i in cfs:
                        val += cfs[i]
                    if i in calls:
                        val = min(val, calls[i] + cfs.get(i, 0.0))

                    V_n[j] = val

            V = V_n

        return float(V[j_max]), state_cache
