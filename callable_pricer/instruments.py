from dataclasses import dataclass, field
from datetime import date

import QuantLib as ql

from .utils import DateUtils, thirty360_usa, us_calendar


@dataclass
class CallableBondSpec:
    """Specification of a (callable) fixed-rate corporate bond.

    The manual engines in this repository follow a validated prototype that
    assumes *regular* coupon payments with a fixed coupon amount per period:

        coupon_amount = face * coupon_rate / payments_per_year

    This is the standard market convention for most fixed-rate bonds (ignoring
    stubs). For stubs or more complex conventions, QuantLib's internal cashflows
    should be preferred.

    Notes
    -----
    - ``call_schedule`` is a list of (call_date, call_price_clean). Each call
      date is treated as a Bermudan exercise date.
    - ``business_convention`` and ``calendar`` are used to build the coupon
      schedule.
    - ``payment_convention`` is used by QuantLib bond constructors.

    """

    face: float
    coupon_rate: float
    coupon_frequency: object  # QuantLib Frequency, Period, or string like "6M"
    issue_date: date
    maturity_date: date

    # Call schedule: list of (date, clean_price). If empty, the bond is non-callable.
    call_schedule: list = field(default_factory=list)

    # Conventions
    # Bond cashflows: 30/360 (USA) is the standard for many USD corporates.
    bond_day_count: object = field(default_factory=thirty360_usa)
    # United States calendar (SOFR is USD). Robust fallbacks are used for
    # compatibility across QuantLib builds.
    calendar: object = field(default_factory=us_calendar)
    business_convention: object = ql.Unadjusted
    payment_convention: object = ql.Following
    date_generation: object = ql.DateGeneration.Backward
    end_of_month: bool = False

    redemption: float = None

    def __post_init__(self):
        if self.redemption is None:
            self.redemption = float(self.face)

        # Normalize call_schedule to list[(ql.Date|date|str, float)]
        if self.call_schedule is None:
            self.call_schedule = []

    # ---------------------------------------------------------------------
    # QuantLib objects
    # ---------------------------------------------------------------------
    def ql_schedule(self):
        issue = DateUtils.to_ql_date(self.issue_date)
        mat = DateUtils.to_ql_date(self.maturity_date)
        period = DateUtils.ensure_period(self.coupon_frequency)
        return ql.Schedule(
            issue,
            mat,
            period,
            self.calendar,
            self.business_convention,
            self.business_convention,
            self.date_generation,
            bool(self.end_of_month),
        )

    def ql_straight_bond(self):
        sch = self.ql_schedule()
        issue = DateUtils.to_ql_date(self.issue_date)
        return ql.FixedRateBond(
            0,
            float(self.face),
            sch,
            [float(self.coupon_rate)],
            self.bond_day_count,
            self.payment_convention,
            float(self.redemption),
            issue,
        )

    def ql_callable_bond(self):
        sch = self.ql_schedule()
        issue = DateUtils.to_ql_date(self.issue_date)

        callabilities = []
        for d, price in self.call_schedule:
            cd = DateUtils.to_ql_date(d)
            callabilities.append(
                ql.Callability(
                    ql.BondPrice(float(price), ql.BondPrice.Clean),
                    ql.Callability.Call,
                    cd,
                )
            )

        return ql.CallableFixedRateBond(
            0,
            float(self.face),
            sch,
            [float(self.coupon_rate)],
            self.bond_day_count,
            self.payment_convention,
            float(self.redemption),
            issue,
            callabilities,
        )

    # ---------------------------------------------------------------------
    # Helper values used by manual engines
    # ---------------------------------------------------------------------
    def accrued_amount(self, evaluation_date=None):
        """Accrued amount from QuantLib (dirty - clean conversion).

        If ``evaluation_date`` is None, QuantLib's global evaluation date is used.
        """
        bond = self.ql_straight_bond()
        if evaluation_date is None:
            return float(bond.accruedAmount())
        return float(bond.accruedAmount(DateUtils.to_ql_date(evaluation_date)))

    def to_engine_bond_data(self):
        """Minimal dictionary passed to manual engines.

        We keep it close to the validated monolithic script to avoid accidental
        behavior changes.
        """
        return {
            'face': float(self.face),
            'coupon_rate': float(self.coupon_rate),
            'coupon_frequency': self.coupon_frequency,
            'issue_date': self.issue_date,
            'maturity_date': self.maturity_date,
            'call_schedule': list(self.call_schedule),
            'bond_day_count': self.bond_day_count,
            'calendar': self.calendar,
            # In the original script the schedule was Unadjusted; we keep the
            # same naming to avoid touching all engines.
            'business_convention': self.business_convention,
            'date_generation': self.date_generation,
            'end_of_month': bool(self.end_of_month),
        }
