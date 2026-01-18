import QuantLib as ql
import pandas as pd


def us_calendar():
    """Return a *generic* United States calendar with robust fallbacks.

    We intentionally keep this as "generic" (Settlement/GovernmentBond/no-arg)
    because different QuantLib builds expose different market enums.
    """

    # Prefer the Settlement calendar when available (broadest in practice)
    try:
        if hasattr(ql.UnitedStates, "Settlement"):
            return ql.UnitedStates(getattr(ql.UnitedStates, "Settlement"))
    except Exception:
        pass

    # Common alternative
    try:
        if hasattr(ql.UnitedStates, "GovernmentBond"):
            return ql.UnitedStates(getattr(ql.UnitedStates, "GovernmentBond"))
    except Exception:
        pass

    # Older/alternative signature
    try:
        return ql.UnitedStates()
    except Exception:
        pass

    # Last-resort fallback (should rarely happen)
    return ql.TARGET()


def us_sofr_calendar():
    """Return the United States SOFR calendar if available (fallback to US).

    Some QuantLib versions expose `UnitedStates.SOFR`. When absent, we fallback
    to a generic US calendar.
    """

    try:
        if hasattr(ql.UnitedStates, "SOFR"):
            return ql.UnitedStates(getattr(ql.UnitedStates, "SOFR"))
    except Exception:
        pass

    return us_calendar()


def thirty360_usa():
    """Return a 30/360 day count with robust fallbacks."""

    try:
        return ql.Thirty360(ql.Thirty360.USA)
    except Exception:
        pass

    try:
        return ql.Thirty360()
    except Exception:
        pass

    return ql.Actual360()


class DateUtils:
    """Small helpers to keep date/period parsing in one place."""

    @staticmethod
    def to_ql_date(d):
        """Convert common Python date representations into QuantLib.Date."""
        if isinstance(d, ql.Date):
            return d
        if isinstance(d, str):
            d = pd.to_datetime(d).date()
        return ql.Date(d.day, d.month, d.year)

    @staticmethod
    def parse_period(s):
        """Parse strings such as '1Mo', '3Mo', '1Yr', '10Yr', '6M', '1Y'."""
        s = str(s).strip().upper()
        # Normalize common suffixes
        s = s.replace("MONTH", "M").replace("MO", "M")
        s = s.replace("YEAR", "Y").replace("YR", "Y")
        if s.endswith("M"):
            n = int(s[:-1])
            return ql.Period(n, ql.Months)
        if s.endswith("Y"):
            n = int(s[:-1])
            return ql.Period(n, ql.Years)
        # Fallback to QuantLib's parser (e.g. '6M')
        return ql.Period(s)

    @staticmethod
    def ensure_period(freq_or_period):
        """Convert Frequency/Period/string to QuantLib.Period."""
        if isinstance(freq_or_period, ql.Period):
            return freq_or_period
        if isinstance(freq_or_period, str):
            return DateUtils.parse_period(freq_or_period)
        # QuantLib Frequency is an int enum (e.g. ql.Semiannual)
        try:
            return ql.Period(freq_or_period)
        except Exception:
            # fallback: assume annual
            return ql.Period(1, ql.Years)

    @staticmethod
    def payments_per_year(freq_or_period):
        """Return coupon payments per year from Period/Frequency.

        This is used only to translate coupon_rate (annual) to coupon amount per period.
        """
        p = DateUtils.ensure_period(freq_or_period)
        n = p.length()
        units = p.units()
        if units == ql.Years:
            return 1.0 / float(n) if n > 0 else 1.0
        if units == ql.Months:
            return 12.0 / float(n) if n > 0 else 2.0
        if units == ql.Weeks:
            return 52.0 / float(n) if n > 0 else 52.0
        if units == ql.Days:
            return 365.0 / float(n) if n > 0 else 365.0
        # fallback
        return 2.0
