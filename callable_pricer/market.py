import io
import pandas as pd
import QuantLib as ql

from .utils import DateUtils, us_sofr_calendar


class MarketLoader:
    """Load market inputs (discount curve + volatility surfaces).

    The loader is intentionally permissive regarding column names to make the
    project easy to run with different SOFR/OIS exports.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        ql.Settings.instance().evaluationDate = cfg.val_date

    def load_curve(self, path, day_count=None, calendar=None, allow_extrapolation=True):
        """Load a discount curve from a CSV and return a relinkable handle.

        The CSV is expected to contain at least:
        - a date column (e.g. 'data_vertice')
        - a discount factor column (e.g. 'fator_desconto')

        The loader will ensure the valuation date exists with DF = 1.0.
        """
        # Business rule (per dissertation): SOFR/OIS discounting with
        # Act/360 and a United States calendar (SOFR if available).
        day_count = day_count or ql.Actual360()
        calendar = calendar or us_sofr_calendar()

        df = pd.read_csv(path)
        # Try to auto-detect columns
        col_date = next(
            (c for c in df.columns if "data" in c.lower() or "vertice" in c.lower()),
            None,
        )
        col_df = next(
            (c for c in df.columns if "fator" in c.lower() or "desconto" in c.lower()),
            None,
        )
        if col_date is None or col_df is None:
            raise ValueError(
                "CSV da curva deve conter coluna de data (data/vertice) e coluna de fator de desconto (fator/desconto)."
            )

        df[col_date] = pd.to_datetime(df[col_date])
        df = df.sort_values(col_date)

        dates = [self.cfg.val_date]
        dfs = [1.0]

        for _, row in df.iterrows():
            d = DateUtils.to_ql_date(row[col_date])
            if d <= self.cfg.val_date:
                continue
            dates.append(d)
            dfs.append(float(row[col_df]))

        curve = ql.DiscountCurve(dates, dfs, day_count, calendar)
        if allow_extrapolation:
            curve.enableExtrapolation()
        return ql.RelinkableYieldTermStructureHandle(curve)

    def load_vols(self, path, kind="HW"):
        """Load vol data from a CSV.

        Parameters
        ----------
        path : str
            CSV path.
        kind : str
            - 'HW': swaption normal vol surface (matrix: Expiry x Tenor)
            - 'CIR': cap/floor vol surface (matrix: Tenor x Strike). 'ATM' is ignored.

        Returns
        -------
        list
            - HW: list of (expiry_period, tenor_period, vol)
            - CIR: list of (tenor_period, strike, vol)
        """
        df = pd.read_csv(path)
        if df.empty:
            return []

        df = df.copy()
        df.set_index(df.columns[0], inplace=True)

        data = []
        for idx_row, row in df.iterrows():
            for idx_col, val in row.items():
                try:
                    if pd.isna(val):
                        continue
                    t = DateUtils.parse_period(idx_row)

                    if kind.upper() == "HW":
                        ten = DateUtils.parse_period(idx_col)
                        data.append((t, ten, float(val)))
                    else:
                        # CIR cap/floor surface: columns are strikes
                        if "ATM" in str(idx_col).upper():
                            continue
                        s = float(str(idx_col).strip().replace("%", "")) / 100.0
                        data.append((t, s, float(val)))
                except Exception:
                    continue
        return data
