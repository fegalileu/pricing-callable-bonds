import os
import sys
from datetime import date

import QuantLib as ql

# Allow running tests without installing the package.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from callable_pricer import AppConfig, MarketLoader, Calibrator, CallableBondSpec, MasterPricer


def test_smoke_run():
    """Basic smoke test: load data, calibrate, price.

    This is not a unit test of financial correctness; it checks that the code
    runs end-to-end without exploding.
    """
    val_date = ql.Date(10, 9, 2025)
    cfg = AppConfig(val_date, oas_bps=73.0, bump_bps=1.0)
    cfg.mc_paths = 2000  # speed-up for CI/smoke
    cfg.apply_global_settings()

    bond = CallableBondSpec(
        face=100.0,
        coupon_rate=0.035,
        coupon_frequency=ql.Semiannual,
        issue_date=date(2015, 12, 2),
        maturity_date=date(2035, 12, 2),
        call_schedule=[(date(2034, 8, 12), 100.0)],
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    loader = MarketLoader(cfg)
    ts = loader.load_curve(os.path.join(data_dir, 'sofr_curve.csv'))

    hw_vols = loader.load_vols(os.path.join(data_dir, 'sup_vol_swaptions_normal_hull_white.csv'), kind='HW')
    cir_vols = loader.load_vols(os.path.join(data_dir, 'sup_vol_capfloor_CIR.csv'), kind='CIR')

    calib = Calibrator(ts)
    p_hw = calib.calibrate_hw(hw_vols)
    p_cir = calib.calibrate_cir(cir_vols)
    p_bk = calib.calibrate_bk(hw_vols)

    pricer = MasterPricer(ts, bond, cfg)
    oas = cfg.oas_bps / 10000.0

    # Prices should be finite
    for method, params in [
        ('STRAIGHT BOND', None),
        ('HW_LSMC', p_hw),
        ('CIR_PDE', p_cir),
        ('BK_MANUAL', p_bk),
    ]:
        p, d, c = pricer.metrics(params, method, oas)
        assert abs(p) < 1e6
        assert abs(d) < 1e6
        assert abs(c) < 1e6
