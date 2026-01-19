from datetime import date
from pathlib import Path

import QuantLib as ql
import pandas as pd

from callable_pricer.calibration import Calibrator
from callable_pricer.config import AppConfig
from callable_pricer.instruments import CallableBondSpec
from callable_pricer.market import MarketLoader
from callable_pricer.pricer import MasterPricer
from callable_pricer.reporting import (
    maybe_plot_curve_from_csv,
    maybe_plot_results,
    maybe_plot_surface_from_csv,
    save_calibration_params,
    save_config_snapshot,
    save_dataframe,
    save_hw_exercise_probabilities,
    maybe_plot_sensitivity,
    save_results_table,
)
from callable_pricer.sensitivity import price_vs_oas, price_vs_rate_shift, price_vs_volatility


def main():
    # -------------------------------------------------------------------------
    # 0. Inputs (adjust these for your bond / dataset)
    # -------------------------------------------------------------------------
    val_date = ql.Date(10, 9, 2025)
    oas_bps = 73.0

    cfg = AppConfig(val_date, oas_bps)
    cfg.apply_global_settings()

    # Bond example (same as validated prototype)
    bond = CallableBondSpec(
        face=100.0,
        coupon_rate=0.035,
        coupon_frequency=ql.Semiannual,
        issue_date=date(2015, 12, 2),
        maturity_date=date(2035, 12, 2),
        call_schedule=[(date(2034, 8, 12), 100.0)],
        # Defaults (see CallableBondSpec):
        # - calendar = United States
        # - schedule convention = Unadjusted
        # - payment convention = Following
        # - day count = 30/360 (USA)
    )

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    out_dir = project_root / "outputs"

    curve_csv = data_dir / "sofr_curve.csv"
    hw_vol_csv = data_dir / "sup_vol_swaptions_normal_hull_white.csv"
    cir_vol_csv = data_dir / "sup_vol_capfloor_CIR.csv"

    # -------------------------------------------------------------------------
    # 1. Load market data
    # -------------------------------------------------------------------------
    print("--- 1. Dados ---")
    loader = MarketLoader(cfg)
    ts_relinkable = loader.load_curve(str(curve_csv))
    hw_vols = loader.load_vols(str(hw_vol_csv), kind="HW")
    cir_vols = loader.load_vols(str(cir_vol_csv), kind="CIR")

    # -------------------------------------------------------------------------
    # 2. Calibrate models
    # -------------------------------------------------------------------------
    print("--- 2. Calibrando ---")
    calib = Calibrator(ts_relinkable)

    p_hw = calib.calibrate_hw(hw_vols)
    print(f"HW: a={p_hw['a']:.4f}, sigma={p_hw['sigma']:.4f}")

    p_cir = calib.calibrate_cir(cir_vols)
    print(f"CIR: theta={p_cir['theta']:.4f}, k={p_cir['k']:.4f}, sigma={p_cir['sigma']:.4f}")

    p_bk = calib.calibrate_bk(hw_vols)
    print(f"BK: a={p_bk['a']:.4f}, sigma={p_bk['sigma']:.4f}")

    calib_params = {"HW": p_hw, "CIR": p_cir, "BK": p_bk}

    # -------------------------------------------------------------------------
    # 3. Price + risk metrics
    # -------------------------------------------------------------------------
    print(
        f"\n--- 3. Resultados (Clean, OAS={oas_bps:.0f}bps, "
        f"RiskBump={cfg.risk_bump_bps:.0f}bps) ---"
    )

    pricer = MasterPricer(ts_relinkable, bond, cfg)

    scenarios = [
        ("STRAIGHT BOND", None, "STRAIGHT BOND"),
        ("HW (LSMC Manual)", p_hw, "HW_LSMC"),
        ("HW (QL Tree)", p_hw, "HW_QL_TREE"),
        ("CIR (PDE Manual)", p_cir, "CIR_PDE"),
        ("CIR (QL Tree)", p_cir, "CIR_QL_TREE"),
        ("BK (Manual Tree)", p_bk, "BK_MANUAL"),
        ("BK (QL Tree)", p_bk, "BK_QL_TREE"),
    ]

    print(f"{'METODO':<25} | {'PRICE':<10} | {'DUR':<10} | {'CONV':<10}")
    print("-" * 65)

    results = []
    hw_state_cache = None

    for label, params, method in scenarios:
        try:
            if method == "HW_LSMC":
                price, dur, conv, state = pricer.metrics(params, method, oas_bps / 10000.0, return_state=True)
                hw_state_cache = state
            else:
                price, dur, conv = pricer.metrics(params, method, oas_bps / 10000.0)

            print(f"{label:<25} | {price:<10.4f} | {dur:<10.4f} | {conv:<10.4f}")
            results.append({"method": label, "price": price, "duration": dur, "convexity": conv})

        except Exception as e:
            print(f"{label:<25} | ERRO: {str(e)}")
            results.append({"method": label, "price": float("nan"), "duration": float("nan"), "convexity": float("nan")})

    results_df = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # 4. Outputs (CSV + figures)
    # -------------------------------------------------------------------------
    save_results_table(results_df, out_dir)
    save_calibration_params(calib_params, out_dir)
    save_config_snapshot(cfg, out_dir)

    maybe_plot_results(results_df, out_dir)
    maybe_plot_curve_from_csv(curve_csv, out_dir)
    maybe_plot_surface_from_csv(hw_vol_csv, out_dir, title="swaption_normal_vol_surface")
    maybe_plot_surface_from_csv(cir_vol_csv, out_dir, title="capfloor_vol_surface")

    save_hw_exercise_probabilities(hw_state_cache, out_dir)

    # ---------------------------------------------------------------------
    # 5. Sensitivity figures (Chapter 4)
    # ---------------------------------------------------------------------
    oas_decimal = oas_bps / 10000.0

    # 5.1 Price vs volatility (scale model sigma)
    vol_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    df_vol = price_vs_volatility(pricer, scenarios, oas_decimal, vol_multipliers)
    save_dataframe(df_vol, out_dir, "sensitivity_price_vs_volatility.csv")
    maybe_plot_sensitivity(
        df_vol,
        out_dir,
        x_col="vol_multiplier",
        title="Price sensitivity vs volatility (sigma multiplier)",
        xlabel="Sigma multiplier",
        ylabel="Clean price",
        filename_png="price_vs_volatility.png",
    )

    # 5.2 Price vs interest rates (parallel shift)
    rate_shifts_bps = [-200, -100, -50, 0, 50, 100, 200]
    df_rate = price_vs_rate_shift(pricer, scenarios, oas_decimal, rate_shifts_bps)
    save_dataframe(df_rate, out_dir, "sensitivity_price_vs_rate_shift.csv")
    maybe_plot_sensitivity(
        df_rate,
        out_dir,
        x_col="rate_shift_bps",
        title="Price sensitivity vs interest rate (parallel shift)",
        xlabel="Parallel shift (bps)",
        ylabel="Clean price",
        filename_png="price_vs_rate_shift.png",
    )

    # 5.3 Price vs credit spread (OAS)
    oas_grid_bps = [0, 25, 50, 75, 90, 100, 150, 200]
    df_oas = price_vs_oas(pricer, scenarios, oas_grid_bps)
    save_dataframe(df_oas, out_dir, "sensitivity_price_vs_oas.csv")
    maybe_plot_sensitivity(
        df_oas,
        out_dir,
        x_col="oas_bps",
        title="Price sensitivity vs credit spread (OAS)",
        xlabel="OAS (bps)",
        ylabel="Clean price",
        filename_png="price_vs_oas.png",
    )

    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
