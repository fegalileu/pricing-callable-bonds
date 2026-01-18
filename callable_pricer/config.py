import numpy as np
import warnings


class AppConfig:
    """Central configuration object.

    This project is intended for a dissertation-style comparison between pricing
    methods (Tree, PDE, Monte Carlo). Reproducibility matters, so all numerical
    knobs live here.

    Parameters
    ----------
    val_date : QuantLib.Date
        Evaluation date for QuantLib.
    oas_bps : float
        Static spread (OAS) in basis points applied to the risk-free curve.
    bump_bps : float
        Bump (in bps) used in the console output / reports and, by default,
        also used to compute effective duration/convexity.

    Notes
    -----
    - ``risk_bump_bps`` is the bump actually used for effective duration/convexity.
      This codebase defaults to using ``bump_bps`` (typically 10bp), which is the
      market-standard effective bump and also the one used in the dissertation text.
    """

    def __init__(self, val_date, oas_bps=0.0, bump_bps=1.0):
        self.val_date = val_date
        self.oas_bps = float(oas_bps)
        self.bump_bps = float(bump_bps)

        # Effective risk bump (used inside MasterPricer.metrics)
        # Default: follow the configured bump (typically 10bp), because
        # 1bp can be too small for some lattice engines, causing noisy convexity.
        self.risk_bump_bps = float(bump_bps)

        # ----------------
        # Monte Carlo (Hull-White)
        # ----------------
        self.mc_paths = 50000
        self.mc_steps_year = 24  # ~quinzenal
        self.mc_seed = 12345

        # ----------------
        # PDE (CIR)
        # ----------------
        self.pde_r_min = 1.0e-5
        self.pde_r_max = 0.80
        self.pde_grid_size = 800
        self.pde_steps_year = 100

        # If True, we shift theta with the curve bump (pragmatic fix for effective duration).
        self.pde_shift_theta_with_curve = True

        # ----------------
        # Manual Tree (KWF-like)
        # ----------------
        self.kwf_steps_year = 52  # semanal

        # ----------------
        # QuantLib Tree engines (reference)
        # ----------------
        self.ql_grid_size = 128

        # ----------------
        # Global flags
        # ----------------
        self.suppress_warnings = True
        self.numpy_seed = 42

    def apply_global_settings(self):
        """Apply global deterministic settings (warnings + RNG seed)."""
        if self.suppress_warnings:
            warnings.filterwarnings("ignore")
        np.random.seed(self.numpy_seed)
