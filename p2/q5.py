# q5.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Q3 helpers (adjust module name if needed)
from q3 import (
    build_m_grid,
    forward_and_discount,
    price_options_from_paths,
    vix_from_option_curve,
)

# ============================================================
# Helpers: Q-weights, FHS sampling, RN parameters, simulation
# ============================================================

def compute_Q_weights(eps_array, eta):
    """
    Exponential-tilt weights q_i(η) ∝ exp(-η ε_i).
    """
    w = np.exp(-eta * eps_array)
    q = w / w.sum()
    return q


def draw_eps_Q(eps_array, q_weights, lam, n_paths, rng):
    """
    Draw shocks under Q via FHS (tilted historical simulation).
    """
    N = len(eps_array)
    idx = rng.choice(N, size=n_paths, replace=True, p=q_weights)
    eps_Q = eps_array[idx] + lam
    return eps_Q


def get_rn_params(params):
    """
    Risk-neutral NGARCH parameters via Duan LRNVR:
      gamma_Q = gamma + lambda, lambda_Q = 0
    """
    params_Q = params.copy()
    params_Q['gamma'] = params['gamma'] + params['lambda']
    params_Q['lambda'] = 0.0
    return params_Q


def simulate_ngarch_paths_Q_FHS(
    S0,
    h0,
    rf_d,
    y_d,
    params_Q,
    eps_array,
    q_weights,
    lam,
    n_days=21,
    n_paths=20_000,
    random_state=None,
):
    """
    Simulate NGARCH paths under Q using FHS shocks.
    """
    rng = np.random.default_rng(random_state)

    omega = params_Q['omega']
    alpha = params_Q['alpha']
    beta  = params_Q['beta']
    gammaQ = params_Q['gamma']

    S = np.full(n_paths, S0, dtype=float)
    h = np.full(n_paths, h0, dtype=float)

    for _ in range(n_days):
        eps_Q = draw_eps_Q(eps_array, q_weights, lam, n_paths, rng)

        # Excess return under Q (Duan LRNVR form)
        r_excess = -0.5 * h + np.sqrt(h) * eps_Q

        # Log return with carry adjustment
        log_ret = r_excess - y_d + rf_d
        S *= np.exp(log_ret)

        # NGARCH variance recursion
        h = omega + alpha * h * (eps_Q - gammaQ) ** 2 + beta * h

    return S, h


def rmse(y_true, y_pred):
    diff = y_pred - y_true
    return np.sqrt(np.mean(diff ** 2))


# ============================================================
# Main entry point for Question 5
# ============================================================

def run_q5_analysis(
    r,
    h,
    eps,
    params,
    vix_close,
    spx,
    rf_daily,
    y_daily,
    vix_model_gauss=None,
    n_paths=20_000,
    horizon_days=21,
):
    """
    Run Question 5 FHS-based VIX analysis (figures + diagnostics).
    """
    # ---------- Prep: Duan annualized vol, shocks, RN params ----------
    h_series = pd.Series(h, index=r.index, name="h")
    duan_vol_ann = np.sqrt(252.0 * h_series) * 100.0  # annualized %

    eps_array = np.asarray(eps)
    lam = params['lambda']

    # Q-weights from exponential tilt
    q_weights = compute_Q_weights(eps_array, lam)

    # Risk-neutral NGARCH params
    params_Q = get_rn_params(params)

    # ---------- FHS-based model VIX path simulation ----------
    vix_start = vix_close.index.min()
    vix_end   = vix_close.index.max()
    common_dates = r.index[(r.index >= vix_start) & (r.index <= vix_end)]

    # Wednesdays only (weekday == 2)
    wed_dates = common_dates[common_dates.weekday == 2]

    m_grid = build_m_grid()
    vix_model_fhs = pd.Series(index=wed_dates, dtype=float)

    for d in wed_dates:
        S0   = spx.loc[d, 'spindx']
        h0   = h_series.loc[d]
        rf_d = rf_daily.loc[d]
        y_d  = y_daily.loc[d]

        S_T, h_T = simulate_ngarch_paths_Q_FHS(
            S0, h0, rf_d, y_d,
            params_Q, eps_array, q_weights, lam,
            n_days=horizon_days,
            n_paths=n_paths,
        )

        F, df, R_f, T = forward_and_discount(S0, rf_d, y_d, T_days=30)
        strikes, C, P = price_options_from_paths(S_T, F, h0, m_grid, df)
        VIX_fhs, _    = vix_from_option_curve(strikes, C, P, F, R_f, T)

        vix_model_fhs.loc[d] = VIX_fhs

    # ============================================================
    # Question 5 – Graphs & Diagnostics
    # ============================================================

    # ---------- 0) FHS-based model VIX vs physical vol and actual VIX ----------
    cols = [
        vix_model_fhs.rename("VIX_FHS"),
        duan_vol_ann.rename("Vol_phys"),
        vix_close.rename("VIX_actual"),
    ]
    plot_df_q5 = pd.concat(cols, axis=1, join="inner").dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_df_q5.plot(ax=ax, color=["steelblue", "red", "black"])
    ax.set_title("FHS-based model VIX vs physical volatility and actual VIX (Wednesdays)")
    ax.set_ylabel("Volatility index level")
    ax.set_xlabel("Date")
    plt.tight_layout()
    plt.show()

    print("Q5 correlation matrix:")
    print(plot_df_q5.corr())
    print("\nQ5 summary statistics:")
    print(plot_df_q5.describe())

    # ---------- 1) Q-shocks: Histogram + KDE (Gaussian vs FHS) ----------
    rng_vis = np.random.default_rng(123)
    n_shocks_vis = 50_000

    # FHS Q-shocks (tilted historical)
    eps_Q_fhs = draw_eps_Q(eps_array, q_weights, lam, n_shocks_vis, rng_vis)

    # Gaussian Q-shocks for comparison
    eps_Q_gauss = rng_vis.standard_normal(n_shocks_vis)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histograms
    axes[0].hist(
        eps_Q_gauss, bins=50, density=True, alpha=0.4,
        color="black", label="Gaussian Q-shocks",
    )
    axes[0].hist(
        eps_Q_fhs, bins=50, density=True, alpha=0.6,
        color="steelblue", label="FHS Q-shocks",
    )
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Histogram of Q-shocks: Gaussian vs FHS")
    axes[0].set_xlabel("Shock")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # KDEs
    pd.Series(eps_Q_gauss, name="Gaussian").plot(
        kind="kde", ax=axes[1], color="black", linewidth=1.5
    )
    pd.Series(eps_Q_fhs, name="FHS").plot(
        kind="kde", ax=axes[1], color="steelblue", linewidth=1.5
    )
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("KDE of Q-shocks: Gaussian vs FHS")
    axes[1].set_xlabel("Shock")

    plt.tight_layout()
    plt.show()

    # ---------- 2) RMSE table: Gaussian vs FHS vs Duan ----------
    rmse_cols = {
        "VIX_actual": vix_close,
        "VIX_FHS":    vix_model_fhs,
        "Vol_phys":   duan_vol_ann,
    }

    if vix_model_gauss is not None:
        rmse_cols["VIX_Gauss"] = vix_model_gauss

    rmse_df = pd.concat(rmse_cols.values(), axis=1, join="inner")
    rmse_df.columns = list(rmse_cols.keys())

    model_cols = [c for c in rmse_df.columns if c != "VIX_actual"]
    rmse_vals = {name: rmse(rmse_df["VIX_actual"], rmse_df[name])
                 for name in model_cols}

    rmse_table = pd.DataFrame.from_dict(rmse_vals, orient="index", columns=["RMSE"])
    print("\nRMSE vs VIX_actual:")
    print(rmse_table)

    # ---------- 3) Scatter: model VIX vs market VIX ----------
    fig, ax = plt.subplots(figsize=(6, 6))

    # FHS points
    ax.scatter(
        rmse_df["VIX_FHS"], rmse_df["VIX_actual"],
        alpha=0.4, color="steelblue", edgecolors="black", label="FHS",
    )

    # Gaussian points if available
    if "VIX_Gauss" in rmse_df.columns:
        ax.scatter(
            rmse_df["VIX_Gauss"], rmse_df["VIX_actual"],
            alpha=0.3, color="red", edgecolors="none", label="Gaussian",
        )

    # Regression line for FHS
    valid = rmse_df[["VIX_FHS", "VIX_actual"]].dropna()
    x_fhs = valid["VIX_FHS"].values
    y_mkt = valid["VIX_actual"].values
    b1, b0 = np.polyfit(x_fhs, y_mkt, 1)  # y = b0 + b1 x

    x_line = np.linspace(x_fhs.min(), x_fhs.max(), 100)
    ax.plot(x_line, b0 + b1 * x_line, color="red", linewidth=2, label="FHS regression")

    # 45° line
    xy_min = min(valid.min().min(), valid["VIX_actual"].min())
    xy_max = max(valid.max().max(), valid["VIX_actual"].max())
    ax.plot([xy_min, xy_max], [xy_min, xy_max],
            color="black", linestyle="--", linewidth=1)

    ax.set_xlabel("Model-based VIX")
    ax.set_ylabel("Market VIX")
    ax.set_title("Model-based vs Market VIX")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # ---------- 4) VIX pricing error with FHS (time-series) ----------
    err_fhs = plot_df_q5["VIX_FHS"] - plot_df_q5["VIX_actual"]
    err_fhs.name = "FHS_VIX_error"

    fig, ax = plt.subplots(figsize=(12, 4))
    err_fhs.plot(ax=ax, color="steelblue", linewidth=1)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, label="Zero error")
    ax.axhline(err_fhs.mean(), color="red", linestyle="-", linewidth=1.5,
               label="Mean error")

    ax.set_title("FHS VIX pricing error (Model - Market)")
    ax.set_ylabel("Error (index points)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\nFHS pricing error summary:")
    print(err_fhs.describe())

    bias     = err_fhs.mean()
    mae      = err_fhs.abs().mean()
    rmse_err = np.sqrt(np.mean(err_fhs ** 2))
    print(f"\nBias (mean error): {bias}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse_err}")

    # ---------- 5) Distribution of FHS pricing error (hist only) ----------
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(err_fhs, bins=30, density=True, alpha=0.7,
            color="steelblue", edgecolor="black")
    ax.axvline(err_fhs.mean(), color="red", linewidth=2, label="Mean error")

    ax.set_title("Distribution of VIX pricing errors (Model - Market)")
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        "vix_model_fhs": vix_model_fhs,
        "plot_df_q5":    plot_df_q5,
        "rmse_table":    rmse_table,
        "errors_fhs":    err_fhs,
        "rmse_err_fhs":  rmse_err,
        "eps_Q_fhs":     eps_Q_fhs,
        "eps_Q_gauss":   eps_Q_gauss,
    }
