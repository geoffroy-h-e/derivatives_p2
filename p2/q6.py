# q6.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# rmse
def rmse(y_true, y_pred):
    diff = y_pred - y_true
    return np.sqrt(np.mean(diff**2))


def simulate_bates_paths_Q(
    S0,
    v0,
    rf_d,
    y_d,
    params,
    n_days=21,
    n_paths=20_000,
    random_state=None,
):

    kappa   = params["kappa"]
    theta   = params["theta"]
    sigma_v = params["sigma_v"]
    rho     = params["rho"]
    mu_J    = params["mu_J"]
    sig_J   = params["sigma_J"]

    # annual jump intensity 
    lambda_annual = params["lambdaJ_annual"]
    lambda_daily  = lambda_annual / 252.0

    # log-normal jumps
    mu_hat_J = np.exp(mu_J + 0.5 * sig_J**2) - 1.0

    rng = np.random.default_rng(random_state)

    S = np.full(n_paths, float(S0))
    v = np.full(n_paths, float(v0))

    dt = 1.0 

    for _ in range(n_days):
        # Brownian increments
        z1 = rng.standard_normal(n_paths)
        z2_indep = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1.0 - rho**2) * z2_indep

        # Poisson jumps per path
        N_jump = rng.poisson(lambda_daily * dt, size=n_paths)

        # Sum of log-jumps per path
        J_sum = np.zeros(n_paths)
        has_jump = N_jump > 0
        if np.any(has_jump):
            m = N_jump[has_jump]
            J_sum[has_jump] = rng.normal(
                loc=mu_J * m,
                scale=sig_J * np.sqrt(m)
            )

        v_clipped = np.clip(v, 1e-10, None)

        # Diffusion + jumps 
        drift = (rf_d - y_d - lambda_daily * mu_hat_J - 0.5 * v_clipped) * dt
        diff  = np.sqrt(v_clipped * dt) * z1
        S *= np.exp(drift + diff + J_sum)

        # variance update
        v = v + kappa * (theta - v) * dt + sigma_v * np.sqrt(v_clipped * dt) * z2
        v = np.clip(v, 1e-10, None)

    return S


def run_bates_vix_series(
    spx,
    rf_daily,
    y_daily,
    h,
    r,
    vix_close,
    bates_params,
    vix_fhs=None,       
    vix_gauss=None,     
    n_paths=20_000,
    horizon_days=21,
    make_plot=True,
    make_error_plot=True,
):

    from q3 import (
        build_m_grid,
        forward_and_discount,
        price_options_from_paths,
        vix_from_option_curve,
    )

    # Physical measure vol (Duan NGARCH)
    h_series = pd.Series(h, index=r.index, name="h")
    duan_vol_ann = np.sqrt(252.0 * h_series) * 100.0  # %

    # Dates where VIX exists
    vix_start = vix_close.index.min()
    vix_end   = vix_close.index.max()
    common_dates = r.index[(r.index >= vix_start) & (r.index <= vix_end)]
    wed_dates = common_dates[common_dates.weekday == 2]

    m_grid = build_m_grid()
    vix_bates = pd.Series(index=wed_dates, dtype=float)

    for d in wed_dates:
        S0   = spx.loc[d, "spindx"]
        v0   = h_series.loc[d]
        rf_d = rf_daily.loc[d]
        y_d  = y_daily.loc[d]

        S_T = simulate_bates_paths_Q(
            S0=S0,
            v0=v0,
            rf_d=rf_d,
            y_d=y_d,
            params=bates_params,
            n_days=horizon_days,
            n_paths=n_paths,
            random_state=None,
        )
        F, df, R_f, T = forward_and_discount(S0, rf_d, y_d, T_days=30)
        strikes, C, P = price_options_from_paths(S_T, F, v0, m_grid, df)
        VIX_b, _      = vix_from_option_curve(strikes, C, P, F, R_f, T)

        vix_bates.loc[d] = VIX_b

    # build a df with results and series
    plot_df = pd.concat(
        [
            vix_bates.rename("VIX_Bates"),
            duan_vol_ann.rename("Vol_phys"),
            vix_close.rename("VIX_actual"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if vix_fhs is not None:
        plot_df = plot_df.join(vix_fhs.rename("VIX_FHS"), how="inner")

    if vix_gauss is not None:
        plot_df = plot_df.join(vix_gauss.rename("VIX_Gauss"), how="inner")

    if make_plot:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(plot_df.index, plot_df["VIX_Bates"],
                label="VIX_Bates", color="steelblue")
        ax.plot(plot_df.index, plot_df["Vol_phys"],
                label="Vol_phys", color="red")
        ax.plot(plot_df.index, plot_df["VIX_actual"],
                label="VIX_actual", color="black")

        ax.set_ylabel("Volatility (%)")
        ax.set_xlabel("Date")
        ax.set_title("Bates(2000) model VIX vs physical and market volatility")
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    print("\nCorrelation matrix (Bates, phys, actual, FHS, Gaussian):")
    print(plot_df.corr())

    #  Bates pricing error plot
    err_bates = plot_df["VIX_Bates"] - plot_df["VIX_actual"]
    err_bates.name = "Bates_VIX_error"

    if make_error_plot:
        fig, ax = plt.subplots(figsize=(12, 4))
        err_bates.plot(ax=ax, color="steelblue", linewidth=1,
                       label="Bates_VIX_error")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1,
                   label="Zero error")
        ax.axhline(err_bates.mean(), color="red", linestyle="-", linewidth=1.5,
                   label="Mean error")

        ax.set_title("Bates VIX pricing error (Model - Market)")
        ax.set_ylabel("Error (index points)")
        ax.set_xlabel("Date")
        ax.legend()
        plt.tight_layout()
        plt.show()

        print("\nBates pricing error summary:")
        print(err_bates.describe())


    print("\nDescriptive statistics:")
    print(plot_df.describe())

    rmse_bates = rmse(plot_df["VIX_actual"], plot_df["VIX_Bates"])
    print("\nRMSE vs VIX_actual:")
    print(f"  Bates VIX : {rmse_bates:.4f}")

    rmse_fhs = None
    if "VIX_FHS" in plot_df.columns:
        rmse_fhs = rmse(plot_df["VIX_actual"], plot_df["VIX_FHS"])
        print(f"  FHS VIX   : {rmse_fhs:.4f}")

    rmse_gauss = None
    if "VIX_Gauss" in plot_df.columns:
        rmse_gauss = rmse(plot_df["VIX_actual"], plot_df["VIX_Gauss"])
        print(f"  Gaussian VIX    : {rmse_gauss:.4f}")

    return {
        "plot_df": plot_df,
        "vix_bates": vix_bates,
        "duan_vol_ann": duan_vol_ann,
        "errors_bates": err_bates,
        "rmse_bates": rmse_bates,
        "rmse_fhs": rmse_fhs,
        "rmse_gauss": rmse_gauss,
    }

# please note that these parameters were found in various papers (Bates (2000), Duffie(2000), Agazzotti (2025), Cape (2014)) 
# not optimization routine was ran to find those. 

bates_params = {
    "kappa": 0.3 / 252.0,
    "theta": (0.04 / 252.0),
    "sigma_v": 0.05 / np.sqrt(252.0),
    "rho": -0.7,
    "lambdaJ_annual": 0.3,
    "mu_J": -0.1,
    "sigma_J": 0.15,
}
