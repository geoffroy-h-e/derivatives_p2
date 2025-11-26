import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#  Helper functions 

def get_rn_params(params):

    params_Q = params.copy()
    params_Q["gamma"]  = params["gamma"] + params["lambda"]
    params_Q["lambda"] = 0.0
    return params_Q


def simulate_ngarch_paths_Q(S0, h0, rf_d, y_d, params_Q,
                            n_days=21, n_paths=20000, random_state=None):
    
    rng = np.random.default_rng(random_state)
    omega = params_Q["omega"]
    alpha = params_Q["alpha"]
    beta  = params_Q["beta"]
    gammaQ = params_Q["gamma"]

    # antithetic variates
    n_half = n_paths // 2
    n_paths = 2 * n_half
    S = np.full(n_paths, S0, dtype=float)
    h = np.full(n_paths, h0, dtype=float)

    for _ in range(n_days):
        z_half = rng.standard_normal(n_half)
        eps = np.concatenate([z_half, -z_half])
        eps = (eps - eps.mean()) / eps.std(ddof=1)

        r_excess = -0.5 * h + np.sqrt(h) * eps
        log_ret = r_excess - y_d + rf_d
        S *= np.exp(log_ret)

        h = omega + alpha * h * (eps - gammaQ) ** 2 + beta * h

    return S, h


def build_m_grid(m_min=-5.0, m_max=5.0, n_m=41):
    """Grid of standardized moneyness m."""
    return np.linspace(m_min, m_max, n_m)


def forward_and_discount(S0, rf_d, y_d, T_days=30):
    """
    Forward price, discount factor and extras for a T_days horizon.
    All rates are daily continuously-compounded.
    """
    R_f = rf_d * T_days      # cumulated risk-free
    Q   = y_d  * T_days      # cumulated dividend yield
    T   = T_days / 365.0
    F   = S0 * np.exp(R_f - Q)
    df  = np.exp(-R_f)
    return F, df, R_f, T


def price_options_from_paths(S_T, F, h0, m_grid, df):
    """
    Price calls and puts on a grid of standardized moneyness m,
    given simulated terminal prices S_T.
    """
    S_T = np.asarray(S_T)
    sigma_bar = np.sqrt((252.0 / 12.0) * h0)
    strikes = F * np.exp(m_grid * sigma_bar)

    call_prices = []
    put_prices  = []
    for K in strikes:
        payoff_call = np.maximum(S_T - K, 0.0)
        payoff_put  = np.maximum(K - S_T, 0.0)
        call_prices.append(df * payoff_call.mean())
        put_prices.append(df * payoff_put.mean())

    return strikes, np.array(call_prices), np.array(put_prices)


def vix_from_option_curve(strikes, call_prices, put_prices, F, R_f, T):
    """
    CBOE-style VIX computation for a single maturity.
    Returns (VIX_level, variance).
    """
    K = np.asarray(strikes)
    C = np.asarray(call_prices)
    P = np.asarray(put_prices)

    # strike closest to forward
    idx0 = np.argmin(np.abs(K - F))
    K0 = K[idx0]

    # OTM option prices Q(K)
    QK = np.zeros_like(K)
    for i, Ki in enumerate(K):
        if Ki < K0:
            QK[i] = P[i]
        elif Ki > K0:
            QK[i] = C[i]
        else:
            QK[i] = 0.5 * (C[i] + P[i])

    # deltaK spacing
    deltaK = np.zeros_like(K)
    deltaK[0]  = K[1] - K[0]
    deltaK[-1] = K[-1] - K[-2]
    deltaK[1:-1] = (K[2:] - K[:-2]) / 2.0

    # variance
    sigma2 = (2.0 * np.exp(R_f) / T) * np.sum(deltaK * QK / (K**2)) \
             - (1.0 / T) * ((F / K0 - 1.0) ** 2)

    sigma2 = max(sigma2, 0.0)
    VIX = 100.0 * np.sqrt(sigma2)
    return VIX, sigma2


#  Main bloc  

def run_model_vix_analysis(
    h,
    r,
    params,
    spx,
    rf_daily,
    y_daily,
    vix_close,
    n_paths=20000,
    horizon_days=21,
    rolling_weeks=52,
    make_plots=True,
    random_state=None,
):

    # physical volatility from NGARCH
    h_series = pd.Series(h, index=r.index, name="h")
    duan_vol_ann = np.sqrt(252.0 * h_series) * 100.0   # 100× physical NGARCH vol

    # risk-neutral parameters
    params_Q = get_rn_params(params)

    # dates where both NGARCH and VIX are available
    vix_start = vix_close.index.min()
    vix_end   = vix_close.index.max()
    common_dates = r.index[(r.index >= vix_start) & (r.index <= vix_end)]

    # Wednesdays only
    wed_dates = common_dates[common_dates.weekday == 2]

    m_grid = build_m_grid()
    model_vix = pd.Series(index=wed_dates, dtype=float)

    for d in wed_dates:
        S0   = spx.loc[d, "spindx"]
        h0   = h_series.loc[d]
        rf_d = rf_daily.loc[d]
        y_d  = y_daily.loc[d]

        # simulate under Q
        S_T, _ = simulate_ngarch_paths_Q(
            S0, h0, rf_d, y_d, params_Q,
            n_days=horizon_days, n_paths=n_paths, random_state=random_state
        )

        F, df, R_f, T = forward_and_discount(S0, rf_d, y_d, T_days=30)
        strikes, C, P = price_options_from_paths(S_T, F, h0, m_grid, df)
        VIX_mod, _ = vix_from_option_curve(strikes, C, P, F, R_f, T)

        model_vix.loc[d] = VIX_mod

    # main comparison dataframe
    plot_df = pd.concat(
        [
            model_vix.rename("VIX_model"),
            duan_vol_ann.rename("Vol_phys"),
            vix_close.rename("VIX_actual"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    # Error series + rolling statistics (using Wednesdays subset)
    scatter_df = plot_df[["VIX_model", "VIX_actual"]].rename(
        columns={"VIX_actual": "VIX_mkt"}
    ).dropna()

    errors = scatter_df["VIX_model"] - scatter_df["VIX_mkt"]
    roll_corr = scatter_df["VIX_model"].rolling(rolling_weeks).corr(
        scatter_df["VIX_mkt"]
    )
    roll_bias = errors.rolling(rolling_weeks).mean()

    # error distribution metrics
    rmse = np.sqrt((errors**2).mean())
    mae  = np.abs(errors).mean()
    bias = errors.mean()

    # OLS via numpy
    X = scatter_df["VIX_model"].values
    Y = scatter_df["VIX_mkt"].values
    A = np.vstack([np.ones_like(X), X]).T
    beta_hat, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)
    alpha_hat, slope_hat = beta_hat

    y_hat  = alpha_hat + slope_hat * X
    ss_res = np.sum((Y - y_hat)**2)
    ss_tot = np.sum((Y - Y.mean())**2)
    r2 = 1 - ss_res / ss_tot

    if make_plots:
        # ---------- Time-series comparison ----------
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(plot_df.index, plot_df["VIX_model"],
                label="Model-based VIX", color="steelblue")
        ax.plot(plot_df.index, plot_df["Vol_phys"],
                label="Physical volatility (Duan)", color="red")
        ax.plot(plot_df.index, plot_df["VIX_actual"],
                label="Market VIX", color="black")
        ax.set_title("Model-based VIX vs physical volatility and actual VIX")
        ax.set_ylabel("Volatility index level")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

        print("Q3 correlation matrix:")
        print(plot_df.corr())
        print("\nQ3 summary statistics:")
        print(plot_df.describe())

        # ---------- Error series + rolling statistics ----------
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # levels
        axes[0].plot(scatter_df.index, scatter_df["VIX_mkt"],
                     label="Market VIX", color="black")
        axes[0].plot(scatter_df.index, scatter_df["VIX_model"],
                     label="Model-based VIX", color="steelblue")
        axes[0].set_ylabel("VIX level")
        axes[0].set_title("Market vs Model-based VIX")
        axes[0].legend(loc="upper left")

        # errors
        axes[1].plot(errors.index, errors, color="red",
                     label="Error (Model - Market)")
        axes[1].axhline(0.0, color="black", lw=0.8)
        axes[1].set_ylabel("Error")
        axes[1].set_title("VIX Pricing Error: Model - Market")
        axes[1].legend(loc="upper left")

        # rolling stats
        axes[2].plot(roll_corr.index, roll_corr,
                     label=f"Rolling corr ({rolling_weeks} weeks)",
                     color="steelblue")
        axes[2].plot(roll_bias.index, roll_bias,
                     label=f"Rolling mean error ({rolling_weeks} weeks)",
                     color="red", linestyle="--")
        axes[2].axhline(0.0, color="black", lw=0.8)
        axes[2].set_ylabel("Rolling stats")
        axes[2].set_xlabel("Date")
        axes[2].set_title("Rolling Correlation and Bias")
        axes[2].legend(loc="upper left")

        plt.tight_layout()
        plt.show()

        # ---------- Distribution of errors ----------
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(errors, bins=40, density=True, alpha=0.6, color="steelblue")
        ax.axvline(errors.mean(), color="red", linewidth=2, label="Mean error")
        ax.set_title("Distribution of VIX pricing errors (Model - Market)")
        ax.set_xlabel("Error")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

        print("Bias (mean error):", bias)
        print("MAE:", mae)
        print("RMSE:", rmse)

        # ---------- Scatter + regression ----------
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            scatter_df["VIX_model"],
            scatter_df["VIX_mkt"],
            s=15,
            alpha=0.35,
            color="steelblue",
            edgecolors="black",
        )

        lims = [0, max(scatter_df.values.max(), 1)]
        ax.plot(lims, lims, "k--", lw=1)

        ax.set_xlabel("Model-based VIX")
        ax.set_ylabel("Market VIX")
        ax.set_title("Model-based vs Market VIX")

        x_line = np.linspace(X.min(), X.max(), 100)
        y_line = alpha_hat + slope_hat * x_line
        ax.plot(x_line, y_line, color="red")

        plt.tight_layout()
        plt.show()

        print(f"alpha (intercept): {alpha_hat:.4f}")
        print(f"beta  (slope)    : {slope_hat:.4f}")
        print(f"R^2              : {r2:.4f}")

    # collect outputs
    results = {
        "h_series": h_series,
        "duan_vol_ann": duan_vol_ann,
        "model_vix": model_vix,
        "plot_df": plot_df,
        "scatter_df": scatter_df,
        "errors": errors,
        "roll_corr": roll_corr,
        "roll_bias": roll_bias,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "alpha": alpha_hat,
        "beta": slope_hat,
        "r2": r2,
    }

    return results
