# ngarch_q1.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

# ============================================================
# 1. Helper functions
# ============================================================

def annual_to_daily(y_annual_decimal, days_per_year=252):
    """
    Convert an annualized yield (decimal, e.g. 0.03 = 3%)
    to a daily yield by simple scaling.
    """
    return y_annual_decimal / days_per_year


def build_dividend_yield_series(trading_days,
                                damodaran_path='damodaran_divyields.csv',
                                spx_opt_div_path='spx_divyields.csv'):
    """
    Build a daily dividend yield series y_t aligned with 'trading_days'.
    """
    dam = pd.read_csv(damodaran_path)

    if {'Year', 'Dividend_Yield'} <= set(dam.columns):
        dam['date'] = pd.to_datetime(dam['Year'].astype(str) + '-12-31')

        div_y_pct = (
            dam['Dividend_Yield']
            .astype(str)
            .str.replace('%', '', regex=False)
            .astype(float)
        )
        dam['div_y_annual_decimal'] = div_y_pct / 100.0

        dam = dam[['date', 'div_y_annual_decimal']]
    else:
        raise ValueError(
            "Damodaran file must have columns 'Year' and 'Dividend_Yield'. "
            f"Found: {dam.columns.tolist()}"
        )

    dam.set_index('date', inplace=True)
    dam = dam.reindex(trading_days, method='ffill')
    dam['y_daily'] = annual_to_daily(dam['div_y_annual_decimal'])

    # ---------- Option-implied part (from 1996-01-04 onward) ----------
    opt = pd.read_csv(spx_opt_div_path,
                      parse_dates=['date', 'expiration'])

    opt['days_to_expiry'] = (opt['expiration'] - opt['date']).dt.days
    opt = opt[opt['days_to_expiry'] > 0]

    opt.sort_values(['date', 'days_to_expiry'], inplace=True)
    opt_short = opt.groupby('date').first().reset_index()

    opt_short = opt_short[['date', 'rate']]
    opt_short.set_index('date', inplace=True)
    opt_short = opt_short.reindex(trading_days, method='ffill')

    opt_short['div_y_annual_decimal'] = opt_short['rate'] / 100.0
    opt_short['y_daily'] = annual_to_daily(opt_short['div_y_annual_decimal'])

    cutoff = pd.Timestamp('1996-01-04')
    y = pd.Series(index=trading_days, dtype=float)

    pre = trading_days < cutoff
    post = ~pre

    y[pre] = dam.loc[trading_days[pre], 'y_daily']
    y[post] = opt_short.loc[trading_days[post], 'y_daily']

    return y


def ngarch_loglik(params, r):
    """
    Negative log-likelihood for Duan's NGARCH(1,1):
        r_t   = λ * sqrt(h_t) - 0.5 * h_t + sqrt(h_t) ε_t
        h_t   = ω + α * h_{t-1} * (ε_{t-1} - γ)^2 + β * h_{t-1}
    """
    lam, omega, alpha, beta, gamma = params

    if omega <= 0 or alpha < 0 or beta < 0:
        return 1e12

    if alpha * (1 + gamma ** 2) + beta >= 0.999:
        return 1e12

    r = np.asarray(r)
    T = len(r)
    h = np.empty(T)
    eps = np.empty(T)

    var_r = np.var(r, ddof=1)
    h[0] = var_r

    max_h = 1e4 * var_r
    max_eps2 = 1e6

    loglik = 0.0
    two_pi = 2.0 * np.pi

    for t in range(T):
        if t > 0:
            z = eps[t - 1] - gamma
            if not np.isfinite(z):
                return 1e12

            h_t = omega + alpha * h[t - 1] * (z * z) + beta * h[t - 1]

            if (h_t <= 0) or (h_t > max_h) or (not np.isfinite(h_t)):
                return 1e12

            h[t] = h_t

        mu_t = lam * np.sqrt(h[t]) - 0.5 * h[t]
        eps[t] = (r[t] - mu_t) / np.sqrt(h[t])

        if (not np.isfinite(eps[t])) or (eps[t] * eps[t] > max_eps2):
            return 1e12

        loglik += 0.5 * (np.log(two_pi) + np.log(h[t]) + eps[t] * eps[t])

    return loglik


# ---------- Multi-start helpers ----------

def generate_ngarch_starting_points(r):
    r = np.asarray(r)
    var_r = np.var(r, ddof=1)

    lambda_grid = [0.0, 0.3]
    alpha_grid  = [0.03, 0.06]
    beta_grid   = [0.92, 0.97]
    gamma_grid  = [-0.2, 0.2]

    starts = []

    for lam in lambda_grid:
        for alpha in alpha_grid:
            for beta in beta_grid:
                for gamma in gamma_grid:
                    rho = alpha * (1.0 + gamma**2) + beta
                    if rho >= 0.999:
                        continue

                    omega = var_r * (1.0 - rho)
                    if omega <= 0:
                        continue

                    starts.append(np.array([lam, omega, alpha, beta, gamma], dtype=float))

    if not starts:
        starts.append(
            np.array([0.1, var_r * 0.1, 0.05, 0.9, 0.1], dtype=float)
        )

    return starts


def estimate_ngarch_multistart(r, start_params_list=None):
    """
    Estimate NGARCH(1,1) via multi-start quasi-MLE, *without* plotting.

    Returns
    -------
    params : dict
    h      : np.ndarray
    eps    : np.ndarray
    opt_res: OptimizeResult
    """
    r = np.asarray(r)

    if start_params_list is None:
        start_params_list = generate_ngarch_starting_points(r)

    bounds = [
        (-5, 5),
        (1e-10, None),
        (1e-6, 1 - 1e-3),
        (1e-6, 1 - 1e-3),
        (-5, 5)
    ]

    best_res = None
    best_x = None

    for x0 in start_params_list:
        x0 = np.array(x0, dtype=float)
        for j, (low, high) in enumerate(bounds):
            if low is not None and x0[j] < low:
                x0[j] = low + 1e-8
            if high is not None and x0[j] > high:
                x0[j] = high - 1e-8

        res = minimize(
            ngarch_loglik,
            x0,
            args=(r,),
            method='L-BFGS-B',
            bounds=bounds
        )

        if not res.success:
            continue

        if (best_res is None) or (res.fun < best_res.fun):
            best_res = res
            best_x = res.x

    if best_res is None:
        raise RuntimeError("NGARCH multi-start optimization failed for all starting points.")

    lam, omega, alpha, beta, gamma = best_x

    T = len(r)
    h = np.empty(T)
    eps = np.empty(T)
    h[0] = np.var(r, ddof=1)

    for t in range(T):
        if t > 0:
            h[t] = omega + alpha * h[t - 1] * (eps[t - 1] - gamma) ** 2 + beta * h[t - 1]
        mu_t = lam * np.sqrt(h[t]) - 0.5 * h[t]
        eps[t] = (r[t] - mu_t) / np.sqrt(h[t])

    params = {
        'lambda': lam,
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'sigma2_uncond': omega / (1 - alpha * (1 + gamma ** 2) - beta)
    }

    return params, h, eps, best_res


def anderson_darling_uniform(u):
    u = np.asarray(u)
    eps = 1e-10
    u = np.clip(u, eps, 1 - eps)
    u_sorted = np.sort(u)
    n = len(u_sorted)
    i = np.arange(1, n + 1)

    s = np.sum((2 * i - 1) / n * (np.log(u_sorted) +
                                  np.log(1 - u_sorted[::-1])))
    A2 = -n - s
    return A2


def acf_1d(x, max_lag):
    """
    Simple ACF for a 1D array up to lag max_lag.
    Returns array of length max_lag with acf[lag-1] = rho_lag.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    T = len(x)
    denom = np.dot(x, x)

    ac = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        num = np.dot(x[:-lag], x[lag:])
        ac[lag - 1] = num / denom
    return ac


# ============================================================
# 2. High-level functions (callable from other files)
# ============================================================

def estimate_ngarch_q1(spx_path='spx.csv',
                       damodaran_path='damodaran_divyields.csv',
                       spx_div_path='spx_divyields.csv'):
    """
    Load data from disk, build cum-dividend excess returns, and estimate NGARCH.

    Returns
    -------
    spx, y_daily, rf_daily, r, h, eps, params, opt_res
    """
    spx = pd.read_csv(spx_path, parse_dates=['date'])
    spx.sort_values('date', inplace=True)
    spx.set_index('date', inplace=True)

    if not {'spindx', 'rf'} <= set(spx.columns):
        raise ValueError("spx.csv must have columns: date, spindx, rf")

    trading_days = spx.index

    y_daily = build_dividend_yield_series(
        trading_days,
        damodaran_path=damodaran_path,
        spx_opt_div_path=spx_div_path
    )

    rf_daily = spx['rf']

    log_ret = np.log(spx['spindx'] / spx['spindx'].shift(1))
    r = log_ret + y_daily.shift(1) - rf_daily.shift(1)
    r = r.dropna()

    params, h, eps, opt_res = estimate_ngarch_multistart(r)

    return spx, y_daily, rf_daily, r, h, eps, params, opt_res


def plot_ngarch_diagnostics(eps, make_qq=True, make_hist=True,
                            make_acf=True, make_pit=True):
    """
    Given standardized residuals eps, produce QQ-plot, histogram+pdf,
    ACF of eps^2, and PIT histogram.
    """
    eps = np.asarray(eps)

    # ---------- QQ plot ----------
    if make_qq:
        fig, ax = plt.subplots()

        res = stats.probplot(eps, dist="norm", plot=None)
        osm, osr = res[0]
        line_params = res[1]

        if len(line_params) == 2:
            slope, intercept = line_params
        else:
            slope, intercept, _ = line_params

        ax.scatter(osm, osr, color="steelblue", s=10)
        ax.plot(osm, slope * osm + intercept, color="red", linewidth=2)

        ax.set_title("QQ-plot of standardized residuals $\epsilon_t$")
        ax.set_xlabel("Theoretical quantiles (N(0,1))")
        ax.set_ylabel("Sample quantiles")

        plt.tight_layout()
        plt.show()

    # ---------- Histogram of eps with N(0,1) pdf ----------
    if make_hist:
        fig1, ax1 = plt.subplots()
        ax1.hist(eps, bins=50, density=True, alpha=0.6, edgecolor="black")
        x_grid = np.linspace(eps.min(), eps.max(), 500)
        ax1.plot(x_grid, stats.norm.pdf(x_grid), color="red", linewidth=2)
        ax1.set_title("Distribution of $\epsilon_t$ with N(0,1) PDF")
        ax1.set_xlabel(r"$\epsilon_t$")
        ax1.set_ylabel("Density")
        plt.tight_layout()
        plt.show()

    # ---------- ACF of eps^2 ----------
    if make_acf:
        eps2 = eps ** 2
        max_lag = 20
        ac_vals = acf_1d(eps2, max_lag)
        lags = np.arange(1, max_lag + 1)
        conf = 1.96 / np.sqrt(len(eps2))

        fig2, ax2 = plt.subplots()
        ax2.stem(lags, ac_vals, basefmt=" ")
        ax2.axhline(0.0, color="black", linewidth=1)
        ax2.axhline(conf, color="gray", linestyle="--", linewidth=1)
        ax2.axhline(-conf, color="gray", linestyle="--", linewidth=1)
        ax2.set_title(r"Autocorrelation of squared residuals $\epsilon_t^2$")
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("ACF")
        plt.tight_layout()
        plt.show()

    # ---------- PIT histogram ----------
    if make_pit:
        u = stats.norm.cdf(eps)
        fig3, ax3 = plt.subplots()
        ax3.hist(u, bins=20, range=(0.0, 1.0),
                 density=True, alpha=0.7, edgecolor="black")
        ax3.axhline(1.0, color="red", linewidth=2)
        ax3.set_title(r"PIT histogram of $u_t = \Phi(\epsilon_t)$")
        ax3.set_xlabel(r"$u_t$")
        ax3.set_ylabel("Density")
        plt.tight_layout()
        plt.show()


def run_q1(spx_path='spx.csv',
           damodaran_path='damodaran_divyields.csv',
           spx_div_path='spx_divyields.csv'):
    """
    Convenience wrapper: estimate NGARCH and immediately plot diagnostics.
    """
    spx, y_daily, rf_daily, r, h, eps, params, opt_res = \
        estimate_ngarch_q1(spx_path, damodaran_path, spx_div_path)

    print("Estimated NGARCH(1,1) parameters (multi-start):")
    for k, v in params.items():
        print(f"  {k:>15s} = {v: .6g}")
    print(f"\nConverged: {opt_res.success}")

    plot_ngarch_diagnostics(eps)

    # residual diagnostics printout
    print("\n========== Residual diagnostics for epsilon t ==========\n")
    jb_stat, jb_pvalue = stats.jarque_bera(eps)
    jb_skew = stats.skew(eps, bias=False)
    jb_kurt = stats.kurtosis(eps, fisher=False, bias=False)

    print("Jarque–Bera test for normality of ε_t:")
    print(f"  JB statistic      : {jb_stat:10.4f}")
    print(f"  p-value           : {jb_pvalue:10.4g}")
    print(f"  Skewness          : {jb_skew:10.4f}")
    print(f"  Pearson kurtosis  : {jb_kurt:10.4f}")
    if jb_pvalue < 0.05:
        print("  -> Reject normality of ε_t at the 5% level.\n")
    else:
        print("  -> Do NOT reject normality of ε_t at the 5% level.\n")

    u = stats.norm.cdf(eps)
    A2 = anderson_darling_uniform(u)
    print("Anderson–Darling test for uniformity of u_t = Φ(ε_t):")
    print(f"  A^2 statistic     : {A2:10.4f}")
    print(f"  5% critical value : {2.492:10.4f}  (large-sample)")
    if A2 > 2.492:
        print("  -> Reject uniformity of u_t at the 5% level.")
    else:
        print("  -> Do NOT reject uniformity of u_t at the 5% level.")

    return spx, y_daily, rf_daily, r, h, eps, params, opt_res


# Only run automatically if the file is executed as a script
if __name__ == '__main__':
    run_q1()
