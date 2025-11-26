
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st


def compute_duan_ann_vol(h, r_index):

    h_series = pd.Series(h, index=r_index, name="h")
    duan_vol_ann = np.sqrt(252 * h_series) * 100.0
    duan_vol_ann.name = "Duan_AnnVol_x100"
    return duan_vol_ann


def load_vix(vix_path="VIX_History.csv"):
  
    vix = pd.read_csv(vix_path)
    vix["DATE"] = pd.to_datetime(vix["DATE"], format="%m/%d/%Y")
    vix.set_index("DATE", inplace=True)
    vix.sort_index(inplace=True)

    vix_close = vix["CLOSE"].astype(float)
    vix_close.name = "VIX"
    return vix_close


def align_vix_duan(duan_vol_ann, vix_close,
                   start="1990-01-01", end="2024-12-31"):
  
    vix_sub = vix_close.loc[start:end]
    duan_sub = duan_vol_ann.loc[start:end]

    df = pd.concat([vix_sub, duan_sub], axis=1).dropna()
    return df


def ols_vix_on_duan(df):

    y = df["VIX"].values
    x = df["Duan_AnnVol_x100"].values
    X = np.column_stack([np.ones_like(x), x])   # [const, NGARCH vol]

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty = X.T @ y
    beta_hat = XtX_inv @ Xty      # [alpha_hat, beta_hat]

    y_hat = X @ beta_hat
    resid = y - y_hat

    n = len(y)
    k = X.shape[1]

    RSS = np.sum(resid**2)
    TSS = np.sum((y - y.mean())**2)
    sigma2 = RSS / (n - k)
    var_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))

    t_alpha = beta_hat[0] / se_beta[0]
    t_beta = beta_hat[1] / se_beta[1]
    R2 = 1.0 - RSS / TSS

    # JB on residuals
    jb_stat, jb_pvalue = st.jarque_bera(resid)

    results = {
        "n": n,
        "alpha": beta_hat[0],
        "beta": beta_hat[1],
        "se_alpha": se_beta[0],
        "se_beta": se_beta[1],
        "t_alpha": t_alpha,
        "t_beta": t_beta,
        "R2": R2,
        "resid": resid,
        "jb_stat": jb_stat,
        "jb_pvalue": jb_pvalue,
    }
    return results


def mean_spread_test(df):

    spread = df["VIX"] - df["Duan_AnnVol_x100"]
    n = spread.shape[0]
    spread_mean = spread.mean()
    spread_std = spread.std(ddof=1)
    se_mean = spread_std / np.sqrt(n)

    t_stat = spread_mean / se_mean
    p_value = 2 * (1 - st.t.cdf(np.abs(t_stat), df=n-1))

    results = {
        "n": n,
        "spread": spread,
        "mean": spread_mean,
        "std": spread_std,
        "se_mean": se_mean,
        "t_stat": t_stat,
        "p_value": p_value,
    }
    return results


def plot_vix_vs_duan(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    df.plot(ax=ax, color=["steelblue", "red"])
    ax.set_title("VIX vs 100× annualized volatility under Duan NGARCH")
    ax.set_ylabel("Volatility (%)")
    ax.set_xlabel("Date")
    ax.legend(["VIX (Cboe)", "100× Duan annualized vol"])
    plt.tight_layout()
    plt.show()


def plot_rolling_corr_and_spread(df, window=252):

    # rolling correlation
    rolling_corr_252 = df["VIX"].rolling(window).corr(df["Duan_AnnVol_x100"])

    fig, ax = plt.subplots(figsize=(12, 4))
    rolling_corr_252.plot(ax=ax, color="steelblue")
    ax.set_title(f"Rolling {window}-Day Correlation: VIX vs NGARCH annualized volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.axhline(0.0, color="black", linewidth=1)
    plt.tight_layout()
    plt.show()

    # rolling mean spread
    spread = df["VIX"] - df["Duan_AnnVol_x100"]
    rolling_spread_252 = spread.rolling(window).mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    rolling_spread_252.plot(ax=ax, color="steelblue")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(f"Rolling {window}-Day Mean Spread: VIX – NGARCH annualized volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread (percentage points)")
    plt.tight_layout()
    plt.show()

    return rolling_corr_252, rolling_spread_252


def run_vix_duan_analysis(h, r,
                          vix_path="VIX_History.csv",
                          start="1990-01-01",
                          end="2024-12-31",
                          make_plots=True):

    # Duan annualized vol
    duan_vol_ann = compute_duan_ann_vol(h, r.index)
    vix_close = load_vix(vix_path)

    df = align_vix_duan(duan_vol_ann, vix_close, start=start, end=end)
    if make_plots:
        plot_vix_vs_duan(df)


    print("Summary statistics (Jan 1990 – Dec 2024):")
    print(df.describe())

    print("\nCorrelation matrix:")
    print(df.corr())

    rolling_corr_252, rolling_spread_252 = (None, None)
    if make_plots:
        rolling_corr_252, rolling_spread_252 = plot_rolling_corr_and_spread(df)

    # OLS regression
    ols_res = ols_vix_on_duan(df)

    print("\nOLS regression: VIX_t = a + b * NGARCH_vol_t + error_t")
    print("--------------------------------------------------------")
    print(f"Number of observations: {ols_res['n']}")
    print(f"R-squared            : {ols_res['R2']:.4f}\n")

    print("Coefficient estimates:")
    print(f"  alpha (intercept) : {ols_res['alpha']: .4f}  "
          f"(SE = {ols_res['se_alpha']: .4f},  t = {ols_res['t_alpha']: .2f})")
    print(f"  beta  (slope)     : {ols_res['beta']: .4f}  "
          f"(SE = {ols_res['se_beta']: .4f},  t = {ols_res['t_beta']: .2f})")

    print("\nJarque–Bera test on OLS residuals:")
    print(f"  JB statistic : {ols_res['jb_stat']:.4f}")
    print(f"  p-value      : {ols_res['jb_pvalue']:.4g}")
    if ols_res["jb_pvalue"] < 0.05:
        print("  -> Reject normality of regression errors at 5% level.")
    else:
        print("  -> Do NOT reject normality of regression errors at 5% level.")

    # Mean spread test
    spread_res = mean_spread_test(df)

    print("\nMean spread test: H0: E[VIX − NGARCH_vol] = 0")
    print(f"  Sample mean spread      : {spread_res['mean']:.4f} %-pts "
          f"({spread_res['mean']*100:.2f} bps)")
    print(f"  Standard error of mean  : {spread_res['se_mean']:.6f}")
    print(f"  t-statistic             : {spread_res['t_stat']:.2f}")
    print(f"  p-value                 : {spread_res['p_value']:.4g}")

    if spread_res["p_value"] < 0.05:
        direction = "positive" if spread_res["mean"] > 0 else "negative"
        print(f"  -> Reject H0 at 5% level: {direction} mean spread,")
        print("     evidence of a statistically significant volatility risk premium.")
    else:
        print("  -> Do NOT reject H0: mean spread not significantly different from zero.")

    results = {
        "df": df,
        "rolling_corr_252": rolling_corr_252,
        "rolling_spread_252": rolling_spread_252,
        "ols": ols_res,
        "spread_test": spread_res,
    }
    return results


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported.\n"
        "Use from your notebook as:\n"
        "  from vix_duan_analysis import run_vix_duan_analysis\n"
        "and then call run_vix_duan_analysis(h, r, ...)."
    )
