# q4.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def compute_Q_weights(eps_array, eta):
    """
    Exponential-tilt weights q_i(η) ∝ exp(-η ε_i).

    Under P the weights are 1/N; under Q they are q_i(η).
    """
    eps_array = np.asarray(eps_array)
    w = np.exp(-eta * eps_array)
    q = w / w.sum()
    return q


def sample_Q_shocks(eps_array, lam, n_samples=50_000, seed=123):
    """
    Sample Q-shocks via filtered historical simulation
    with exponential tilt parameter λ.
    """
    eps_array = np.asarray(eps_array)

    rng = np.random.default_rng(seed)
    N = len(eps_array)
    q = compute_Q_weights(eps_array, lam)
    idx = rng.choice(N, size=n_samples, replace=True, p=q)
    eps_Q = eps_array[idx] + lam
    return eps_Q, q


def run_exponential_tilt_analysis(
    eps,
    lam,
    n_samples=50_000,
    seed=123,
    make_plots=True,
):
    """
    Question 4 – Exponential tilt and filtered historical simulation.

    Parameters
    ----------
    eps : array-like
        Historical NGARCH shocks ε_t.
    lam : float
        Market price of risk λ.
    n_samples : int
        Number of Q-shocks to simulate.
    seed : int
        RNG seed.
    make_plots : bool
        If True, produce density / difference plots.

    Returns
    -------
    results : dict
        Contains eps_array, eps_Q, q_weights, grid, f_P, f_Q,
        empirical likelihood ratio, theoretical likelihood ratio, etc.
    """
    eps_array = np.asarray(eps)
    eps_Q, q_weights = sample_Q_shocks(eps_array, lam, n_samples=n_samples, seed=seed)

    print("Empirical mean under P  (ε):   ", eps_array.mean())
    print("Empirical mean under Q (ε^Q): ", eps_Q.mean())

    # KDEs for P and Q-shocks
    xmin = min(eps_array.min(), eps_Q.min()) - 0.5
    xmax = max(eps_array.max(), eps_Q.max()) + 0.5
    grid = np.linspace(xmin, xmax, 400)

    kde_P = gaussian_kde(eps_array)
    kde_Q = gaussian_kde(eps_Q)

    f_P = kde_P(grid)
    f_Q = kde_Q(grid)

    eps_denom = 1e-10
    likelihood_ratio_emp = f_Q / np.maximum(f_P, eps_denom)

    # Theoretical likelihood ratio dQ/dP
    normalization_const = np.mean(np.exp(-lam * eps_array))
    theoretical_lr = np.exp(-lam * grid) / normalization_const

    if make_plots:
        # --- Densities + theoretical likelihood ratio ---
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(
            grid,
            f_P,
            color="red",
            label=r"Physical density $f_P(\varepsilon)$",
            lw=2,
        )
        ax1.plot(
            grid,
            f_Q,
            color="steelblue",
            label=r"Tilted density $f_Q(\varepsilon)$",
            lw=2,
            linestyle="--",
        )
        ax1.set_xlabel(r"Shock $\varepsilon$")
        ax1.set_ylabel("Density")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(
            grid,
            theoretical_lr,
            color="gray",
            alpha=0.7,
            label=r"Likelihood ratio $dQ/dP$",
        )
        ax2.set_ylabel("Likelihood ratio")
        ax2.legend(loc="upper right")

        plt.title("Filtered historical simulation with exponential tilt")
        plt.tight_layout()
        plt.show()

        # --- Difference in densities f_Q - f_P ---
        fig_diff, ax_diff = plt.subplots(figsize=(8, 5))

        f_diff = f_Q - f_P
        ax_diff.plot(grid, f_diff, color="steelblue", lw=2)
        ax_diff.axhline(0, color="black", linestyle="--")

        ax_diff.set_title(
            r"Difference in Density: $f_Q(\varepsilon) - f_P(\varepsilon)$"
        )
        ax_diff.set_xlabel(r"Shock $\varepsilon$")
        ax_diff.set_ylabel("Density Difference")
        ax_diff.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.show()

    results = {
        "eps_array": eps_array,
        "lam": lam,
        "eps_Q": eps_Q,
        "q_weights": q_weights,
        "grid": grid,
        "kde_P": kde_P,
        "kde_Q": kde_Q,
        "f_P": f_P,
        "f_Q": f_Q,
        "likelihood_ratio_emp": likelihood_ratio_emp,
        "theoretical_lr": theoretical_lr,
    }

    return results