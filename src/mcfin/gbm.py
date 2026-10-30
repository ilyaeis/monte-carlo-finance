"""Brownian motion and geometric Brownian motion paths (exact discretisation)."""
import numpy as np


def brownian_paths(T, n_steps, n_paths, rng):
    """Standard Brownian motion W on a uniform grid; shape (n_paths, n_steps + 1)."""
    dt = T / n_steps
    dw = np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    w = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dw, axis=1)], axis=1)
    return np.linspace(0.0, T, n_steps + 1), w


def gbm_paths(S0, mu, sigma, T, n_steps, n_paths, rng, z=None):
    """GBM paths S_t = S0 exp((mu - sigma^2/2) t + sigma W_t).

    Uses the exact log-normal step, so there is no discretisation bias at
    the grid points (unlike an Euler step). Pass z to reuse normals, for
    example -z for antithetic paths.
    """
    dt = T / n_steps
    if z is None:
        z = rng.standard_normal((n_paths, n_steps))
    log_steps = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_s = np.log(S0) + np.cumsum(log_steps, axis=1)
    s = np.concatenate([np.full((z.shape[0], 1), S0), np.exp(log_s)], axis=1)
    return np.linspace(0.0, T, n_steps + 1), s
