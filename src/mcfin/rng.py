"""Random number generation from scratch: an LCG and the Box-Muller transform.

Rewritten from my first lab: the Park-Miller generator (a = 16807, c = 0,
m = 2^31 - 1) and Box-Muller pairs that turn uniforms into N(0, 1) draws.
"""
import numpy as np

PARK_MILLER = dict(a=16807, c=0, m=2**31 - 1)


def lcg(n, seed=1, a=PARK_MILLER["a"], c=PARK_MILLER["c"], m=PARK_MILLER["m"]):
    """Return n uniforms u_k = x_k / m with x_{k+1} = (a x_k + c) mod m."""
    out = np.empty(n)
    x = seed
    for k in range(n):
        x = (a * x + c) % m
        out[k] = x / m
    return out


def box_muller(u1, u2):
    """Map two uniform arrays to two independent standard normal arrays."""
    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2
    return radius * np.cos(angle), radius * np.sin(angle)


def lcg_normals(n, seed=1):
    """n standard normals from the LCG via Box-Muller."""
    u = lcg(2 * ((n + 1) // 2), seed=seed)
    z1, z2 = box_muller(u[0::2], u[1::2])
    return np.column_stack([z1, z2]).ravel()[:n]
