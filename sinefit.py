"""Shared helpers: fit y = A*sin(B*(x - C)) + D to data, and plot the result."""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def _model(x, A, B, C, D):
    return A * np.sin(B * (x - C)) + D


def fit_sinusoid(x, y, period_range=None, fixed_period=None, n_grid=4000):
    """
    Best-fit sinusoid  y = A*sin(B*(x - C)) + D.

    Give EITHER fixed_period (e.g. 365) OR period_range=(low, high) to search.
    Method: for every candidate period, solve the linear least-squares problem
    exactly, keep the best period, then polish with scipy's curve_fit.
    This avoids the bad-initial-guess failures common with curve_fit alone.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def linear_fit(P):
        w = 2 * np.pi / P
        M = np.column_stack([np.sin(w * x), np.cos(w * x), np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        r = y - M @ coef
        return P, coef, float(r @ r)

    if fixed_period is not None:
        periods = [float(fixed_period)]
    elif period_range is not None:
        periods = np.linspace(period_range[0], period_range[1], n_grid)
    else:
        raise ValueError("Provide fixed_period or period_range")

    P, (a, b, d), _ = min((linear_fit(p) for p in periods), key=lambda t: t[2])
    w = 2 * np.pi / P
    A = float(np.hypot(a, b))                 # a sin + b cos = A sin(wx + phi)
    phi = float(np.arctan2(b, a))
    B, C, D = w, (-phi / w) % P, float(d)

    if fixed_period is None:                  # polish period too
        try:
            (A2, B2, C2, D2), _ = curve_fit(_model, x, y, p0=[A, B, C, D], maxfev=20000)
            if A2 < 0:                        # normalise to positive amplitude
                A2, C2 = -A2, C2 + np.pi / B2
            B2 = abs(B2)
            if np.isfinite([A2, B2, C2, D2]).all():
                A, B, C, D = float(A2), float(B2), float(C2), float(D2)
        except (RuntimeError, ValueError):
            pass                              # keep the linear-fit answer

    P = 2 * np.pi / B
    C = C % P
    resid = y - _model(x, A, B, C, D)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "amplitude": A, "B": B, "period": P, "phase_shift": C,
        "center": D, "r2": r2,
        "cos_phase_shift": (C + P / 4) % P,   # same curve written as a cosine
    }


def equation_strings(p, yname="y", xname="x"):
    sign = "-"
    sine = (f"{yname} = {p['amplitude']:.4f} sin({p['B']:.5f}({xname} {sign} {p['phase_shift']:.3f})) "
            f"+ {p['center']:.4f}")
    cosine = (f"{yname} = {p['amplitude']:.4f} cos({p['B']:.5f}({xname} {sign} {p['cos_phase_shift']:.3f})) "
              f"+ {p['center']:.4f}")
    return sine, cosine


def print_summary(p, xname="x", yname="y", units=""):
    sine, cosine = equation_strings(p, yname, xname)
    print("Function type : sinusoidal")
    print(f"Amplitude     : {p['amplitude']:.4f} {units}")
    print(f"Period        : {p['period']:.4f} days   (B = 2*pi/period = {p['B']:.5f})")
    print(f"Phase shift   : {p['phase_shift']:.3f} days (sine form)")
    print(f"Vert. shift   : {p['center']:.4f} {units}")
    print(f"R^2 of fit    : {p['r2']:.4f}")
    print("Sine form     :", sine)
    print("Cosine form   :", cosine)


def plot_fit(x, y, p, title, xlabel, ylabel, xmin, xmax, save_path=None, point_size=25):
    xs = np.linspace(xmin, xmax, 2000)
    sine, _ = equation_strings(p)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(x, y, s=point_size, color="tab:blue", zorder=3, label="Data")
    ax.plot(xs, _model(xs, p["amplitude"], p["B"], p["phase_shift"], p["center"]),
            color="tab:red", lw=2, label=f"Fit: {sine}")
    ax.axhline(p["center"], color="gray", ls="--", lw=1, label="Center of oscillation")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xmin, xmax)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved graph -> {save_path}")
    plt.show()