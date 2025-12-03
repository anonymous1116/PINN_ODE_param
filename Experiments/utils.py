import numpy as np

def h1_error_trajectory(t, u_pred, u_true):
    """
    t      : [N] array (time grid, assumed uniform)
    u_pred: [N, d] array (predicted trajectory, e.g. (V,R))
    u_true: [N, d] array (true trajectory)

    returns: scalar H^1 error over [t[0], t[-1]]
    """
    t = np.asarray(t)
    u_pred = np.asarray(u_pred)
    u_true = np.asarray(u_true)

    # ensure 2D: [N, d]
    if u_pred.ndim == 1:
        u_pred = u_pred[:, None]
        u_true = u_true[:, None]

    dt = t[1] - t[0]  # assumes uniform grid

    # ----- value (L2) part -----
    val_diff = u_pred - u_true              # [N, d]
    val_term = np.sum(val_diff**2) * dt     # scalar

    # ----- derivative (H1 seminorm) part -----
    # approximate du/dt by finite differences
    du_pred = np.gradient(u_pred, t, axis=0)   # [N, d]
    du_true = np.gradient(u_true, t, axis=0)   # [N, d]

    der_diff = du_pred - du_true
    der_term = np.sum(der_diff**2) * dt

    h1_sq = val_term + der_term
    return np.sqrt(h1_sq)
