from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

def surv_y(event, time) -> np.ndarray:
    return Surv.from_arrays(np.asarray(event).astype(bool), np.asarray(time).astype(float))

def uno_c(y_train, event, time, risk, tau: float) -> float:
    return float(concordance_index_ipcw(y_train, surv_y(event, time), np.asarray(risk), tau=tau)[0])

def c_truncated_no_ipcw(event, time, risk, tau: float) -> float:
    from sksurv.metrics import concordance_index_censored
    ev = np.asarray(event).astype(bool) & (np.asarray(time) < tau)
    return float(concordance_index_censored(ev, np.minimum(np.asarray(time, float), tau), np.asarray(risk))[0])

def survival_matrix(lp: np.ndarray, h0_times: np.ndarray, h0_values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(h0_times, grid, side='right') - 1
    H = np.where(idx >= 0, h0_values[np.clip(idx, 0, None)], 0.0)
    return np.exp(-np.outer(np.exp(lp), H))

def brier_contrib(y_train, event, time, S: np.ndarray, grid: np.ndarray) -> np.ndarray:
    event = np.asarray(event).astype(bool)
    time = np.asarray(time).astype(float)
    g_t = np.ones(len(grid))
    g_T = np.ones_like(time)
    if y_train is not None:
        cens = CensoringDistributionEstimator().fit(y_train)
        g_t = cens.predict_proba(grid)
        use = event & (time <= grid[-1])
        if use.any():
            g_T[use] = cens.predict_proba(time[use])
    died = (time[:, None] <= grid[None, :]) & event[:, None]
    alive = time[:, None] > grid[None, :]
    return np.where(died, S ** 2 / g_T[:, None], 0.0) + np.where(alive, (1 - S) ** 2 / g_t[None, :], 0.0)

def ibs_per_cve(contrib: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.trapezoid(contrib, grid, axis=1) / (grid[-1] - grid[0])

def topk_recall(score: np.ndarray, y: np.ndarray, frac: float, ids: np.ndarray | None=None) -> float:
    n = len(score)
    k = max(1, math.ceil(frac * n))
    order = np.lexsort((ids if ids is not None else np.arange(n), -np.asarray(score)))
    y = np.asarray(y)
    tot = y.sum()
    return float(y[order[:k]].sum() / tot) if tot > 0 else float('nan')

def calibration_table(pred: np.ndarray, y: np.ndarray, bins: int=10, fallback: int=5) -> tuple[pd.DataFrame, int]:
    for b in (bins, fallback):
        g = pd.qcut(pd.Series(pred).rank(method='first'), b, labels=False)
        t = pd.DataFrame({'g': g, 'p': pred, 'y': y}).groupby('g').agg(n=('y', 'size'), events=('y', 'sum'), mean_pred=('p', 'mean'), obs_rate=('y', 'mean'))
        if (t['events'] >= 5).all() or b == fallback:
            return (t.reset_index(), b)
    raise AssertionError

def calibration_intercept_slope(pred: np.ndarray, y: np.ndarray) -> dict:
    import statsmodels.api as sm
    p = np.clip(pred, 1e-09, 1 - 1e-09)
    lg = np.log(p / (1 - p))
    out = {'observed': float(np.mean(y)), 'expected': float(np.mean(pred)), 'o_e_ratio': float(np.mean(y) / np.mean(pred))}
    try:
        slope = sm.GLM(y, sm.add_constant(lg), family=sm.families.Binomial()).fit()
        icpt = sm.GLM(y, np.ones_like(lg), family=sm.families.Binomial(), offset=lg).fit()
        out.update({'slope': float(slope.params[1]), 'intercept_citl': float(icpt.params[0])})
    except Exception as e:
        out.update({'slope': None, 'intercept_citl': None, 'error': str(e)})
    return out
