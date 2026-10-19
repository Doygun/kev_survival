from __future__ import annotations
import json
import math
import pickle
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kev_survival.eval import metrics as M
from kev_survival.io.access import read_parquet
from kev_survival.io.common import ROOT, load_project, sha256_file, utc_now
warnings.filterwarnings('ignore')
CFG = load_project()
LCFG = yaml.safe_load(open(ROOT / 'configs' / 'models.yaml', encoding='utf-8'))['logistic']
assert CFG['locked'] is True
SNAP = str(CFG['t_freeze']).replace('-', '')
PROC, LAB = (ROOT / 'data' / 'processed', ROOT / 'data' / 'processed' / 'labels')
MODELS, METRICS, TABLES = (ROOT / 'results' / 'models', ROOT / 'results' / 'metrics', ROOT / 'results' / 'tables')
BUDGETS = CFG['budget_fractions']
H = int(LCFG['target_horizon_days'])
SEED = int(CFG['seeds'][0])
REPS = int(CFG['bootstrap_reps'])
GRID = np.arange(1, 366, dtype=float)

def xy(ctx: str, phase: str, eligible_only: bool):
    X = read_parquet(PROC / f'X_disclosure_{ctx}_B_{SNAP}.parquet', 'S10', phase)
    y = read_parquet(LAB / f'labels_{ctx}_B_{SNAP}.parquet', 'S10', phase)
    assert (X['cve_id'].values == y['cve_id'].values).all()
    y30 = ((y['event'] == 1) & (y['duration_days'] <= H)).astype(int)
    elig = (y['duration_days'] >= H) | (y30 == 1)
    if eligible_only:
        return (X.loc[elig].drop(columns='cve_id'), y30[elig].values, y.loc[elig], int((~elig).sum()))
    return (X.drop(columns='cve_id'), y30.values, y, int((~elig).sum()))

def logistic():
    Xtr, ytr, _, ex_tr = xy('validmodel', 'selection', True)
    Xva, yva, lva, ex_va = xy('valid', 'selection', True)
    Cs = np.logspace(math.log10(LCFG['C_grid']['low']), math.log10(LCFG['C_grid']['high']), LCFG['C_grid']['n'])
    rows = []
    for l1 in LCFG['l1_ratio']:
        for C in Cs:
            m = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=l1, C=C, max_iter=LCFG['max_iter'], random_state=SEED).fit(Xtr.values, ytr)
            p = m.predict_proba(Xva.values)[:, 1]
            rows.append({'l1_ratio': l1, 'C': float(C), 'n_nonzero': int((m.coef_ != 0).sum()), 'recall_1pct_valid': M.topk_recall(p, yva, 0.01, lva['cve_id'].values), 'log_loss_valid': float(log_loss(yva, p, labels=[0, 1]))})
    g = pd.DataFrame(rows)
    best = g['recall_1pct_valid'].max()
    sel = g[g['recall_1pct_valid'] == best].sort_values(['log_loss_valid', 'n_nonzero']).iloc[0]
    g['selected'] = (g['l1_ratio'] == sel['l1_ratio']) & (g['C'] == sel['C'])
    g.to_csv(TABLES / f's10_logit_selection_grid_{SNAP}.csv', index=False)
    Xtm, ytm, _, ex_tm = xy('testmodel', 'refit', True)
    m = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=sel['l1_ratio'], C=sel['C'], max_iter=LCFG['max_iter'], random_state=SEED).fit(Xtm.values, ytm)
    with open(MODELS / f'logit_disclosure_B_{SNAP}.pkl', 'wb') as f:
        pickle.dump({'model': m, 'columns': list(Xtm.columns), 'selection': sel.to_dict()}, f)
    info = {'l1_ratio': float(sel['l1_ratio']), 'C': float(sel['C']), 'recall_1pct_valid': float(best), 'log_loss_valid': float(sel['log_loss_valid']), 'n_nonzero_validmodel': int(sel['n_nonzero']), 'n_nonzero_testmodel': int((m.coef_ != 0).sum()), 'excluded_insufficient_followup': {'validmodel': ex_tr, 'valid': ex_va, 'testmodel': ex_tm}, 'train_events_30d': {'validmodel': int(ytr.sum()), 'valid': int(yva.sum()), 'testmodel': int(ytm.sum())}, 'n_ties_at_best_recall': int((g['recall_1pct_valid'] == best).sum())}
    pd.DataFrame({'column': Xtm.columns, 'coef': m.coef_[0]}).to_csv(TABLES / f's10_logit_coefficients_{SNAP}.csv', index=False)
    return (m, info)

def cox_baseline():
    b = pickle.load(open(MODELS / f'cox_disclosure_B_{SNAP}.pkl', 'rb'))
    bl = b['model']._get_baseline_model(b['alpha']).cum_baseline_hazard_
    return (np.asarray(bl.x, float), np.asarray(bl.y, float))

def S_at(lp, hx, hy, t):
    t = np.asarray(t, float)
    idx = np.searchsorted(hx, t, side='right') - 1
    Ht = np.where(idx >= 0, hy[np.clip(idx, 0, None)], 0.0)
    return np.exp(-Ht * np.exp(lp))

def metrics_block(score, y, ids, tag):
    out = {'model': tag, 'n': len(y), 'events': int(y.sum()), 'brier': float(np.mean((y - score) ** 2)), **{f'recall_{b}': M.topk_recall(score, y, b, ids) for b in BUDGETS}, 'cal': M.calibration_intercept_slope(score, y)}
    t, nb = M.calibration_table(score, y, CFG['calibration_bins_default'], CFG['calibration_bins_fallback'])
    return (out, t.assign(model=tag, n_bins=nb))

def boot(n, stat, rng):
    vals, fails = ([], 0)
    for _ in range(REPS):
        idx = rng.integers(0, n, n)
        v = stat(idx)
        if v is None or not np.all(np.isfinite(v)):
            fails += 1
            vals.append(np.full(np.shape(v) if v is not None else (1,), np.nan))
        else:
            vals.append(np.asarray(v, float))
    return (np.vstack(vals), fails)

def ci(arr):
    a = arr[np.isfinite(arr).all(axis=1)]
    return np.percentile(a, [2.5, 97.5], axis=0)

def main() -> None:
    t0 = utc_now()
    logit, linfo = logistic()
    Xte, y30, lte, _ = xy('test', 'test_eval', False)
    ids = lte['cve_id'].values
    p_log = logit.predict_proba(Xte.values)[:, 1]
    cox = read_parquet(PROC / f'predictions_test_cox_{SNAP}.parquet', 'S10', 'test_eval')
    cb = cox[cox['model'] == 'cox_disclosure_B'].set_index('cve_id').loc[ids]
    cc = cox[cox['model'] == 'cox_current_B'].set_index('cve_id').loc[ids]
    r_cox = cb['risk_30d'].values
    res, cals = ([], [])
    for sc, tag in ((r_cox, 'cox_disclosure_B'), (p_log, 'logit_disclosure_B')):
        o, t = metrics_block(sc, y30, ids, tag)
        res.append({**o, 'cohort': 'test_all'})
        cals.append(t.assign(cohort='test_all'))
    ep = pd.read_parquet(PROC.parent / 'interim' / f'epss_first_scores_{SNAP}.parquet')
    hx, hy = cox_baseline()
    t_tr = read_parquet(LAB / f'labels_testmodel_B_{SNAP}.parquet', 'S10', 'refit')
    km_t = np.sort(t_tr['duration_days'].values)
    epss_rows, epss_sets = ([], {})
    for tol in (1, 2, 7):
        d = lte[['cve_id', 't_pub', 't_kev', 'event']].merge(ep[['cve_id', 'first_epss_available_at', 'first_epss_score', 'epss_delay_days', 'epss_model_version']], on='cve_id', how='left')
        d['lp'] = cb['lp'].values
        d['p_log'] = p_log
        within = d['epss_delay_days'].le(tol).fillna(False)
        fe, tk, tp = (pd.to_datetime(d[c]) for c in ('first_epss_available_at', 't_kev', 't_pub'))
        te = pd.concat([fe, tp], axis=1).max(axis=1)
        ev = d['t_kev'].notna()
        epss_before = ~ev | (fe < tk)
        same_start = ev & (tk == te)
        keep = within & epss_before & ~same_start
        x = d[keep].copy()
        de = (te[keep] - tp[keep]).dt.days.values.astype(float)
        x['y'] = (x['t_kev'].notna() & (tk[keep] > te[keep]) & (tk[keep] <= te[keep] + pd.Timedelta(days=30))).astype(int).values
        x['cox_cond'] = 1 - S_at(x['lp'].values, hx, hy, de + 30) / S_at(x['lp'].values, hx, hy, de)
        epss_sets[tol] = x
        row = {'tolerance_days': tol, 'n': int(keep.sum()), 'events_30d_from_t_e': int(x['y'].sum()), 'kev_events_in_subcohort': int(x['t_kev'].notna().sum()), 'excluded_not_within_tol': int((~within).sum()), 'excluded_epss_not_before_kev': int((within & ~epss_before).sum()), 'excluded_t_kev_eq_t_e': int((within & epss_before & same_start).sum()), 'epss_versions': x['epss_model_version'].value_counts().to_dict()}
        for sc, tag in ((x['first_epss_score'].values, 'epss'), (x['cox_cond'].values, 'cox_disclosure_B_conditional'), (x['p_log'].values, 'logit_disclosure_B')):
            o, t = metrics_block(sc, x['y'].values, x['cve_id'].values, tag)
            res.append({**o, 'cohort': f'epss_fair_{tol}d'})
            cals.append(t.assign(cohort=f'epss_fair_{tol}d'))
        epss_rows.append(row)
    sub = epss_sets[1][['cve_id', 'first_epss_available_at', 'first_epss_score', 'epss_delay_days', 'y', 'cox_cond', 'p_log']]
    sub.to_parquet(PROC / f'epss_common_cohort_{SNAP}.parquet', index=False)
    from sksurv.nonparametric import kaplan_meier_estimator
    kt, ks = kaplan_meier_estimator(t_tr['event'].astype(bool).values, t_tr['duration_days'].astype(float).values)
    S_null = np.tile(np.interp(GRID, kt, ks, left=1.0)[None, :], (len(lte), 1))
    y_ipcw = M.surv_y(t_tr['event'], t_tr['duration_days'])
    ibs_null = float(M.ibs_per_cve(M.brier_contrib(y_ipcw, lte['event'].values, lte['duration_days'].values, S_null, GRID), GRID).mean())
    rng = np.random.default_rng(SEED)
    n = len(ids)
    ev_all, T_all = (lte['event'].values, lte['duration_days'].values)

    def rec(score, y, idx):
        s, yy = (score[idx], y[idx])
        if yy.sum() == 0:
            return np.nan
        k = max(1, math.ceil(0.01 * len(idx)))
        order = np.lexsort((np.arange(len(idx)), -s))
        return yy[order[:k]].sum() / yy.sum()

    def s1(idx):
        return [rec(r_cox, y30, idx) - rec(p_log, y30, idx), np.mean((y30[idx] - r_cox[idx]) ** 2) - np.mean((y30[idx] - p_log[idx]) ** 2)]
    b1, f1 = boot(n, s1, rng)
    x1 = epss_sets[1]
    yc, sc_cox, sc_ep = (x1['y'].values, x1['cox_cond'].values, x1['first_epss_score'].values)

    def s2(idx):
        return [rec(sc_cox, yc, idx) - rec(sc_ep, yc, idx), np.mean((yc[idx] - sc_cox[idx]) ** 2) - np.mean((yc[idx] - sc_ep[idx]) ** 2)]
    b2, f2 = boot(len(yc), s2, rng)
    lp_d, lp_c = (cb['lp'].values, cc['lp'].values)
    ibs_d, ibs_c = (cb['ibs_contrib'].values, cc['ibs_contrib'].values)
    rc_c = cc['risk_30d'].values

    def s3(idx):
        e, T = (ev_all[idx], T_all[idx])
        if e[T < 365].sum() == 0:
            return [np.nan] * 3
        try:
            cd = M.uno_c(y_ipcw, e, T, lp_d[idx], 365.0)
            ccur = M.uno_c(y_ipcw, e, T, lp_c[idx], 365.0)
        except Exception:
            return [np.nan] * 3
        return [ccur - cd, ibs_c[idx].mean() - ibs_d[idx].mean(), rec(rc_c, y30, idx) - rec(r_cox, y30, idx)]
    b3, f3 = boot(n, s3, rng)
    full = np.arange(n)
    diffs = {'cox_minus_logit': {'cohort': 'test_all', 'n': n, 'events_30d': int(y30.sum()), 'failed_samples': f1, 'metrics': ['recall_1pct_30d', 'brier_30d'], 'estimate': s1(full), 'ci95': ci(b1).tolist()}, 'cox_minus_epss': {'cohort': 'epss_fair_1d', 'n': len(yc), 'events_30d': int(yc.sum()), 'failed_samples': f2, 'metrics': ['recall_1pct_30d_from_t_e', 'brier_30d_from_t_e'], 'estimate': s2(np.arange(len(yc))), 'ci95': ci(b2).tolist()}, 'leaky_minus_disclosure_cox': {'cohort': 'test_all', 'n': n, 'failed_samples': f3, 'metrics': ['uno_c_tau365', 'ibs_1_365', 'recall_1pct_30d'], 'estimate': s3(full), 'ci95': ci(b3).tolist()}, 'reps': REPS, 'seed': SEED, 'paired': True, 'method': 'CVE düzeyinde eşlenik yeniden örnekleme, yüzdelik %95 GA'}
    for k in ('cox_minus_logit', 'cox_minus_epss', 'leaky_minus_disclosure_cox'):
        diffs[k]['estimate'] = [float(v) for v in diffs[k]['estimate']]
    np.save(METRICS / f'bootstrap_samples_{SNAP}.npy', np.hstack([b1, np.vstack([b2]), b3]))
    (METRICS / f'bootstrap_diffs_{SNAP}.json').write_text(json.dumps(diffs, indent=1), encoding='utf-8')
    flat = pd.json_normalize(res)
    flat.to_csv(TABLES / f's10_test_metrics_{SNAP}.csv', index=False)
    pd.concat(cals).to_csv(TABLES / f's10_calibration_30d_{SNAP}.csv', index=False)
    pd.DataFrame(epss_rows).to_csv(TABLES / f's10_epss_subcohorts_{SNAP}.csv', index=False)
    out = {'started': t0, 'finished': utc_now(), 'logistic': linfo, 'epss_subcohorts': epss_rows, 'ibs_reference_km_null': ibs_null, 'test_metrics': res, 'bootstrap': diffs, 'inputs_sha256': {'predictions_test_cox': sha256_file(PROC / f'predictions_test_cox_{SNAP}.parquet'), 'epss_first_scores': sha256_file(PROC.parent / 'interim' / f'epss_first_scores_{SNAP}.parquet')}}
    (METRICS / f's10_summary_{SNAP}.json').write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
    print(json.dumps(linfo, indent=1))
    print(pd.DataFrame(epss_rows).to_string())
    print(flat.drop(columns=[c for c in flat.columns if c.startswith('cal.')]).to_string())
    print(flat[['model', 'cohort'] + [c for c in flat.columns if c.startswith('cal.')]].to_string())
    print('ibs_null', ibs_null)
    print(json.dumps(diffs, indent=1))
if __name__ == '__main__':
    main()
