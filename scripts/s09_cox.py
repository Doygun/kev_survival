from __future__ import annotations
import datetime as dt
import json
import pickle
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sksurv.linear_model import CoxnetSurvivalAnalysis
from kev_survival.eval import metrics as M
from kev_survival.io.access import read_parquet
from kev_survival.io.common import ROOT, load_project, sha256_file, utc_now
warnings.filterwarnings('ignore', category=FutureWarning)
CFG = load_project()
MCFG = yaml.safe_load(open(ROOT / 'configs' / 'models.yaml', encoding='utf-8'))['cox']
assert CFG['locked'] is True
SNAP = str(CFG['t_freeze']).replace('-', '')
PROC, LAB = (ROOT / 'data' / 'processed', ROOT / 'data' / 'processed' / 'labels')
MODELS, METRICS, TABLES = (ROOT / 'results' / 'models', ROOT / 'results' / 'metrics', ROOT / 'results' / 'tables')
GRID = np.arange(1, 366, dtype=float)
TAU = float(CFG['tau_days'])
BUDGETS = CFG['budget_fractions']

def load(fs: str, ctx: str, s: str, phase: str):
    X = read_parquet(PROC / f'X_{fs}_{ctx}_{s}_{SNAP}.parquet', 'S9', phase)
    y = read_parquet(LAB / f'labels_{ctx}_{s}_{SNAP}.parquet', 'S9', phase)
    assert (X['cve_id'].values == y['cve_id'].values).all()
    return (X.drop(columns='cve_id'), y)

def fit(X, y, l1: float, alphas=None) -> CoxnetSurvivalAnalysis:
    kw = dict(l1_ratio=l1, fit_baseline_model=True, max_iter=MCFG['max_iter'])
    kw.update(alphas=alphas) if alphas is not None else kw.update(n_alphas=MCFG['n_alphas'], alpha_min_ratio=MCFG['alpha_min_ratio'])
    return CoxnetSurvivalAnalysis(**kw).fit(X.values, M.surv_y(y['event'], y['duration_days']))

def select(fs: str) -> tuple[dict, pd.DataFrame]:
    Xtr, ytr = load(fs, 'validmodel', 'B', 'selection')
    Xva, yva = load(fs, 'valid', 'B', 'selection')
    y_train = M.surv_y(ytr['event'], ytr['duration_days'])
    tau_v = float(min(TAU, yva['duration_days'].max()))
    rows = []
    for l1 in MCFG['l1_ratio']:
        m = fit(Xtr, ytr, l1)
        for j, a in enumerate(m.alphas_):
            nz = int((m.coef_[:, j] != 0).sum())
            c = M.uno_c(y_train, yva['event'], yva['duration_days'], m.predict(Xva.values, alpha=a), tau_v) if nz else 0.5
            rows.append({'l1_ratio': l1, 'alpha_index': j, 'alpha': float(a), 'n_nonzero': nz, 'uno_c_valid': c})
    g = pd.DataFrame(rows)
    best = g['uno_c_valid'].max()
    near = g[g['uno_c_valid'] >= best - MCFG['near_tie_tolerance_c']]
    sel = near.sort_values(['n_nonzero', 'alpha'], ascending=[True, False]).iloc[0]
    g['selected'] = (g['l1_ratio'] == sel['l1_ratio']) & (g['alpha_index'] == sel['alpha_index'])
    paths = {l1: fit(Xtr, ytr, l1).alphas_.tolist() for l1 in [sel['l1_ratio']]}
    return ({'feature_set': fs, 'l1_ratio': float(sel['l1_ratio']), 'alpha': float(sel['alpha']), 'alpha_index': int(sel['alpha_index']), 'n_nonzero_validmodel': int(sel['n_nonzero']), 'uno_c_valid': float(sel['uno_c_valid']), 'best_uno_c_valid': float(best), 'tau_valid': tau_v, 'n_near': int(len(near)), 'alpha_path': paths[sel['l1_ratio']], 'validmodel_events': int(ytr['event'].sum()), 'valid_events': int(yva['event'].sum())}, g)

def baseline(m: CoxnetSurvivalAnalysis, alpha: float):
    b = m._get_baseline_model(alpha)
    h = b.cum_baseline_hazard_
    return (np.asarray(h.x, dtype=float), np.asarray(h.y, dtype=float))

def evaluate(m, alpha: float, fs: str, s: str, y_train_ipcw, tag: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    X, y = load(fs, 'test', s, 'test_eval')
    lp = m.predict(X.values, alpha=alpha)
    hx, hy = baseline(m, alpha)
    S = M.survival_matrix(lp, hx, hy, GRID)
    sf = m.predict_survival_function(X.values[:3], alpha=alpha)
    assert np.allclose([f(30.0) for f in sf], S[:3, 29], atol=1e-10)
    ev, T = (y['event'].values, y['duration_days'].values)
    contrib = M.brier_contrib(y_train_ipcw, ev, T, S, GRID)
    ibs_i = M.ibs_per_cve(contrib, GRID)
    r30, r365 = (1 - S[:, 29], 1 - S[:, 364])
    y30 = ((ev == 1) & (T <= 30)).astype(int)
    y365 = ((ev == 1) & (T <= 365)).astype(int)
    ids = y['cve_id'].values
    res = {'model': tag, 'n_test': len(y), 'events_test': int(ev.sum()), 'events_30d': int(y30.sum()), 'events_365d': int(y365.sum()), 'uno_c_tau365': M.uno_c(y_train_ipcw, ev, T, lp, TAU), 'c_tau365_no_ipcw_check': M.c_truncated_no_ipcw(ev, T, lp, TAU), 'ibs_1_365': float(ibs_i.mean()), 'ibs_1_365_no_ipcw_check': float(M.ibs_per_cve(M.brier_contrib(None, ev, T, S, GRID), GRID).mean()), 'brier_30d': float(np.mean((y30 - r30) ** 2)), 'brier_365d': float(np.mean((y365 - r365) ** 2)), **{f'recall_{b}_30d': M.topk_recall(r30, y30, b, ids) for b in BUDGETS}, **{f'recall_{b}_365d': M.topk_recall(r365, y365, b, ids) for b in BUDGETS}, 'cal_30d': M.calibration_intercept_slope(r30, y30), 'cal_365d': M.calibration_intercept_slope(r365, y365)}
    cal = []
    for h, p, yy in ((30, r30, y30), (365, r365, y365)):
        t, nb = M.calibration_table(p, yy, CFG['calibration_bins_default'], CFG['calibration_bins_fallback'])
        cal.append(t.assign(model=tag, horizon_days=h, n_bins=nb))
    pred = pd.DataFrame({'cve_id': ids, 'lp': lp, 'risk_30d': r30, 'risk_365d': r365, 'ibs_contrib': ibs_i, 'event': ev, 'duration_days': T, 'model': tag})
    return (res, pd.concat(cal), pred)

def coef_table(m, alpha, columns, tag) -> pd.DataFrame:
    j = int(np.argmin(np.abs(np.asarray(m.alphas_) - alpha)))
    return pd.DataFrame({'model': tag, 'column': columns, 'coef': m.coef_[:, j]})

def ph_check(columns_nz: list[str]) -> dict:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    X, y = load('disclosure', 'testmodel', 'B', 'ph')
    df = X[columns_nz].copy()
    df['T'] = y['duration_days'].values.astype(float) + 0.0
    df['E'] = y['event'].values
    df.loc[df['T'] <= 0, 'T'] = 0.5
    cph = CoxPHFitter(penalizer=0.01).fit(df, 'T', 'E')
    t = proportional_hazard_test(cph, df, time_transform='rank').summary
    out = {'n_vars': len(columns_nz), 'method': 'lifelines proportional_hazard_test, rank time, penalizer 0.01, sıfır süreler yalnız bu tanıda 0.5 gün', 'global_note': 'değişken başına test; çoklu test düzeltmesi Bonferroni', 'alpha_bonferroni': 0.05 / max(1, len(columns_nz)), 'tests': t[['test_statistic', 'p']].reset_index().rename(columns={'index': 'variable'}).to_dict(orient='records')}
    out['violations_bonferroni'] = [r['variable'] for r in out['tests'] if r['p'] < out['alpha_bonferroni']]
    return out

def main() -> None:
    t0 = utc_now()
    y_ipcw_lab = read_parquet(LAB / f'labels_testmodel_B_{SNAP}.parquet', 'S9', 'refit')
    y_ipcw = M.surv_y(y_ipcw_lab['event'], y_ipcw_lab['duration_days'])
    results, cals, preds, coefs, sels = ([], [], [], [], {})
    for fs in ('disclosure', 'current'):
        sel, grid = select(fs)
        grid.to_csv(TABLES / f's09_cox_selection_grid_{fs}_{SNAP}.csv', index=False)
        sels[fs] = sel
        Xtm, ytm = load(fs, 'testmodel', 'B', 'refit')
        m = fit(Xtm, ytm, sel['l1_ratio'], alphas=sel['alpha_path'])
        tag = 'cox_disclosure_B' if fs == 'disclosure' else 'cox_current_B'
        with open(MODELS / f'{tag}_{SNAP}.pkl', 'wb') as f:
            pickle.dump({'model': m, 'alpha': sel['alpha'], 'columns': list(Xtm.columns), 'selection': sel}, f)
        coefs.append(coef_table(m, sel['alpha'], list(Xtm.columns), tag))
        r, c, p = evaluate(m, sel['alpha'], fs, 'B', y_ipcw, tag)
        r['n_nonzero_testmodel'] = int((coefs[-1]['coef'] != 0).sum())
        results.append(r)
        cals.append(c)
        preds.append(p)
        if fs == 'disclosure':
            for s in ('A', 'B7'):
                Xs, ys = load(fs, 'testmodel', s, 'refit')
                ms = fit(Xs, ys, sel['l1_ratio'], alphas=sel['alpha_path'])
                tg = f'cox_disclosure_{s}'
                with open(MODELS / f'{tg}_{SNAP}.pkl', 'wb') as f:
                    pickle.dump({'model': ms, 'alpha': sel['alpha'], 'columns': list(Xs.columns), 'hyperparameters_from': 'B'}, f)
                coefs.append(coef_table(ms, sel['alpha'], list(Xs.columns), tg))
                r, c, p = evaluate(ms, sel['alpha'], fs, s, y_ipcw, tg)
                r['n_nonzero_testmodel'] = int((coefs[-1]['coef'] != 0).sum())
                r['training_events'] = int(ys['event'].sum())
                results.append(r)
                cals.append(c)
                preds.append(p)
    coef = pd.concat(coefs)
    coef.to_csv(TABLES / f's09_cox_coefficients_{SNAP}.csv', index=False)
    pd.concat(cals).to_csv(TABLES / f's09_cox_calibration_{SNAP}.csv', index=False)
    pr = pd.concat(preds)
    pr.to_parquet(PROC / f'predictions_test_cox_{SNAP}.parquet', index=False)
    nz = coef[(coef['model'] == 'cox_disclosure_B') & (coef['coef'] != 0)]['column'].tolist()
    ph = ph_check(nz)
    (METRICS / f's09_ph_diagnostics_{SNAP}.json').write_text(json.dumps(ph, indent=1, default=str), encoding='utf-8')
    flat = pd.json_normalize(results)
    flat.to_csv(TABLES / f's09_cox_test_metrics_{SNAP}.csv', index=False)
    out = {'started': t0, 'finished': utc_now(), 'selection': sels, 'test_metrics': results, 'ipcw_censoring_from': f'labels_testmodel_B_{SNAP}.parquet', 'grid_days': [1, 365], 'tau': TAU, 'zero_time_offset_applied': False, 'predictions_sha256': sha256_file(PROC / f'predictions_test_cox_{SNAP}.parquet')}
    (METRICS / f'cox_selection_{SNAP}.json').write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in sels['disclosure'].items() if k != 'alpha_path'}, indent=1))
    print(json.dumps({k: v for k, v in sels['current'].items() if k != 'alpha_path'}, indent=1))
    print(flat.drop(columns=[c for c in flat.columns if c.startswith('cal_')]).T.to_string())
    print(flat[[c for c in flat.columns if c.startswith('cal_')]].T.to_string())
    print('PH violations:', ph['violations_bonferroni'])
if __name__ == '__main__':
    main()
