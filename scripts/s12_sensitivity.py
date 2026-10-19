from __future__ import annotations
import json
import math
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sksurv.linear_model import CoxnetSurvivalAnalysis
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kev_survival.eval import metrics as M
from kev_survival.io.access import read_parquet
from kev_survival.io.common import ROOT, load_project, sha256_file, utc_now
from kev_survival.report.macros import tr_num
CFG = load_project()
assert CFG['locked'] is True
SNAP = str(CFG['t_freeze']).replace('-', '')
PROC, LAB = (ROOT / 'data' / 'processed', ROOT / 'data' / 'processed' / 'labels')
MODELS, MET, TAB = (ROOT / 'results' / 'models', ROOT / 'results' / 'metrics', ROOT / 'results' / 'tables')
GRID = np.arange(1, 366, dtype=float)
REPS, SEED = (int(CFG['bootstrap_reps']), int(CFG['seeds'][1]))
VENDOR_COLS = ['vendor_cve_12m', 'vendor_kev_ever', 'vendor_kev_12m']
MAIN = pickle.load(open(MODELS / f'cox_disclosure_B_{SNAP}.pkl', 'rb'))
LOGIT = pickle.load(open(MODELS / f'logit_disclosure_B_{SNAP}.pkl', 'rb'))
ALPHA, L1, ALPHAS = (MAIN['alpha'], MAIN['model'].l1_ratio, MAIN['model'].alphas_)

def data(fs, ctx, s, phase):
    X = read_parquet(PROC / f'X_{fs}_{ctx}_{s}_{SNAP}.parquet', 'S12', phase)
    y = read_parquet(LAB / f'labels_{ctx}_{s}_{SNAP}.parquet', 'S12', phase)
    assert (X['cve_id'].values == y['cve_id'].values).all()
    return (X, y)

def fit_cox(X, y):
    return CoxnetSurvivalAnalysis(l1_ratio=L1, alphas=ALPHAS, fit_baseline_model=True, max_iter=100000).fit(X, M.surv_y(y['event'], y['duration_days']))

def cox_eval(m, Xte, yte, y_ipcw):
    lp = m.predict(Xte, alpha=ALPHA)
    h = m._get_baseline_model(ALPHA).cum_baseline_hazard_
    S = M.survival_matrix(lp, np.asarray(h.x, float), np.asarray(h.y, float), GRID)
    ev, T = (yte['event'].values, yte['duration_days'].values)
    ibs_i = M.ibs_per_cve(M.brier_contrib(y_ipcw, ev, T, S, GRID), GRID)
    return pd.DataFrame({'cve_id': yte['cve_id'].values, 'lp': lp, 'risk_30d': 1 - S[:, 29], 'ibs_contrib': ibs_i, 'event': ev, 'duration_days': T})

def logit_fit_pred(Xtr, ytr, Xte):
    y30 = ((ytr['event'] == 1) & (ytr['duration_days'] <= 30)).astype(int)
    el = (ytr['duration_days'] >= 30) | (y30 == 1)
    s = LOGIT['selection']
    m = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=s['l1_ratio'], C=s['C'], max_iter=5000, random_state=int(CFG['seeds'][0])).fit(Xtr[el.values], y30[el].values)
    return m.predict_proba(Xte)[:, 1]

def y30_of(p):
    return ((p['event'] == 1) & (p['duration_days'] <= 30)).astype(int).values

def rec1(score, y, idx=None):
    if idx is None:
        idx = np.arange(len(score))
    s, yy = (score[idx], y[idx])
    if yy.sum() == 0:
        return np.nan
    k = max(1, math.ceil(0.01 * len(idx)))
    return yy[np.lexsort((np.arange(len(idx)), -s))[:k]].sum() / yy.sum()

def point(p, y_ipcw):
    y30 = y30_of(p)
    return {'uno_c': M.uno_c(y_ipcw, p['event'], p['duration_days'], p['lp'], 365.0), 'ibs': float(p['ibs_contrib'].mean()), 'recall_1pct_30d': rec1(p['risk_30d'].values, y30), 'brier_30d': float(np.mean((y30 - p['risk_30d'].values) ** 2)), 'n': len(p), 'events': int(p['event'].sum()), 'events_30d': int(y30.sum())}

def paired_diff(a: pd.DataFrame, b: pd.DataFrame, y_ipcw, rng) -> dict:
    assert (a['cve_id'].values == b['cve_id'].values).all()
    ev, T, y30 = (a['event'].values, a['duration_days'].values, y30_of(a))
    out, fails = ([], 0)
    for _ in range(REPS):
        idx = rng.integers(0, len(a), len(a))
        try:
            v = [M.uno_c(y_ipcw, ev[idx], T[idx], a['lp'].values[idx], 365.0) - M.uno_c(y_ipcw, ev[idx], T[idx], b['lp'].values[idx], 365.0), a['ibs_contrib'].values[idx].mean() - b['ibs_contrib'].values[idx].mean(), rec1(a['risk_30d'].values, y30, idx) - rec1(b['risk_30d'].values, y30, idx)]
        except Exception:
            v = [np.nan] * 3
        if not np.all(np.isfinite(v)):
            fails += 1
        out.append(v)
    arr = np.asarray(out, float)
    ok = arr[np.isfinite(arr).all(axis=1)]
    pa, pb = (point(a, y_ipcw), point(b, y_ipcw))
    est = [pa['uno_c'] - pb['uno_c'], pa['ibs'] - pb['ibs'], pa['recall_1pct_30d'] - pb['recall_1pct_30d']]
    lo, hi = np.percentile(ok, [2.5, 97.5], axis=0)
    return {'metrics': ['uno_c_tau365', 'ibs_1_365', 'recall_1pct_30d'], 'estimate': est, 'ci95_low': lo.tolist(), 'ci95_high': hi.tolist(), 'failed_samples': fails, 'a': pa, 'b': pb}

def main() -> None:
    t0 = utc_now()
    rng = np.random.default_rng(SEED)
    Xtm, ytm = data('disclosure', 'testmodel', 'B', 'sensitivity_refit')
    Xte, yte = data('disclosure', 'test', 'B', 'sensitivity_eval')
    y_ipcw = M.surv_y(ytm['event'], ytm['duration_days'])
    cols = [c for c in Xtm.columns if c != 'cve_id']
    preds = read_parquet(PROC / f'predictions_test_cox_{SNAP}.parquet', 'S12', 'sensitivity_eval')
    main_p = preds[preds['model'] == 'cox_disclosure_B'].reset_index(drop=True)
    assert (main_p['cve_id'].values == yte['cve_id'].values).all()
    m0 = fit_cox(Xtm[cols].values, ytm)
    assert np.allclose(m0.coef_, MAIN['model'].coef_)
    res, rows = ({}, [])

    def add(analysis, variant, metrics: dict, ref: str, cohort: str, d: dict | None=None, note=''):
        r = {'analysis': analysis, 'variant': variant, 'cohort': cohort, 'reference': ref, **{k: metrics[k] for k in ('n', 'events', 'events_30d', 'uno_c', 'ibs', 'brier_30d', 'recall_1pct_30d')}, 'note': note}
        if d:
            for i, k in enumerate(d['metrics']):
                r[f'diff_{k}'], r[f'diff_{k}_low'], r[f'diff_{k}_high'] = (d['estimate'][i], d['ci95_low'][i], d['ci95_high'][i])
            r['failed_samples'] = d['failed_samples']
        rows.append(r)
    add('main', 'cox_disclosure_B', point(main_p, y_ipcw), '-', 'test_all')
    zt = ~((ytm['event'] == 1) & (ytm['duration_days'] == 0)).values
    ze = ~((yte['event'] == 1) & (yte['duration_days'] == 0)).values
    yz = ytm[zt]
    mz = fit_cox(Xtm.loc[zt, cols].values, yz)
    pz = cox_eval(mz, Xte.loc[ze, cols].values, yte[ze], M.surv_y(yz['event'], yz['duration_days']))
    pm_z = cox_eval(MAIN['model'], Xte.loc[ze, cols].values, yte[ze], M.surv_y(yz['event'], yz['duration_days']))
    d = paired_diff(pz, pm_z, M.surv_y(yz['event'], yz['duration_days']), rng)
    res['zero_day_excluded'] = {**d, 'excluded_train_zero_day': int((~zt).sum()), 'excluded_test_zero_day': int((~ze).sum())}
    add('zero_day_excluded', 'cox_refit_without_zero_day', d['a'], 'main model on same restricted test set', 'test_without_zero_day', d)
    lz = logit_fit_pred(Xtm.loc[zt, cols].values, yz, Xte.loc[ze, cols].values)
    lm = logit_fit_pred(Xtm[cols].values, ytm, Xte.loc[ze, cols].values)
    y30z = y30_of(yte[ze])
    res['zero_day_excluded_logit'] = {'recall_1pct_30d_refit': rec1(lz, y30z), 'recall_1pct_30d_main': rec1(lm, y30z), 'brier_30d_refit': float(np.mean((y30z - lz) ** 2)), 'brier_30d_main': float(np.mean((y30z - lm) ** 2)), 'events_30d': int(y30z.sum())}
    pb7 = preds[preds['model'] == 'cox_disclosure_B7'].reset_index(drop=True)
    d = paired_diff(pb7, main_p, y_ipcw, rng)
    res['B7'] = d
    add('B7', 'cox_disclosure_B7', d['a'], 'main', 'test_all', d, 'JSON 4 sadakati 7 gün; B hiperparametreleri')
    pa = preds[preds['model'] == 'cox_disclosure_A'].reset_index(drop=True)
    d = paired_diff(pa, main_p, y_ipcw, rng)
    res['A'] = d
    add('scenario_A', 'cox_disclosure_A', d['a'], 'main', 'test_all', d, 'yalnız cvelistV5; B hiperparametreleri')
    sdt = pd.read_parquet(PROC / f'cohort_{SNAP}.parquet', columns=['cve_id', 'date_quality_class'])
    sd_ids = set(sdt.loc[sdt['date_quality_class'] == 'same_day_tz', 'cve_id'])
    keep = ~ytm['cve_id'].isin(sd_ids).values
    ms = fit_cox(Xtm.loc[keep, cols].values, ytm[keep])
    ps = cox_eval(ms, Xte[cols].values, yte, y_ipcw)
    d = paired_diff(ps, main_p, y_ipcw, rng)
    res['same_day_tz_excluded'] = {**d, 'excluded_train': int((~keep).sum()), 'in_test': int(yte['cve_id'].isin(sd_ids).sum())}
    add('same_day_tz_excluded', 'cox_refit_without_same_day_tz', d['a'], 'main', 'test_all', d)
    mv = fit_cox(Xtm[VENDOR_COLS].values, ytm)
    pv = cox_eval(mv, Xte[VENDOR_COLS].values, yte, y_ipcw)
    d = paired_diff(pv, main_p, y_ipcw, rng)
    res['vendor_only'] = {**d, 'n_nonzero': int((mv.coef_[:, int(np.argmin(np.abs(ALPHAS - ALPHA)))] != 0).sum())}
    add('vendor_only_ablation', 'cox_vendor_history_only', d['a'], 'main', 'test_all', d, '3 terim; aynı alpha')
    lv = logit_fit_pred(Xtm[VENDOR_COLS].values, ytm, Xte[VENDOR_COLS].values)
    y30 = y30_of(yte)
    p_log = LOGIT['model'].predict_proba(Xte[cols].values)[:, 1]
    res['vendor_only_logit'] = {'recall_1pct_30d': rec1(lv, y30), 'brier_30d': float(np.mean((y30 - lv) ** 2)), 'main_logit_recall_1pct_30d': rec1(p_log, y30), 'main_logit_brier_30d': float(np.mean((y30 - p_log) ** 2))}
    boot = json.load(open(MET / f'bootstrap_diffs_{SNAP}.json', encoding='utf-8'))
    cox = json.load(open(MET / f'cox_selection_{SNAP}.json', encoding='utf-8'))
    s10 = json.load(open(MET / f's10_summary_{SNAP}.json', encoding='utf-8'))
    res['leakage'] = boot['leaky_minus_disclosure_cox']
    res['ipcw_vs_unweighted'] = [{'model': r['model'], 'uno_c_ipcw': r['uno_c_tau365'], 'c_unweighted': r['c_tau365_no_ipcw_check'], 'ibs_ipcw': r['ibs_1_365'], 'ibs_unweighted': r['ibs_1_365_no_ipcw_check']} for r in cox['test_metrics']]
    ec = pd.read_parquet(PROC / f'epss_common_cohort_{SNAP}.parquet')
    epss_tol = []
    for r in s10['test_metrics']:
        if r['cohort'].startswith('epss_fair'):
            epss_tol.append({'tolerance': r['cohort'], 'model': r['model'], 'n': r['n'], 'events': r['events'], 'recall_1pct': r['recall_0.01'], 'brier': r['brier']})
    res['epss_tolerances'] = epss_tol
    j = int(np.argmin(np.abs(ALPHAS - ALPHA)))
    coef = MAIN['model'].coef_[:, j]
    Xa, ya = (Xtm[cols].values, ytm)
    stab = np.zeros((REPS, len(cols)))
    fails = 0
    rs = np.random.default_rng(int(CFG['seeds'][2]))
    for b in range(REPS):
        idx = rs.integers(0, len(Xa), len(Xa))
        try:
            mb = CoxnetSurvivalAnalysis(l1_ratio=L1, alphas=ALPHAS[:j + 1], max_iter=100000).fit(Xa[idx], M.surv_y(ya['event'].values[idx], ya['duration_days'].values[idx]))
            stab[b] = mb.coef_[:, j]
        except Exception:
            stab[b] = np.nan
            fails += 1
    ok = stab[np.isfinite(stab).all(axis=1)]
    ct = pd.DataFrame({'column': cols, 'coef': coef, 'hr_per_sd': np.exp(coef), 'selection_rate': (ok != 0).mean(axis=0), 'sign_agreement': [float(np.mean(np.sign(ok[:, i][ok[:, i] != 0]) == np.sign(coef[i]))) if coef[i] != 0 and (ok[:, i] != 0).any() else np.nan for i in range(len(cols))], 'hr_q025': np.exp(np.percentile(ok, 2.5, axis=0)), 'hr_median': np.exp(np.median(ok, axis=0)), 'hr_q975': np.exp(np.percentile(ok, 97.5, axis=0))})
    ct['selected_main'] = ct['coef'] != 0
    ct['stable'] = (ct['selection_rate'] >= 0.8) & (ct['sign_agreement'] >= 0.95)
    ct = ct.sort_values('coef', key=np.abs, ascending=False)
    ct.to_csv(TAB / f's12_coefficient_summary_{SNAP}.csv', index=False)
    res['coefficient_stability'] = {'reps': REPS, 'failed': fails, 'seed': int(CFG['seeds'][2]), 'data': 'labels_testmodel_B (eğitim+doğrulama); test kullanılmadı', 'n_selected_main': int(ct['selected_main'].sum()), 'n_stable': int(ct['stable'].sum()), 'stable_columns': ct.loc[ct['stable'], 'column'].tolist(), 'rule': 'seçilme oranı ≥ 0,8 ve işaret uyumu ≥ 0,95 (önceden tanımlı betimleyici eşik)'}
    sens = pd.DataFrame(rows)
    abl = sens[sens['analysis'].isin(['main', 'vendor_only_ablation'])]
    sens[~sens['analysis'].isin(['vendor_only_ablation'])].to_csv(TAB / f's12_sensitivity_results_{SNAP}.csv', index=False)
    abl.to_csv(TAB / f's12_ablation_results_{SNAP}.csv', index=False)
    verdict = {}
    for k in ('zero_day_excluded', 'B7', 'A', 'same_day_tz_excluded', 'vendor_only'):
        d = res[k]
        verdict[k] = {'delta_uno_c': d['estimate'][0], 'ci95': [d['ci95_low'][0], d['ci95_high'][0]], 'c_above_0_5': d['a']['uno_c'] > 0.5, 'ci_includes_zero': d['ci95_low'][0] <= 0 <= d['ci95_high'][0]}
    res['verdict'] = verdict
    f3 = lambda x: tr_num(x, 3)
    macros = {'zeroDayC': f3(res['zero_day_excluded']['a']['uno_c']), 'zeroDayMainC': f3(res['zero_day_excluded']['b']['uno_c']), 'zeroDayDiffC': f3(res['zero_day_excluded']['estimate'][0]), 'zeroDayDiffCLow': f3(res['zero_day_excluded']['ci95_low'][0]), 'zeroDayDiffCHigh': f3(res['zero_day_excluded']['ci95_high'][0]), 'zeroDayRecall': f3(res['zero_day_excluded']['a']['recall_1pct_30d']), 'zeroDayTestEvents': tr_num(res['zero_day_excluded']['excluded_test_zero_day'], 0), 'bsevenDiffC': f3(res['B7']['estimate'][0]), 'bsevenDiffCLow': f3(res['B7']['ci95_low'][0]), 'bsevenDiffCHigh': f3(res['B7']['ci95_high'][0]), 'scenarioADiffC': f3(res['A']['estimate'][0]), 'scenarioADiffCLow': f3(res['A']['ci95_low'][0]), 'scenarioADiffCHigh': f3(res['A']['ci95_high'][0]), 'sameDayDiffC': f3(res['same_day_tz_excluded']['estimate'][0]), 'vendorC': f3(res['vendor_only']['a']['uno_c']), 'vendorRecall': f3(res['vendor_only']['a']['recall_1pct_30d']), 'vendorDiffC': f3(res['vendor_only']['estimate'][0]), 'vendorDiffCLow': f3(res['vendor_only']['ci95_low'][0]), 'vendorDiffCHigh': f3(res['vendor_only']['ci95_high'][0]), 'vendorDiffRecall': f3(res['vendor_only']['estimate'][2]), 'vendorDiffRecallLow': f3(res['vendor_only']['ci95_low'][2]), 'vendorDiffRecallHigh': f3(res['vendor_only']['ci95_high'][2]), 'nStableCoef': tr_num(res['coefficient_stability']['n_stable'], 0), 'nSelectedCoef': tr_num(res['coefficient_stability']['n_selected_main'], 0)}
    out = {'started': t0, 'finished': utc_now(), 'reps': REPS, 'seed': SEED, 'results': res, 'macros': macros, 'inputs_sha256': {'predictions_test_cox': sha256_file(PROC / f'predictions_test_cox_{SNAP}.parquet'), 'cox_model': sha256_file(MODELS / f'cox_disclosure_B_{SNAP}.pkl')}}
    (MET / f's12_sensitivity_summary_{SNAP}.json').write_text(json.dumps(out, indent=1, default=str, ensure_ascii=False), encoding='utf-8')
    print(sens.round(4).to_string())
    print(json.dumps(verdict, indent=1))
    print(ct.round(3).head(25).to_string())
    print(json.dumps({k: res[k] for k in ('zero_day_excluded_logit', 'vendor_only_logit')}, indent=1))
    print(json.dumps(res['coefficient_stability'], indent=1, ensure_ascii=False))
if __name__ == '__main__':
    main()
