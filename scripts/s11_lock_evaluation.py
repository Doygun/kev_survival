from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kev_survival.eval import metrics as M
from kev_survival.io.common import ROOT, load_project, sha256_file, utc_now
CFG = load_project()
SNAP = str(CFG['t_freeze']).replace('-', '')
MET, TAB, PROC = (ROOT / 'results' / 'metrics', ROOT / 'results' / 'tables', ROOT / 'data' / 'processed')
LOCKED = {'l1_ratio': 0.9, 'alpha_approx': 0.000378, 'scenario': 'B', 'n_test': 39384, 'events_test': 154}

def load():
    cox = json.load(open(MET / f'cox_selection_{SNAP}.json', encoding='utf-8'))
    s10 = json.load(open(MET / f's10_summary_{SNAP}.json', encoding='utf-8'))
    boot = json.load(open(MET / f'bootstrap_diffs_{SNAP}.json', encoding='utf-8'))
    ph = json.load(open(MET / f's09_ph_diagnostics_{SNAP}.json', encoding='utf-8'))
    return (cox, s10, boot, ph)

def access_audit() -> dict:
    a = pd.DataFrame([json.loads(x) for x in (ROOT / 'logs' / 'file_access.jsonl').read_text(encoding='utf-8').splitlines()])
    a['run_context'] = a.get('run_context', pd.Series(index=a.index, dtype=object)).fillna('primary')
    test = a[a['path'].str.contains('labels_test_|X_\\w+_test_|predictions_test', regex=True)]
    reads = test[test['mode'] == 'read']
    sel = a[a['phase'] == 'selection']
    runs = reads.groupby(['run_context', 'stage', 'phase']).agg(n_reads=('path', 'size'), first=('time', 'min'), last=('time', 'max'), files=('path', lambda s: sorted(set(s)))).reset_index()
    first_test = reads['time'].min()
    sel_after_test = sel[sel['time'] > first_test]
    return {'n_log_entries': len(a), 'test_file_reads': int(len(reads)), 'test_reads_by_stage_phase': runs.to_dict(orient='records'), 'selection_reads': int(len(sel)), 'selection_reads_test_files': int(sel['path'].str.contains('_test_').sum()), 'selection_after_first_test_read': sel_after_test[['stage', 'phase', 'time', 'path']].to_dict(orient='records'), 'first_test_read': first_test, 'phases_reading_test': sorted(set(reads['phase']))}

def recompute(cox_json) -> list[dict]:
    pr = pd.read_parquet(PROC / f'predictions_test_cox_{SNAP}.parquet')
    lab_tm = pd.read_parquet(PROC / 'labels' / f'labels_testmodel_B_{SNAP}.parquet')
    ytr = M.surv_y(lab_tm['event'], lab_tm['duration_days'])
    out = []
    ref = {r['model']: r for r in cox_json['test_metrics']}
    for tag, g in pr.groupby('model'):
        y30 = ((g['event'] == 1) & (g['duration_days'] <= 30)).astype(int).values
        mine = {'uno_c_tau365': M.uno_c(ytr, g['event'], g['duration_days'], g['lp'], 365.0), 'ibs_1_365': float(g['ibs_contrib'].mean()), 'brier_30d': float(np.mean((y30 - g['risk_30d'].values) ** 2)), 'recall_0.01_30d': M.topk_recall(g['risk_30d'].values, y30, 0.01, g['cve_id'].values)}
        for k, v in mine.items():
            out.append({'model': tag, 'metric': k, 'json': ref[tag][k], 'recomputed': v, 'abs_diff': abs(ref[tag][k] - v)})
    return out

def reselect() -> bool:
    import importlib.util
    from kev_survival.io.access import read_parquet as rp
    spec = importlib.util.spec_from_file_location('s09', ROOT / 'scripts' / 's09_cox.py')
    s09 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s09)

    def load_audit(fs, ctx, s, phase):
        assert ctx in ('validmodel', 'valid'), ctx
        X = rp(PROC / f'X_{fs}_{ctx}_{s}_{SNAP}.parquet', 'S11', 'selection_audit')
        y = rp(PROC / 'labels' / f'labels_{ctx}_{s}_{SNAP}.parquet', 'S11', 'selection_audit')
        return (X.drop(columns='cve_id'), y)
    s09.load = load_audit
    ok = True
    for fs in ('disclosure', 'current'):
        _, g = s09.select(fs)
        ref = pd.read_csv(TAB / f's09_cox_selection_grid_{fs}_{SNAP}.csv')
        ok &= bool(np.allclose(g['uno_c_valid'].values, ref['uno_c_valid'].values, atol=1e-12) and (g['selected'].values == ref['selected'].values).all() and (g['n_nonzero'].values == ref['n_nonzero'].values).all())
    return ok

def main() -> None:
    cox, s10, boot, ph = load()
    sel = cox['selection']['disclosure']
    b = pickle.load(open(ROOT / 'results' / 'models' / f'cox_disclosure_B_{SNAP}.pkl', 'rb'))
    checks = {'locked_l1_ratio': sel['l1_ratio'] == LOCKED['l1_ratio'] == b['model'].l1_ratio, 'locked_alpha': abs(sel['alpha'] - LOCKED['alpha_approx']) < 5e-06 and b['alpha'] == sel['alpha'], 'model_family_cox_and_logistic_only': sorted((p.name.split('_')[0] for p in (ROOT / 'results' / 'models').glob('*.pkl'))) in (['cox'] * 4 + ['logit'],), 'feature_terms_locked_27': len(json.load(open(MET / f's08_encoders_{SNAP}.json', encoding='utf-8'))['terms']) == 27}
    aud = access_audit()
    checks['no_test_files_in_selection'] = aud['selection_reads_test_files'] == 0
    checks['selection_reproduced_without_test_access'] = reselect()
    rec = recompute(cox)
    checks['json_matches_recomputed_1e-9'] = max((r['abs_diff'] for r in rec)) < 1e-09
    tm = {r['model']: r for r in cox['test_metrics']}
    t10 = {(r['model'], r['cohort']): r for r in s10['test_metrics']}
    cb, lg = (t10['cox_disclosure_B', 'test_all'], t10['logit_disclosure_B', 'test_all'])
    checks['test_set_size'] = tm['cox_disclosure_B']['n_test'] == LOCKED['n_test'] and tm['cox_disclosure_B']['events_test'] == LOCKED['events_test']
    checks['s9_s10_cox_consistent'] = abs(cb['brier'] - tm['cox_disclosure_B']['brier_30d']) < 1e-12 and abs(cb['recall_0.01'] - tm['cox_disclosure_B']['recall_0.01_30d']) < 1e-12
    cl, ce, lk = (boot['cox_minus_logit'], boot['cox_minus_epss'], boot['leaky_minus_disclosure_cox'])
    ibs_null = s10['ibs_reference_km_null']
    ibs_cox = tm['cox_disclosure_B']['ibs_1_365']
    interp = [{'id': 1, 'text_tr': 'Cox ve lojistik recall@%1 bakımından eşittir.', 'holds': cb['recall_0.01'] == lg['recall_0.01'] and cl['ci95'][0][0] < 0 < cl['ci95'][1][0], 'evidence': {'cox': cb['recall_0.01'], 'logit': lg['recall_0.01'], 'diff_ci95': [cl['ci95'][0][0], cl['ci95'][1][0]]}}, {'id': 2, 'text_tr': 'Cox, 30 günlük Brier puanında lojistikten anlamlı biçimde daha iyidir.', 'holds': cl['ci95'][1][1] < 0, 'evidence': {'diff': cl['estimate'][1], 'ci95': [cl['ci95'][0][1], cl['ci95'][1][1]]}}, {'id': 3, 'text_tr': 'Cox ile EPSS arasında üstünlük veya eşdeğerlik sonucu çıkarılamaz.', 'holds': all((ce['ci95'][0][i] < 0 < ce['ci95'][1][i] for i in (0, 1))) and ce['events_30d'] == 44, 'evidence': {'events_30d': ce['events_30d'], 'ci95_recall': [ce['ci95'][0][0], ce['ci95'][1][0]], 'ci95_brier': [ce['ci95'][0][1], ce['ci95'][1][1]]}}, {'id': 4, 'text_tr': 'Sızıntılı özellikler bütün ana ölçütlerde performansı anlamlı biçimde şişirmiştir.', 'holds': lk['ci95'][0][0] > 0 and lk['ci95'][1][1] < 0 and (lk['ci95'][0][2] > 0), 'evidence': {'metrics': lk['metrics'], 'estimate': lk['estimate'], 'ci95': lk['ci95']}}, {'id': 5, 'text_tr': 'Cox ve lojistik 2024 mutlak riskini fazla tahmin etmiştir.', 'holds': cb['cal']['o_e_ratio'] < 1 and lg['cal']['o_e_ratio'] < 1, 'evidence': {'o_e_cox_30d': cb['cal']['o_e_ratio'], 'o_e_logit_30d': lg['cal']['o_e_ratio'], 'o_e_cox_365d': tm['cox_disclosure_B']['cal_365d']['o_e_ratio']}}, {'id': 6, 'text_tr': 'PH varsayımında Bonferroni düzeyinde anlamlı ihlal yoktur.', 'holds': len(ph['violations_bonferroni']) == 0, 'evidence': {'alpha_bonferroni': ph['alpha_bonferroni'], 'n_nominal_p_lt_0_05': sum((t['p'] < 0.05 for t in ph['tests']))}}, {'id': 7, 'text_tr': "Cox'un IBS kazanımı Kaplan-Meier referansına göre küçüktür ve abartılmamalıdır.", 'holds': (ibs_null - ibs_cox) / ibs_null < 0.1, 'evidence': {'ibs_cox': ibs_cox, 'ibs_km_null': ibs_null, 'relative_gain': (ibs_null - ibs_cox) / ibs_null}}]
    prim = [('cox_disclosure_B', 'uno_c_tau365_ipcw', tm['cox_disclosure_B']['uno_c_tau365']), ('cox_disclosure_B', 'ibs_1_365_ipcw', ibs_cox), ('km_null_reference', 'ibs_1_365_ipcw', ibs_null), ('cox_disclosure_B', 'brier_30d', cb['brier']), ('logit_disclosure_B', 'brier_30d', lg['brier']), ('cox_disclosure_B', 'recall_1pct_30d', cb['recall_0.01']), ('logit_disclosure_B', 'recall_1pct_30d', lg['recall_0.01']), ('cox_current_B', 'uno_c_tau365_ipcw', tm['cox_current_B']['uno_c_tau365']), ('cox_current_B', 'ibs_1_365_ipcw', tm['cox_current_B']['ibs_1_365']), ('cox_current_B', 'recall_1pct_30d', tm['cox_current_B']['recall_0.01_30d'])]
    pd.DataFrame(prim, columns=['model', 'metric', 'value']).assign(n_test=LOCKED['n_test']).to_csv(TAB / f's11_primary_results_{SNAP}.csv', index=False)
    rob = []
    for m in ('cox_disclosure_B', 'cox_current_B', 'cox_disclosure_A', 'cox_disclosure_B7'):
        r = tm[m]
        rob += [(m, 'test_all', 'c_tau365_unweighted', r['c_tau365_no_ipcw_check']), (m, 'test_all', 'ibs_1_365_unweighted', r['ibs_1_365_no_ipcw_check'])]
    for (m, coh), r in t10.items():
        for k in ('recall_0.005', 'recall_0.01', 'recall_0.02', 'brier'):
            rob.append((m, coh, k, r[k]))
    pd.DataFrame(rob, columns=['model', 'cohort', 'metric', 'value']).to_csv(TAB / f's11_robustness_results_{SNAP}.csv', index=False)
    cal = []
    for (m, coh), r in t10.items():
        cal.append({'model': m, 'cohort': coh, 'horizon_days': 30, **r['cal']})
    for m in ('cox_disclosure_B', 'cox_current_B', 'cox_disclosure_A', 'cox_disclosure_B7'):
        cal.append({'model': m, 'cohort': 'test_all', 'horizon_days': 365, **tm[m]['cal_365d']})
    pd.DataFrame(cal).to_csv(TAB / f's11_calibration_summary_{SNAP}.csv', index=False)
    bs = []
    for k in ('cox_minus_logit', 'cox_minus_epss', 'leaky_minus_disclosure_cox'):
        d = boot[k]
        for i, met in enumerate(d['metrics']):
            lo, hi = (d['ci95'][0][i], d['ci95'][1][i])
            bs.append({'comparison': k, 'cohort': d['cohort'], 'metric': met, 'estimate': d['estimate'][i], 'ci95_low': lo, 'ci95_high': hi, 'excludes_zero': not lo <= 0 <= hi, 'failed_samples': d['failed_samples'], 'reps': boot['reps']})
    pd.DataFrame(bs).to_csv(TAB / f's11_bootstrap_summary_{SNAP}.csv', index=False)
    ok = all(checks.values()) and all((i['holds'] for i in interp))
    out = {'created': utc_now(), 'status': 'locked' if ok else 'FAILED', 'checks': checks, 'locked_decisions': {**LOCKED, 'alpha': sel['alpha'], 'primary_metrics_ipcw': True, 'unweighted_metrics_role': 'robustness_only'}, 'test_access': aud, 'recomputation': rec, 'interpretations': interp, 'sources_sha256': {p: sha256_file(MET / p) for p in (f'cox_selection_{SNAP}.json', f's10_summary_{SNAP}.json', f'bootstrap_diffs_{SNAP}.json', f's09_ph_diagnostics_{SNAP}.json')}}
    (MET / f's11_locked_evaluation_{SNAP}.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    write_audit(aud, checks)
    print(json.dumps(checks, indent=1))
    print([(i['id'], i['holds']) for i in interp])
    print('STATUS', out['status'])
    if not ok:
        sys.exit(1)

def write_audit(aud: dict, checks: dict) -> None:
    rows = '\n'.join((f"| {r['run_context']} | {r['stage']} | {r['phase']} | {r['n_reads']} | {r['first']} | {r['last']} | {', '.join((Path(f).name for f in r['files']))} |" for r in aud['test_reads_by_stage_phase']))
    sel_after = aud['selection_after_first_test_read']
    txt = f"# Test Kümesi Erişim Denetimi\n\nÜretim: `scripts/s11_lock_evaluation.py` (otomatik). Kaynak: `logs/file_access.jsonl` ({aud['n_log_entries']} kayıt).\n\n## 1. Bağlam\n\nproje planı, test kümesinin yalnızca S11'de ve bir kez açılmasını öngörür. Kullanıcı 2026-10-06 tarihli talimatıyla test değerlendirmesini S9 ve S10'da istedi. Test kümesi bu nedenle S9'da açıldı. S11 test kümesini yeniden açmaz; yalnızca kaydedilmiş tahminleri ve ölçütleri denetler ve kilitler.\n\n## 2. Test dosyası okumaları\n\n| Çalıştırma | Aşama | Evre | Okuma | İlk | Son | Dosyalar |\n|---|---|---|---|---|---|---|\n{rows}\n\n`primary` asıl çalıştırmayı, `S14_repro` ise S14 tekrar üretilebilirlik denetimindeki yeniden çalıştırmayı gösterir. `test_eval`, `sensitivity_eval` (S12, önceden planlanmış duyarlılık) ve `artifact` (S13, kilitli modelden şekil verisi) dışında test dosyası okunmadı: S7 ve S8 `build` evreleri test etiket ve matris dosyalarını üretti; `refit` evrelerinde okunan `predictions_test_cox` dosyası kaydedilmiş tahmindir.\n\nS9 test değerlendirmesi iki çalıştırmada kayıtlıdır. İlk çalıştırma, değerlendirme kodundaki bir hata (IPCW'siz kontrol ölçütünün tanımı) nedeniyle tamamlanmadan durdu. Hata düzeltildi ve betik aynı kilitli hiperparametrelerle yeniden çalıştırıldı. İki çalıştırma arasında seçim, öznitelik, ızgara veya model değişmedi. Bu, test sonuçlarına göre yapılmış bir değişiklik değildir. İlk çalıştırmanın yazdığı seçim ızgarası ve model dosyası, ikinci çalıştırmada aynı içerikle yeniden yazıldı. İlk çalıştırma hiçbir test sonucu dosyası yazmadı.\n\n## 3. Hiperparametre seçimi\n\nSeçim evresinde {aud['selection_reads']} okuma kayıtlıdır; bunların {aud['selection_reads_test_files']} tanesi test dosyasıdır.\n\nKayıtta S9 seçim okumalarının bir kısmı ilk test okumasından sonra görünür. Bunun iki nedeni vardır:\n- Hatalı ilk çalıştırmadan sonra betik baştan çalıştırıldı.\n- Sızıntılı Cox'un seçimi, betiğin sırası gereği yayın anı Cox'unun test değerlendirmesinden sonra yapılır.\n\nSeçim işlevi yalnızca `validmodel` ve `valid` dosyalarını okur ve deterministiktir. S11'de seçim, test dosyalarına erişmeden yeniden çalıştırıldı (`selection_audit` evresi). Elde edilen ızgara, seçilen model ve katsayı sayıları kilitli S9 ızgarasıyla birebir aynı çıktı (`selection_reproduced_without_test_access`).\n\nS10 lojistik seçimi, S9 test değerlendirmesinden sonra çalıştırıldı. Seçim yalnızca `validmodel` ve `valid` dosyalarını okudu (S10'da {len([r for r in sel_after if r['stage'] == 'S10'])} seçim okuması). Izgara ve seçim kuralı S5'te kilitlendi (`configs/models.yaml`). Test sonuçları lojistik seçimine girmedi.\n\n## 4. Kayıt dışı okumalar\n\nPytest doğrulama testleri (`tests/test_s9_cox.py`, `tests/test_s10_logit_epss.py`) değerlendirmeden sonra test etiketlerini ve matrislerini doğrudan okur. Bu okumalar erişim kaydına girmez ve hiçbir modeli veya seçimi etkilemez.\n\n## 5. Denetim sonuçları\n\n{chr(10).join((f'- `{k}`: {v}' for k, v in checks.items()))}\n"
    (ROOT / 'reports' / 'test_access_audit.md').write_text(txt, encoding='utf-8')
if __name__ == '__main__':
    main()
