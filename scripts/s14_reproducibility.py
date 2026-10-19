from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results' / 'metrics' / 's14_baseline_20261006.json'
OUT = ROOT / 'results' / 'metrics' / 's14_reproducibility_20261006.json'
DIRS = ['data/processed', 'data/interim', 'results/metrics', 'results/tables', 'results/models', 'manuscript/generated', 'manuscript/tables/generated', 'manuscript/figures/generated']
SKIP = {'s14_baseline_20261006.json', 's14_reproducibility_20261006.json'}
STAGES = ('s06', 's07', 's08', 's09', 's10', 's11', 's12', 's13', 'cohort_', 'labels_', 'X_', 'features_raw', 'feature_provenance', 'predictions_', 'epss_common', 'cox_', 'logit_', 'bootstrap_', 'disclosure_fields_s06', 'vendor_universe_s08', 'results_macros', 'T1_', 'T2_', 'T3_', 'T4_', 'A1_', 'A2_', 'F2_', 'F3_', 'F4_')
TOL = 1e-08

def files() -> list[Path]:
    out = []
    for d in DIRS:
        for p in sorted((ROOT / d).rglob('*')):
            if p.is_file() and p.name not in SKIP and p.name.startswith(STAGES) and ('__pycache__' not in p.parts):
                out.append(p)
    return out

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def key_metrics() -> dict:
    lock = json.load(open(ROOT / 'results' / 'metrics' / 's11_locked_evaluation_20261006.json', encoding='utf-8'))
    cox = json.load(open(ROOT / 'results' / 'metrics' / 'cox_selection_20261006.json', encoding='utf-8'))
    s10 = json.load(open(ROOT / 'results' / 'metrics' / 's10_summary_20261006.json', encoding='utf-8'))
    boot = json.load(open(ROOT / 'results' / 'metrics' / 'bootstrap_diffs_20261006.json', encoding='utf-8'))
    tm = {r['model']: r for r in cox['test_metrics']}
    t10 = {(r['model'], r['cohort']): r for r in s10['test_metrics']}
    return {'status': lock['status'], 'alpha': cox['selection']['disclosure']['alpha'], 'l1': cox['selection']['disclosure']['l1_ratio'], 'uno_c': tm['cox_disclosure_B']['uno_c_tau365'], 'ibs': tm['cox_disclosure_B']['ibs_1_365'], 'brier_cox': t10['cox_disclosure_B', 'test_all']['brier'], 'brier_logit': t10['logit_disclosure_B', 'test_all']['brier'], 'recall_cox': tm['cox_disclosure_B']['recall_0.01_30d'], 'recall_logit': t10['logit_disclosure_B', 'test_all']['recall_0.01'], 'leaky_c': tm['cox_current_B']['uno_c_tau365'], 'boot_ci': {k: boot[k]['ci95'] for k in ('cox_minus_logit', 'cox_minus_epss', 'leaky_minus_disclosure_cox')}}

def numeric_equal(a: Path, b_sha: str) -> tuple[bool, str]:
    return (False, 'temel çizgi içeriği saklanmadı; yalnız özet karşılaştırıldı')

def compare_frames(new: pd.DataFrame, old: pd.DataFrame) -> tuple[bool, float]:
    if list(new.columns) != list(old.columns) or len(new) != len(old):
        return (False, float('inf'))
    worst = 0.0
    for c in new.columns:
        if pd.api.types.is_numeric_dtype(new[c]) and pd.api.types.is_numeric_dtype(old[c]):
            d = np.nanmax(np.abs(new[c].to_numpy(float) - old[c].to_numpy(float))) if len(new) else 0.0
            worst = max(worst, float(d) if np.isfinite(d) else 0.0)
            if not np.allclose(new[c].to_numpy(float), old[c].to_numpy(float), atol=TOL, rtol=0, equal_nan=True):
                return (False, worst)
        elif not new[c].astype(str).equals(old[c].astype(str)):
            return (False, float('inf'))
    return (True, worst)

def main(cmd: str) -> None:
    if cmd == 'baseline':
        snap = ROOT / 'results' / 'metrics' / 's14_baseline_copies'
        snap.mkdir(exist_ok=True)
        rec = {}
        for p in files():
            rel = str(p.relative_to(ROOT)).replace('\\', '/')
            rec[rel] = sha(p)
            if p.suffix in ('.csv', '.parquet', '.json'):
                (snap / rel.replace('/', '__')).write_bytes(p.read_bytes())
        BASE.write_text(json.dumps({'files': rec, 'key_metrics': key_metrics()}, indent=1), encoding='utf-8')
        print(len(rec), 'dosya temel çizgiye alındı')
        return
    base = json.load(open(BASE, encoding='utf-8'))
    snap = ROOT / 'results' / 'metrics' / 's14_baseline_copies'
    rows = []
    for rel, h in base['files'].items():
        p = ROOT / rel
        if not p.exists():
            rows.append({'file': rel, 'status': 'missing'})
            continue
        if sha(p) == h:
            rows.append({'file': rel, 'status': 'identical'})
            continue
        old = snap / rel.replace('/', '__')
        status, worst = ('differs', None)
        if p.suffix in ('.csv', '.parquet') and old.exists():
            rd = pd.read_parquet if p.suffix == '.parquet' else pd.read_csv
            ok, worst = compare_frames(rd(p), rd(old))
            status = 'numeric_equal_tol' if ok else 'differs'
        elif p.suffix == '.json' and old.exists():
            a, b = (json.loads(p.read_text(encoding='utf-8')), json.loads(old.read_text(encoding='utf-8')))
            for k in ('started', 'finished', 'created', 'manifest_created_utc'):
                a.pop(k, None)
                b.pop(k, None)
            status = 'identical_except_timestamps' if a == b else 'differs_json'
        elif p.suffix in ('.pkl', '.pdf', '.png', '.npy'):
            status = 'binary_differs'
        rows.append({'file': rel, 'status': status, 'max_abs_diff': worst})
    km_new, km_old = (key_metrics(), base['key_metrics'])
    km_ok = all((abs(km_new[k] - km_old[k]) <= TOL for k in ('alpha', 'l1', 'uno_c', 'ibs', 'brier_cox', 'brier_logit', 'recall_cox', 'recall_logit', 'leaky_c'))) and km_new['status'] == 'locked'
    km_ok = km_ok and np.allclose(np.array(json.dumps(km_new['boot_ci']) == json.dumps(km_old['boot_ci'])), True)
    t = pd.DataFrame(rows)
    out = {'tolerance_abs': TOL, 'n_files': len(t), 'status_counts': t['status'].value_counts().to_dict(), 'key_metrics_reproduced': bool(km_ok), 'key_metrics_new': km_new, 'key_metrics_baseline': km_old, 'non_identical': t[t['status'] != 'identical'].to_dict(orient='records')}
    OUT.write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('n_files', 'status_counts', 'key_metrics_reproduced')}, indent=1))
    for r in out['non_identical']:
        print(r)
if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'compare')
