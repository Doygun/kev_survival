from __future__ import annotations
import datetime as dt
import gzip
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kev_survival.features import fields as fd
from kev_survival.features.matrix import TERMS, Encoder, raw_terms, vendor_history
from kev_survival.io.access import read_parquet, write_parquet
from kev_survival.io.common import ROOT, load_project, raw_dir, sha256_file
from kev_survival.reconstruct import cvelist as cl
CFG = load_project()
assert CFG['locked'] is True
FCFG = yaml.safe_load(open(ROOT / 'configs' / 'features.yaml', encoding='utf-8'))
assert FCFG['locked'] is True and len(TERMS) == FCFG['n_feature_terms']
SNAP = str(CFG['t_freeze']).replace('-', '')
T_FREEZE = dt.date.fromisoformat(str(CFG['t_freeze']))
INTERIM, PROC = (ROOT / 'data' / 'interim', ROOT / 'data' / 'processed')
UNIV = INTERIM / f'vendor_universe_s08_{SNAP}.parquet'
LAB = PROC / 'labels'
GROUP = {t['name']: t['group'] for t in FCFG['features']}

def universe_head() -> pd.DataFrame:
    if UNIV.exists():
        return pd.read_parquet(UNIV)
    repo = raw_dir() / 'cvelistV5'
    paths = cl.head_paths(repo)
    rows = []
    for spec, data in cl.cat_file_batch(repo, (f'HEAD:{p}' for p in paths.values())):
        try:
            r = cl.load_record(data)
        except Exception:
            continue
        m = r.get('cveMetadata') or {}
        aff = ((r.get('containers') or {}).get('cna') or {}).get('affected') or []
        vend = [a.get('vendor') for a in aff if isinstance(a, dict) and fd._ok(a.get('vendor'))]
        d = cl.parse_date_utc(m.get('datePublished'))
        rows.append({'cve_id': m.get('cveId'), 'state': m.get('state'), 't_pub_current': d.date() if d else None, 'vendor_key_current': fd.vendor_key(vend[0]) if vend else None})
    u = pd.DataFrame(rows).dropna(subset=['cve_id']).drop_duplicates('cve_id').sort_values('cve_id')
    u.to_parquet(UNIV, index=False)
    return u

def nvd_fields(ids: set[str]) -> pd.DataFrame:
    rows = []
    for f in sorted((raw_dir() / 'nvd_cves').glob('*.json.gz')):
        with gzip.open(f, 'rt', encoding='utf-8') as fh:
            d = json.load(fh)
        for v in d.get('vulnerabilities', []):
            c = v.get('cve', {})
            if c.get('id') in ids:
                rows.append({'cve_id': c['id'], **fd.parse_nvd(c)})
    return pd.DataFrame(rows).drop_duplicates('cve_id', keep='last').set_index('cve_id')

def main() -> None:
    coh = read_parquet(PROC / f'cohort_{SNAP}.parquet', 'S8', 'build')
    flds = read_parquet(INTERIM / f'disclosure_fields_s06_{SNAP}.parquet', 'S8', 'build').set_index('cve_id')
    kev = pd.DataFrame(json.load(open(raw_dir() / 'cisa_kev' / 'known_exploited_vulnerabilities.json', encoding='utf-8'))['vulnerabilities']).rename(columns={'cveID': 'cve_id'})
    kev['t_kev'] = pd.to_datetime(kev['dateAdded']).dt.date
    kev = kev[['cve_id', 't_kev']]
    head = universe_head()
    uni = head[head['state'] == 'PUBLISHED'].rename(columns={'t_pub_current': 't_pub', 'vendor_key_current': 'vendor_key'})
    uni = uni[['cve_id', 't_pub', 'vendor_key']].set_index('cve_id')
    cv = coh.set_index('cve_id')
    uni.loc[uni.index.intersection(cv.index), 't_pub'] = cv['t_pub']
    uni.loc[uni.index.intersection(cv.index), 'vendor_key'] = cv['vendor_key']
    uni = uni.reset_index()
    target = coh[['cve_id', 't_pub', 'vendor_key']].reset_index(drop=True)
    vh_d = vendor_history(target, uni, kev, current=False)
    vh_c = vendor_history(target, uni, kev, current=True, t_freeze=T_FREEZE)
    flags = FCFG['description_flag_regex']
    fdisc = flds.loc[target['cve_id']].reset_index()
    xd = raw_terms(fdisc, target['t_pub'], flags)
    xd[['vendor_cve_12m', 'vendor_kev_ever', 'vendor_kev_12m']] = vh_d.values
    nvd = nvd_fields(set(target['cve_id']))
    miss = ~target['cve_id'].isin(nvd.index)
    fcur = nvd.reindex(target['cve_id']).reset_index()
    fcur['cvss_version'] = fcur['cvss_version'].where(fcur['cvss_version'].notna(), None)
    xc = raw_terms(fcur, target['t_pub'], flags)
    xc[['vendor_cve_12m', 'vendor_kev_ever', 'vendor_kev_12m']] = vh_c.values
    for x, name in ((xd, 'disclosure'), (xc, 'current')):
        x.insert(0, 'cve_id', target['cve_id'].values)
        write_parquet(x[['cve_id'] + TERMS], PROC / f'features_raw_{name}_{SNAP}.parquet', 'S8', 'build')
    src_d = fdisc['source'].values
    prov = []
    for t in TERMS:
        g = GROUP[t]
        sd = np.where(g == 'vendor_history', 'vendor_universe(cvelistV5)+kev(t_kev<t_pub)', np.where(g == 'period', 't_pub', src_d))
        prov.append(pd.DataFrame({'cve_id': target['cve_id'], 'feature_name': t, 'feature_set': 'disclosure', 'feature_value': xd[t].astype(str).values, 'source': sd, 'available_at': target['t_pub'].values, 'extraction_rule': f'configs/features.yaml#{t}', 'snapshot_id': SNAP, 'is_available_at_disclosure': True}))
        sc_ = 'vendor_universe(cvelistV5)+kev(t_freeze)' if g == 'vendor_history' else 't_pub' if g == 'period' else 'nvd_current_record'
        prov.append(pd.DataFrame({'cve_id': target['cve_id'], 'feature_name': t, 'feature_set': 'current', 'feature_value': xc[t].astype(str).values, 'source': sc_, 'available_at': T_FREEZE if g != 'period' else target['t_pub'].values, 'extraction_rule': f'configs/features.yaml#{t}', 'snapshot_id': SNAP, 'is_available_at_disclosure': g == 'period'}))
    prov = pd.concat(prov, ignore_index=True)
    prov['available_at'] = pd.to_datetime(prov['available_at']).dt.date
    write_parquet(prov, PROC / f'feature_provenance_{SNAP}.parquet', 'S8', 'build')
    enc_params, summ = ({}, [])
    jobs = [('disclosure', 'B', 'validmodel', 'valid'), ('disclosure', 'B', 'testmodel', 'test'), ('current', 'B', 'validmodel', 'valid'), ('current', 'B', 'testmodel', 'test'), ('disclosure', 'A', 'testmodel', 'test'), ('disclosure', 'B7', 'testmodel', 'test')]
    raw = {'disclosure': xd.set_index('cve_id'), 'current': xc.set_index('cve_id')}
    for fs, s, fit_ctx, apply_ctx in jobs:
        fit_ids = pd.read_parquet(LAB / f'labels_{fit_ctx}_{s}_{SNAP}.parquet', columns=['cve_id'])['cve_id']
        app_ids = pd.read_parquet(LAB / f'labels_{apply_ctx}_{s}_{SNAP}.parquet', columns=['cve_id'])['cve_id']
        enc = Encoder(FCFG['encoding']['cwe_top_group_levels']).fit(raw[fs].loc[fit_ids, TERMS])
        enc_params[f'{fs}_{s}_{fit_ctx}'] = enc.params()
        for ctx, ids in ((fit_ctx, fit_ids), (apply_ctx, app_ids)):
            X = enc.transform(raw[fs].loc[ids, TERMS])
            X.insert(0, 'cve_id', ids.values)
            p = PROC / f'X_{fs}_{ctx}_{s}_{SNAP}.parquet'
            write_parquet(X, p, 'S8', 'build')
            summ.append({'feature_set': fs, 'scenario': s, 'context': ctx, 'encoder_fit': fit_ctx, 'n_rows': len(X), 'n_columns': X.shape[1] - 1, 'sha256': sha256_file(p)})
    (ROOT / 'results' / 'metrics' / f's08_encoders_{SNAP}.json').write_text(json.dumps({'n_terms': len(TERMS), 'terms': TERMS, 'nvd_missing_in_cohort': int(miss.sum()), 'encoders': enc_params}, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    pd.DataFrame(summ).to_csv(ROOT / 'results' / 'tables' / f's08_matrices_{SNAP}.csv', index=False)
    m = coh['in_main_cohort'].values == 1
    rows = []
    for t in TERMS:
        a, b = (xd.loc[m, t], xc.loc[m, t])
        num = pd.api.types.is_numeric_dtype(a)
        rows.append({'term': t, 'group': GROUP[t], 'disclosure_nonmissing': float(a.notna().mean()), 'current_nonmissing': float(b.notna().mean()), 'disclosure_mean': float(a.mean()) if num else None, 'current_mean': float(b.mean()) if num else None, 'changed_share': float((a.astype(str) != b.astype(str)).mean())})
    pd.DataFrame(rows).to_csv(ROOT / 'results' / 'tables' / f's08_feature_summary_{SNAP}.csv', index=False)
    print(pd.DataFrame(summ).to_string())
    print(pd.DataFrame(rows).round(3).to_string())
    print('nvd missing', int(miss.sum()))
if __name__ == '__main__':
    main()
