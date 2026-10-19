from __future__ import annotations
import datetime as dt
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kev_survival.events import scenarios as sc
from kev_survival.features import fields as fd
from kev_survival.io.access import write_parquet
from kev_survival.io.common import ROOT, load_project, raw_dir, sha256_file
from kev_survival.reconstruct import cvelist as cl
from kev_survival.split.labels import at_cutoff
CFG = load_project()
assert CFG['locked'] is True, 'S6 yalnız kilitli yapılandırmayla çalışır'
SNAP = str(CFG['t_freeze']).replace('-', '')
T_FREEZE = dt.date.fromisoformat(str(CFG['t_freeze']))
P = {k: tuple((dt.date.fromisoformat(x) for x in v)) for k, v in CFG['periods'].items()}
START = dt.date.fromisoformat(CFG['cohort_start'])
INTERIM, PROC = (ROOT / 'data' / 'interim', ROOT / 'data' / 'processed')
TABLES, METRICS = (ROOT / 'results' / 'tables', ROOT / 'results' / 'metrics')
COHORT = PROC / f'cohort_{SNAP}.parquet'
FIELDS = INTERIM / f'disclosure_fields_s06_{SNAP}.parquet'
V1_TO_V2 = {'kev_before_cve_record_publication_sources_consistent': 'pre_disclosure_kev_listing', 'calendar_or_time_precision': 'same_day_tz'}

def build_cohort() -> pd.DataFrame:
    r = pd.read_parquet(INTERIM / f's03_scenario_records_{SNAP}.parquet')
    dq = pd.read_parquet(INTERIM / f'date_quality_s03_{SNAP}.parquet')
    sets = {s: r[(r['scenario'] == s) & (r['t_pub'] >= START)] for s in ('A', 'B', 'B7')}
    ids = {s: set(g['cve_id']) for s, g in sets.items()}
    assert ids['A'] <= ids['B'] <= ids['B7'], 'senaryo kapsama ilişkisi bozuk'
    u = sets['B7'].copy()
    b = sets['B'].set_index('cve_id')
    common = u['cve_id'].isin(ids['B'])
    assert (u.loc[common].set_index('cve_id')['t_pub'] == b.loc[u.loc[common, 'cve_id'], 't_pub']).all()
    cls = dq[dq['scenario'] == 'B7'].set_index('cve_id')['negative_cause'].map(V1_TO_V2).fillna('unexplained')
    u['date_quality_class'] = u['cve_id'].map(cls)
    u['in_scenario_B'] = u['cve_id'].isin(ids['B']).astype(int)
    u['in_scenario_A'] = u['cve_id'].isin(ids['A']).astype(int)
    u['in_scenario_B7'] = 1
    excl = u['date_quality_class'].isin(['pre_disclosure_kev_listing', 'unexplained'])
    u['exclusion_reason'] = np.select([excl, u['in_scenario_B'].eq(0)], [u['date_quality_class'], 'json4_fidelity_gt24h_le7d_B7_only'], default=None)
    u['exclusion_reason'] = u['exclusion_reason'].replace({'0': None})
    u['in_main_cohort'] = ((u['in_scenario_B'] == 1) & ~excl).astype(int)
    lab = at_cutoff(u.loc[~excl, ['cve_id', 't_pub', 't_kev', 'date_quality_class']], T_FREEZE).set_index('cve_id')
    u['event_freeze'] = u['t_kev'].notna().astype(int)
    u['duration_days_freeze'] = u['duration_days']
    u.loc[~excl, 'duration_days_freeze'] = u.loc[~excl, 'cve_id'].map(lab['duration_days']).values
    u.loc[~excl, 'event_freeze'] = u.loc[~excl, 'cve_id'].map(lab['event']).values
    u['zero_day_event'] = ((u['event_freeze'] == 1) & (u['duration_days_freeze'] == 0)).astype(int)
    u['period'] = [sc.period_of(d, P) for d in u['t_pub']]
    nvd = pd.read_parquet(INTERIM / f'nvd_current_s02_{SNAP}.parquet', columns=['cve_id', 'nvd_published_raw'])
    u = u.merge(nvd, on='cve_id', how='left')
    u['t_pub_nvd'] = pd.to_datetime(u['nvd_published_raw'], errors='coerce').dt.date
    u['rejected_at_freeze'] = 0
    return u

def read_fields(u: pd.DataFrame) -> pd.DataFrame:
    recon = pd.read_parquet(INTERIM / f'first_published_records_s02_{SNAP}.parquet', columns=['cve_id', 'first_published_commit']).set_index('cve_id')
    j4 = pd.read_parquet(INTERIM / f'json4_public_pre2023-04_s03_{SNAP}.parquet', columns=['cve_id', 'json4_commit']).set_index('cve_id')
    rows = []
    jj = u[u['t_pub_source'].str.startswith('json4')]
    repo4 = raw_dir() / 'cvelist_json4'
    p4 = cl.head_paths(repo4, pathspec='.')
    specs = {f"{j4.loc[c, 'json4_commit']}:{p4[c]}": c for c in jj['cve_id']}
    for spec, data in cl.cat_file_batch(repo4, specs):
        rec = json.loads(data.decode('utf-8', 'replace'), strict=False)
        rows.append({'cve_id': specs[spec], 'source': 'json4_first_public', 'source_commit': spec.split(':')[0], **fd.parse_v4(rec)})
    vv = u[~u['t_pub_source'].str.startswith('json4')]
    repo5 = raw_dir() / 'cvelistV5'
    p5 = cl.head_paths(repo5)
    specs = {f"{recon.loc[c, 'first_published_commit']}:{p5[c]}": c for c in vv['cve_id']}
    for spec, data in cl.cat_file_batch(repo5, specs):
        rows.append({'cve_id': specs[spec], 'source': 'cvelistV5_first_published', 'source_commit': spec.split(':')[0], **fd.parse_v5_cna(cl.load_record(data))})
    f = pd.DataFrame(rows).sort_values('cve_id').reset_index(drop=True)
    assert len(f) == len(u) and f['cve_id'].is_unique
    return f

def main() -> None:
    u = build_cohort()
    f = read_fields(u)
    write_parquet(f, FIELDS, 'S6', 'build')
    u = u.merge(f[['cve_id', 'vendor_key', 'source']].rename(columns={'source': 'disclosure_record_source'}), on='cve_id', how='left')
    cols = ['cve_id', 't_pub', 't_pub_nvd', 't_kev', 'duration_days_freeze', 'event_freeze', 'zero_day_event', 'date_quality_class', 'in_main_cohort', 'exclusion_reason', 'period', 'vendor_key', 'rejected_at_freeze', 'in_scenario_B', 'in_scenario_A', 'in_scenario_B7', 't_pub_source', 'disclosure_record_source', 'disclosure_fidelity']
    u = u[cols].sort_values('cve_id').reset_index(drop=True)
    write_parquet(u, COHORT, 'S6', 'build')
    recon = pd.read_parquet(INTERIM / f'first_published_records_s02_{SNAP}.parquet')
    inwin = recon['t_pub'].notna() & (recon['t_pub'] >= START) & (recon['t_pub'] <= sc.END)
    curwin = pd.to_datetime(recon['current_date_published'], errors='coerce', utc=True).dt.date
    flow = [('S2 yeniden kurma kapsamı (güncel datePublished 2021-10 – 2025-01)', len(recon), None), ('t_pub 2022-07-01 – 2024-12-31 (ilk PUBLISHED kayıttan)', int(inwin.sum()), None), ('  güncel durumu REJECTED (dışlandı)', int((inwin & (recon['current_state'] != 'PUBLISHED')).sum()), None), ('  datePublished eksik MITRE (dışlandı; ilk kayıtta t_pub yok, güncel tarih penceresinde)', int(((recon['disclosure_fidelity'] == 'missing_date_published') & curwin.between(START, sc.END)).sum()), None), ('B7 evreni (24 sa veya JSON 4 ≤7 gün)', len(u), int(u['event_freeze'].sum())), ('  yalnız B7 (JSON 4 >24 sa, ≤7 gün)', int((u['in_scenario_B'] == 0).sum()), int(u.loc[u['in_scenario_B'] == 0, 'event_freeze'].sum())), ('Senaryo B (24 sa)', int(u['in_scenario_B'].sum()), int(u.loc[u['in_scenario_B'] == 1, 'event_freeze'].sum())), ('  pre_disclosure_kev_listing (dışlandı)', int(((u['in_scenario_B'] == 1) & (u['date_quality_class'] == 'pre_disclosure_kev_listing')).sum()), None), ('  unexplained (dışlandı)', int(((u['in_scenario_B'] == 1) & (u['date_quality_class'] == 'unexplained')).sum()), None), ('Ana kohort (B)', int(u['in_main_cohort'].sum()), int(u.loc[u['in_main_cohort'] == 1, 'event_freeze'].sum()))]
    m = u[u['in_main_cohort'] == 1]
    for p in ('train', 'valid', 'test'):
        x = m[m['period'] == p]
        flow.append((f'  ana kohort {p}', len(x), int(x['event_freeze'].sum())))
    a = u[(u['in_scenario_A'] == 1) & u['exclusion_reason'].isna()]
    b7 = u[(u['in_scenario_B7'] == 1) & ~u['date_quality_class'].isin(['pre_disclosure_kev_listing', 'unexplained'])]
    flow += [('Senaryo A kohortu', len(a), int(a['event_freeze'].sum())), ('Senaryo B7 kohortu', len(b7), int(b7['event_freeze'].sum()))]
    pd.DataFrame(flow, columns=['step', 'n_cve', 'kev_events_freeze']).to_csv(TABLES / f's06_cohort_flow_{SNAP}.csv', index=False)
    ctx = {'validmodel_train': (['train'], P['valid'][0]), 'testmodel_train_valid': (['train', 'valid'], P['test'][0]), 'valid_eval': (['valid'], P['test'][0]), 'test_eval': (['test'], T_FREEZE)}
    rows = []
    for s, sel in (('B', m), ('A', a), ('B7', b7)):
        for name, (per, cut) in ctx.items():
            x = sel[sel['period'].isin(per)]
            lab = at_cutoff(x[['cve_id', 't_pub', 't_kev', 'date_quality_class']], cut)
            post = int((x['t_kev'].notna() & (pd.to_datetime(x['t_kev']) > pd.Timestamp(cut))).sum())
            rows.append({'scenario': s, 'context': name, 't_cutoff': str(cut), 'n_cve': len(lab), 'events': int(lab['event'].sum()), 'events_30d': int(((lab['event'] == 1) & (lab['duration_days'] <= 30)).sum()), 'zero_day': int(((lab['event'] == 1) & (lab['duration_days'] == 0)).sum()), 'kev_after_cutoff_censored': post})
    lt = pd.DataFrame(rows)
    lt.to_csv(TABLES / f's06_labels_at_cutoffs_{SNAP}.csv', index=False)
    summ = {'cohort_file': str(COHORT.relative_to(ROOT)), 'cohort_sha256': sha256_file(COHORT), 'fields_sha256': sha256_file(FIELDS), 'n_rows': len(u), 'n_main': int(u['in_main_cohort'].sum()), 'fields_source_counts': f['source'].value_counts().to_dict(), 'cvss_version_counts_main': f[f['cve_id'].isin(m['cve_id'])]['cvss_version'].fillna('missing').value_counts().to_dict()}
    (METRICS / f's06_summary_{SNAP}.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding='utf-8')
    print(pd.DataFrame(flow).to_string())
    print(lt.to_string())
    print(summ)
if __name__ == '__main__':
    main()
