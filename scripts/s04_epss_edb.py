from __future__ import annotations
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kev_survival.events import scenarios as sc
from kev_survival.features import epss as ep
from kev_survival.features import exploitdb as ed
from kev_survival.io.common import ROOT, load_project, raw_dir, utc_now
from kev_survival.reconstruct import cvelist as cl
from kev_survival.reconstruct import json4 as j4
CFG = load_project()
SNAP = str(CFG['t_freeze']).replace('-', '')
T_FREEZE = dt.date.fromisoformat(str(CFG['t_freeze']))
INTERIM, TABLES, METRICS = (ROOT / 'data' / 'interim', ROOT / 'results' / 'tables', ROOT / 'results' / 'metrics')
PERIODS = {k: tuple((dt.date.fromisoformat(x) for x in v)) for k, v in CFG['periods'].items()}
B_START = dt.date.fromisoformat(CFG['cohort_start'])
PRE_DISCLOSURE_V1 = 'kev_before_cve_record_publication_sources_consistent'
EPSS_MISSING_DAYS = (dt.date(2022, 7, 14), dt.date(2024, 12, 1))
COHORT_OUT = INTERIM / f's04_cohorts_{SNAP}.parquet'
EPSS_OUT = INTERIM / f'epss_first_scores_{SNAP}.parquet'
EDB_OUT = INTERIM / f'exploitdb_matches_{SNAP}.parquet'
EDB_REC_OUT = INTERIM / f'exploitdb_record_matches_{SNAP}.parquet'

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

def load_kev() -> pd.DataFrame:
    k = pd.DataFrame(json.load(open(raw_dir() / 'cisa_kev' / 'known_exploited_vulnerabilities.json', encoding='utf-8'))['vulnerabilities']).rename(columns={'cveID': 'cve_id'})
    k['t_kev'] = pd.to_datetime(k['dateAdded']).dt.date
    return k[['cve_id', 't_kev']]

def build_cohorts() -> pd.DataFrame:
    recs = pd.read_parquet(INTERIM / f's03_scenario_records_{SNAP}.parquet')
    dq = pd.read_parquet(INTERIM / f'date_quality_s03_{SNAP}.parquet')
    out = []
    for s, start in (('B', B_START), ('A', sc.STRICT_START)):
        g = recs[(recs['scenario'] == s) & (recs['t_pub'] >= start)].copy()
        q = dq[dq['scenario'] == s].set_index('cve_id')
        g['date_quality_class'] = g['cve_id'].map(q['negative_cause'].map({PRE_DISCLOSURE_V1: 'pre_disclosure_kev_listing', 'calendar_or_time_precision': 'same_day_tz'}).fillna('unexplained'))
        g['in_cohort'] = ~g['date_quality_class'].isin(['pre_disclosure_kev_listing', 'unexplained'])
        g['exclusion_reason'] = np.where(g['in_cohort'], None, g['date_quality_class'])
        sd = g['date_quality_class'] == 'same_day_tz'
        g.loc[sd, 'duration_days'] = 0
        g.loc[sd, 'zero_day_event'] = 1
        g['period'] = [sc.period_of(d, PERIODS) for d in g['t_pub']]
        g['kev_30d'] = ((g['event_freeze'] == 1) & g['duration_days'].between(0, 30)).astype(int)
        g['kev_365d'] = ((g['event_freeze'] == 1) & g['duration_days'].between(0, 365)).astype(int)
        g['scenario'] = s
        out.append(g)
    c = pd.concat(out).sort_values(['scenario', 'cve_id']).reset_index(drop=True)
    assert (c.loc[c['in_cohort'] & (c['event_freeze'] == 1), 'duration_days'] >= 0).all()
    assert set(c.loc[c['in_cohort'], 'period']) <= set(PERIODS)
    c.to_parquet(COHORT_OUT, index=False)
    flow = c.groupby(['scenario', 'date_quality_class'], dropna=False).size().rename('n').reset_index()
    flow.to_csv(TABLES / f's04_cohort_exclusions_{SNAP}.csv', index=False)
    return c

def cohorts() -> pd.DataFrame:
    return pd.read_parquet(COHORT_OUT) if COHORT_OUT.exists() else build_cohorts()

def smd(a: pd.Series, b: pd.Series, binary: bool) -> float:
    a, b = (a.dropna().astype(float), b.dropna().astype(float))
    if binary:
        p, q = (a.mean(), b.mean())
        den = np.sqrt((p * (1 - p) + q * (1 - q)) / 2)
        return float((p - q) / den) if den > 0 else 0.0
    den = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / den) if den > 0 else 0.0

def json4bias() -> dict:
    recon = pd.read_parquet(INTERIM / f'first_published_records_s02_{SNAP}.parquet')
    jdf = pd.read_parquet(INTERIM / f'json4_public_pre2023-04_s03_{SNAP}.parquet')
    base = sc._base(recon)
    win = base[base['t_pub'].notna() & (base['t_pub'] >= B_START) & (base['t_pub'] < sc.STRICT_START)]
    j = win.merge(jdf, on='cve_id', how='inner')
    lag = (pd.to_datetime(j['json4_public_at'], utc=True) - pd.to_datetime(j['cve_date_published'], utc=True, format='ISO8601')).dt.total_seconds() / 3600
    j['json4_delay_hours'] = lag.values
    j['group'] = np.where(lag.abs() <= sc.H24, 'included_json4_24h', 'excluded_json4_over_24h')
    repo = raw_dir() / 'cvelist_json4'
    paths = cl.head_paths(repo, pathspec='.')
    specs = {f'{cm}:{paths[c]}': c for c, cm in zip(j['cve_id'], j['json4_commit'])}
    det = {}
    for spec, data in cl.cat_file_batch(repo, specs):
        try:
            r = json.loads(data.decode('utf-8', 'replace'), strict=False) if data else {}
        except json.JSONDecodeError:
            r = {}
        det[specs[spec]] = j4.detail_v4(r if isinstance(r, dict) else {})
    j = j.merge(pd.DataFrame.from_dict(det, orient='index').rename_axis('cve_id').reset_index(), on='cve_id')
    kev = load_kev()
    j = j.merge(kev[kev['t_kev'] <= T_FREEZE], on='cve_id', how='left')
    j['kev'] = j['t_kev'].notna().astype(int)
    j['kev_duration_days'] = [(k - p).days if isinstance(k, dt.date) else np.nan for k, p in zip(j['t_kev'], j['t_pub'])]
    j['kev_365d'] = (j['kev'].eq(1) & j['kev_duration_days'].between(0, 365)).astype(int)
    j['pub_month_index'] = [(d.year - 2022) * 12 + d.month for d in j['t_pub']]
    inc, exc = (j[j['group'] == 'included_json4_24h'], j[j['group'] == 'excluded_json4_over_24h'])
    rows = []

    def add(var, kind, a, b, note=''):
        rows.append({'variable': var, 'kind': kind, 'included_n': int(a.notna().sum()), 'excluded_n': int(b.notna().sum()), 'included_value': float(a.mean()), 'excluded_value': float(b.mean()), 'difference': float(a.mean() - b.mean()), 'smd': smd(a, b, kind == 'proportion'), 'note': note})
    add('pub_month_index', 'mean', inc['pub_month_index'], exc['pub_month_index'], '2022-07=7 … 2023-03=15')
    add('v4_desc_len_words', 'mean', inc['v4_desc_len_words'], exc['v4_desc_len_words'])
    add('v4_has_cvss', 'proportion', inc['v4_has_cvss'], exc['v4_has_cvss'])
    add('v4_cvss_base_score', 'mean', inc['v4_cvss_base_score'], exc['v4_cvss_base_score'], 'CVSS bulunanlar')
    add('v4_has_cwe', 'proportion', inc['v4_has_cwe'], exc['v4_has_cwe'])
    add('v4_has_vendor', 'proportion', inc['v4_has_vendor'], exc['v4_has_vendor'])
    add('v4_has_product', 'proportion', inc['v4_has_product'], exc['v4_has_product'])
    add('v4_n_refs', 'mean', inc['v4_n_refs'], exc['v4_n_refs'])
    add('kev_ever_by_freeze', 'proportion', inc['kev'], exc['kev'])
    add('kev_within_365d', 'proportion', inc['kev_365d'], exc['kev_365d'])
    add('kev_duration_days', 'mean', inc['kev_duration_days'], exc['kev_duration_days'], 'yalnızca KEV olayları')
    top = j['disc_assigner_short_name'].value_counts().head(10).index
    cna = j['disc_assigner_short_name'].where(j['disc_assigner_short_name'].isin(top), '_other')
    for k in list(top) + ['_other']:
        a, b = (cna[inc.index] == k, cna[exc.index] == k)
        add(f'cna_share:{k}', 'proportion', a, b)
    for y in sorted(j['t_pub'].map(lambda d: f'{d.year}-{d.month:02d}').unique()):
        ym = j['t_pub'].map(lambda d: f'{d.year}-{d.month:02d}')
        add(f'pub_month_share:{y}', 'proportion', ym[inc.index] == y, ym[exc.index] == y)
    tab = pd.DataFrame(rows)
    tab['abs_smd'] = tab['smd'].abs()
    tab.to_csv(TABLES / f's04_json4_selection_bias_{SNAP}.csv', index=False)
    n_win, n_j = (len(win), len(j))
    key = tab[~tab['variable'].str.startswith(('cna_share', 'pub_month_share'))]
    summary = {'window': [str(B_START), '2023-03-31'], 'n_cve_window_v5_published': n_win, 'n_with_json4_public': n_j, 'n_without_json4_public': n_win - n_j, 'n_included_24h': len(inc), 'n_excluded_over_24h': len(exc), 'excluded_share_of_json4': len(exc) / n_j, 'excluded_delay_hours_quantiles': exc['json4_delay_hours'].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).to_dict(), 'excluded_json4_after_v5_share': float((exc['json4_delay_hours'] > 0).mean()), 'kev_events': {'included': int(inc['kev'].sum()), 'excluded': int(exc['kev'].sum())}, 'kev_rate': {'included': float(inc['kev'].mean()), 'excluded': float(exc['kev'].mean())}, 'kev_rate_difference': float(inc['kev'].mean() - exc['kev'].mean()), 'kev_rate_ratio_excluded_over_included': float(exc['kev'].mean() / inc['kev'].mean()), 'max_abs_smd_key_variables': float(key['abs_smd'].max()), 'n_key_vars_abs_smd_ge_0_1': int((key['abs_smd'] >= 0.1).sum()), 'n_key_vars_abs_smd_ge_0_2': int((key['abs_smd'] >= 0.2).sum()), 'max_abs_smd_cna_shares': float(tab[tab['variable'].str.startswith('cna_share')]['abs_smd'].max()), 'max_abs_smd_month_shares': float(tab[tab['variable'].str.startswith('pub_month_share')]['abs_smd'].max()), 'global_test': 'uygulanmadı (talimat: en fazla bir; p değeri karar ölçütü değil)', 'excluded_top_cna': exc['disc_assigner_short_name'].value_counts().head(10).to_dict(), 'included_top_cna': inc['disc_assigner_short_name'].value_counts().head(10).to_dict()}
    write_json(METRICS / f's04_json4_selection_bias_{SNAP}.json', summary)
    j[['cve_id', 't_pub', 'group', 'json4_delay_hours', 'disc_assigner_short_name', 'v4_desc_len_words', 'v4_cvss_base_score', 'v4_n_refs', 'v4_has_cvss', 'v4_has_cwe', 'v4_has_vendor', 'v4_has_product', 'kev', 'kev_duration_days']].to_parquet(INTERIM / f's04_json4_bias_records_{SNAP}.parquet', index=False)
    return summary

def epss_run() -> dict:
    c = cohorts()
    per = c[['cve_id', 't_pub']].drop_duplicates('cve_id')
    files = sorted((raw_dir() / 'epss').glob('epss_scores-*.csv.gz'))
    raw_first = INTERIM / f's04_epss_first_raw_{SNAP}.parquet'
    if raw_first.exists() and (TABLES / f's04_epss_files_{SNAP}.csv').exists():
        e = pd.read_parquet(raw_first)
        days = pd.read_csv(TABLES / f's04_epss_files_{SNAP}.csv')
    else:
        e, days = ep.first_scores(files, set(per['cve_id']))
        e.to_parquet(raw_first, index=False)
        days.to_csv(TABLES / f's04_epss_files_{SNAP}.csv', index=False)
    out = ep.attach(per, e)
    out.to_parquet(EPSS_OUT, index=False)
    cc = c.merge(out.drop(columns='t_pub'), on='cve_id', how='left')
    cc = cc[cc['in_cohort']]
    ev = cc['event_freeze'] == 1
    rows = []
    for s, g in cc.groupby('scenario'):
        groups = {'all': g, **{f'period:{p}': x for p, x in g.groupby('period')}, 'kev:1': g[g['event_freeze'] == 1], 'kev:0': g[g['event_freeze'] == 0]}
        groups |= {f'epss_version_at_first:{v}': x for v, x in g.groupby(g['epss_model_version'].fillna('none'))}
        for name, x in groups.items():
            for t in ep.TOLERANCES:
                k = x[f'epss_available_within_{t}d']
                rows.append({'scenario': s, 'group': name, 'tolerance_days': t, 'n_cve': len(x), 'n_available': int(k.sum()), 'coverage': float(k.mean()) if len(x) else np.nan})
    cov = pd.DataFrame(rows)
    cov.to_csv(TABLES / f's04_epss_coverage_{SNAP}.csv', index=False)
    sub = []
    for s, g in cc.groupby('scenario'):
        for t in ep.TOLERANCES:
            m = g[f'epss_available_within_{t}d']
            x, gev = (g[m], g['event_freeze'] == 1)
            row = {'scenario': s, 'tolerance_days': t, 'n_cve': int(m.sum()), 'n_excluded': int((~m).sum()), 'kev_events': int(x['event_freeze'].sum()), 'kev_events_excluded': int((gev & ~m).sum()), 'kev_30d_events': int(x['kev_30d'].sum())}
            for p in PERIODS:
                xp = x[x['period'] == p]
                row[f'{p}_n_cve'] = len(xp)
                row[f'{p}_kev_events'] = int(xp['event_freeze'].sum())
                row[f'{p}_kev_30d_events'] = int(xp['kev_30d'].sum())
            row['test_n_excluded'] = int(((g['period'] == 'test') & ~m).sum())
            row['test_kev_events_excluded'] = int(((g['period'] == 'test') & ~m & gev).sum())
            row['kev_events_epss_first_after_t_kev'] = int((x['event_freeze'].eq(1) & (pd.to_datetime(x['first_epss_available_at']) > pd.to_datetime(x['t_kev']))).sum())
            sub.append(row)
    sub = pd.DataFrame(sub)
    sub.to_csv(TABLES / f's04_epss_common_subcohort_{SNAP}.csv', index=False)
    miss = {}
    for d0 in EPSS_MISSING_DAYS:
        nxt = d0 + dt.timedelta(days=1)
        for s, g in cc.groupby('scenario'):
            first_next = g['first_epss_available_at'] == nxt
            pub_on = g['t_pub'] == d0
            win = {t: int(((g['t_pub'] <= d0) & (g['t_pub'] >= d0 - dt.timedelta(days=t)) & (g['first_epss_available_at'] > d0) & ~g[f'epss_available_within_{t}d']).sum()) for t in ep.TOLERANCES}
            miss[f'{d0}:{s}'] = {'cve_published_on_missing_day': int(pub_on.sum()), 'cve_first_epss_on_next_day': int(first_next.sum()), 'cve_t_pub_window_touches_missing_day_and_not_covered': win, 'of_which_test': int((first_next & (g['period'] == 'test')).sum()), 'kev_events_first_epss_on_next_day': int((first_next & (g['event_freeze'] == 1)).sum())}
    d = cc['epss_delay_days'].dropna().astype(int)
    summary = {'n_epss_files': len(files), 'missing_days': [str(x) for x in EPSS_MISSING_DAYS], 'model_versions_by_file': days.groupby('model_version_raw', dropna=False)['day'].agg(['min', 'max', 'count']).astype(str).to_dict(orient='index'), 'first_version_counts': {f'{a}:{b}': int(n) for (a, b), n in cc.groupby(['scenario', cc['epss_model_version'].fillna('none')]).size().items()}, 'n_with_any_epss': cc.groupby('scenario')['first_epss_available_at'].apply(lambda x: int(x.notna().sum())).to_dict(), 'n_without_epss_in_archive': cc.groupby('scenario')['first_epss_available_at'].apply(lambda x: int(x.isna().sum())).to_dict(), 'n_negative_delay': cc.groupby('scenario')['epss_delay_days'].apply(lambda x: int((x.dropna() < 0).sum())).to_dict(), 'delay_quantiles_all': d.quantile([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict(), 'missing_day_impact': miss, 'outputs_sha256': {p.name: sha(p) for p in (EPSS_OUT,)}}
    write_json(METRICS / f's04_epss_coverage_{SNAP}.json', summary)
    return summary

def edb_source() -> Path:
    files = sorted((raw_dir() / 'exploitdb').glob('files_exploits_*.csv'))
    assert len(files) == 1, files
    return files[0]

def edb_build(c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta = ed.read_metadata(edb_source())
    rec = ed.record_matches(meta)
    partial = [t for codes in meta['codes'] for t in ed.cve_tokens(codes)[1]]
    per = c[c['in_cohort']][['cve_id', 't_pub']].drop_duplicates('cve_id')
    out = pd.concat([ed.per_cve(per, rec, d) for d in ed.DATE_DEFINITIONS]).reset_index(drop=True)
    info = {'n_metadata_records': len(meta), 'n_records_with_exact_cve': int(rec['edb_id'].nunique()), 'n_record_cve_pairs': len(rec), 'n_distinct_cve_all_edb': int(rec['cve_id'].nunique()), 'n_partial_cve_like_tokens_rejected': len(partial), 'partial_token_examples': partial[:10], 'n_records_aliases_with_cve': int(meta['aliases'].str.contains('CVE').sum())}
    return (rec, out, info)

def edb_run() -> dict:
    c = cohorts()
    rec, out, info = edb_build(c)
    rec.to_parquet(EDB_REC_OUT, index=False)
    out.to_parquet(EDB_OUT, index=False)
    cc = c[c['in_cohort']].merge(out, on=['cve_id', 't_pub'], how='left')
    rows = []

    def put(s, d, item, group, n, den=None):
        rows.append({'scenario': s, 'date_definition': d, 'item': item, 'group': group, 'n': int(n), 'denominator': None if den is None else int(den), 'rate': None if not den else n / den})
    for (s, d), g in cc.groupby(['scenario', 'date_definition']):
        m = g['n_exploitdb_records'] > 0
        f = g['disclosure_time_exploitdb_poc'] == 1
        put(s, d, 1, 'all', len(g))
        put(s, d, 2, 'all', m.sum(), len(g))
        for y, x in g.groupby(g['t_pub'].map(lambda v: v.year)):
            put(s, d, 3, f'year:{y}', (x['n_exploitdb_records'] > 0).sum(), len(x))
        for p, x in g.groupby('period'):
            put(s, d, 4, f'period:{p}', (x['n_exploitdb_records'] > 0).sum(), len(x))
            put(s, d, 7, f'period:{p}', (x['disclosure_time_exploitdb_poc'] == 1).sum(), len(x))
            put(s, d, 8, f'period:{p}', ((x['disclosure_time_exploitdb_poc'] == 1) & (x['event_freeze'] == 1)).sum(), (x['disclosure_time_exploitdb_poc'] == 1).sum())
        for k, x in g.groupby('event_freeze'):
            put(s, d, 5, f'kev:{k}', (x['n_exploitdb_records'] > 0).sum(), len(x))
        put(s, d, 6, 'all', g['exploitdb_delay_days'].le(0).fillna(False).sum(), len(g))
        put(s, d, 7, 'all', f.sum(), len(g))
        put(s, d, 8, 'all', (f & (g['event_freeze'] == 1)).sum(), f.sum())
        put(s, d, 8, 'kev_30d', (f & (g['kev_30d'] == 1)).sum(), f.sum())
        put(s, d, 8, 'kev_rate_if_flag0', (~f & (g['event_freeze'] == 1)).sum(), (~f).sum())
        dl = g.loc[m, 'exploitdb_delay_days'].astype(float)
        bins = [-np.inf, -366, -1, 0, 7, 30, 365, np.inf]
        labels = ['<=-366', '-365..-1', '0', '1..7', '8..30', '31..365', '>365']
        for lab, n in pd.cut(dl, bins, labels=labels).value_counts().reindex(labels).items():
            put(s, d, 9, f'delay:{lab}', n, m.sum())
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            rows.append({'scenario': s, 'date_definition': d, 'item': 9, 'group': f'delay_q{int(q * 100)}', 'n': None, 'denominator': int(m.sum()), 'rate': float(dl.quantile(q)) if len(dl) else None})
        put(s, d, 10, 'n_records>1', (g['n_exploitdb_records'] > 1).sum(), m.sum())
        put(s, d, 11, 'cve_with_record_missing_date', (g['n_records_missing_date'] > 0).sum(), m.sum())
        put(s, d, 11, 'cve_with_date_conflict', g['has_date_conflict'].fillna(False).sum(), m.sum())
    w = cc.pivot_table(index=['scenario', 'cve_id'], columns='date_definition', values='disclosure_time_exploitdb_poc').reset_index()
    for s, g in w.groupby('scenario'):
        diff = g['date_added'] != g['date_published']
        put(s, 'both', 12, 'flag_differs_between_definitions', diff.sum(), len(g))
        put(s, 'both', 12, 'flag1_added_only', ((g['date_added'] == 1) & (g['date_published'] == 0)).sum(), len(g))
        put(s, 'both', 12, 'flag1_published_only', ((g['date_added'] == 0) & (g['date_published'] == 1)).sum(), len(g))
    cov = pd.DataFrame(rows)
    cov_path = TABLES / f's04_exploitdb_coverage_{SNAP}.csv'
    cov.to_csv(cov_path, index=False)
    rc = rec[rec['cve_id'].isin(set(cc['cve_id']))]
    da = pd.to_datetime(rc['date_added']) - pd.to_datetime(rc['date_published'])
    dates = {'cohort_matched_records': int(rc['edb_id'].nunique()), 'added_minus_published_days_cohort': da.dt.days.describe().to_dict(), 'added_eq_published_share_cohort': float((da.dt.days == 0).mean()), 'added_before_published_cohort': int((da.dt.days < 0).sum()), 'added_after_published_cohort': int((da.dt.days > 0).sum()), 'missing_date_added_cohort': int(rc['date_added'].isna().sum()), 'missing_date_published_cohort': int(rc['date_published'].isna().sum()), 'missing_date_updated_cohort': int(rc['date_updated'].isna().sum()), 'updated_before_added_cohort': int(rc['conflict_updated_before_added'].sum()), 'all_records_added_before_published': int(rec.drop_duplicates('edb_id')['conflict_added_before_published'].sum())}
    summary = {'source_file': edb_source().name, 'source_sha256': sha(edb_source()), **info, 'dates': dates, 'outputs_sha256': {p.name: sha(p) for p in (EDB_OUT, EDB_REC_OUT, cov_path)}}
    write_json(METRICS / f's04_exploitdb_coverage_{SNAP}.json', summary)
    return summary

def feasibility_table() -> None:
    sub = pd.read_csv(TABLES / f's04_epss_common_subcohort_{SNAP}.csv')
    cov = pd.read_csv(TABLES / f's04_exploitdb_coverage_{SNAP}.csv')
    rows = []
    for r in sub.itertuples():
        rows.append({'source': 'epss', 'scenario': r.scenario, 'definition': f'tolerance_{r.tolerance_days}d', 'metric': 'common_test_n_cve', 'value': r.test_n_cve})
        rows.append({'source': 'epss', 'scenario': r.scenario, 'definition': f'tolerance_{r.tolerance_days}d', 'metric': 'common_test_kev_events', 'value': r.test_kev_events})
        rows.append({'source': 'epss', 'scenario': r.scenario, 'definition': f'tolerance_{r.tolerance_days}d', 'metric': 'common_test_kev_30d_events', 'value': r.test_kev_30d_events})
    for r in cov[cov['item'].isin([2, 7]) & cov['group'].isin(['all', 'period:test'])].itertuples():
        rows.append({'source': 'exploitdb', 'scenario': r.scenario, 'definition': r.date_definition, 'metric': f'item{r.item}_{r.group}_n', 'value': r.n})
    pd.DataFrame(rows).to_csv(TABLES / 'feasibility_epss_edb.csv', index=False)

def epss_fair() -> pd.DataFrame:
    c = cohorts()
    e = pd.read_parquet(EPSS_OUT).drop(columns='t_pub')
    cc = c[c['in_cohort']].merge(e, on='cve_id', how='left')
    fe = pd.to_datetime(cc['first_epss_available_at'])
    tk = pd.to_datetime(cc['t_kev'])
    t_e = pd.concat([fe, pd.to_datetime(cc['t_pub'])], axis=1).max(axis=1)
    ev = cc['event_freeze'] == 1
    cc['epss_before_kev'] = ~ev | (fe < tk)
    cc['kev_30d_from_t_e'] = ev & (tk > t_e) & (tk <= t_e + pd.Timedelta(days=30))
    rows = []
    for s, g in cc.groupby('scenario'):
        for t in (1, 2, 7):
            m = g[f'epss_available_within_{t}d']
            keep = m & g['epss_before_kev']
            drop = m & ~g['epss_before_kev']
            gev = g['event_freeze'] == 1
            row = {'scenario': s, 'tolerance_days': t, 'role': 'main' if t == 1 else 'robustness', 'n_cve': int(keep.sum()), 'kev_events': int((keep & gev).sum()), 'kev_30d_from_t_e': int((keep & g['kev_30d_from_t_e']).sum()), 'excluded_epss_not_before_kev': int(drop.sum()), 'excluded_same_day': int((drop & (pd.to_datetime(g['first_epss_available_at']) == pd.to_datetime(g['t_kev']))).sum()), 'excluded_all_zero_day': bool((g.loc[drop, 'duration_days'] == 0).all())}
            for p in PERIODS:
                pp = g['period'] == p
                row[f'{p}_n_cve'] = int((keep & pp).sum())
                row[f'{p}_kev_events'] = int((keep & pp & gev).sum())
                row[f'{p}_kev_30d_from_t_e'] = int((keep & pp & g['kev_30d_from_t_e']).sum())
                row[f'{p}_excluded_epss_not_before_kev'] = int((drop & pp).sum())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / f's04_epss_fair_subcohort_{SNAP}.csv', index=False)
    return out

def predisclosure() -> pd.DataFrame:
    c = cohorts()
    pre = c[c['date_quality_class'] == 'pre_disclosure_kev_listing']
    recon = pd.read_parquet(INTERIM / f'first_published_records_s02_{SNAP}.parquet', columns=['cve_id', 'disc_assigner_short_name', 'disc_has_cvss_any', 'disc_has_cwe'])
    j4df = pd.read_parquet(INTERIM / f'json4_public_pre2023-04_s03_{SNAP}.parquet', columns=['cve_id', 'v4_has_cvss', 'v4_has_cwe'])
    kev = pd.DataFrame(json.load(open(raw_dir() / 'cisa_kev' / 'known_exploited_vulnerabilities.json', encoding='utf-8'))['vulnerabilities']).rename(columns={'cveID': 'cve_id'})
    x = pre.merge(recon, on='cve_id', how='left').merge(j4df, on='cve_id', how='left').merge(kev[['cve_id', 'vendorProject', 'knownRansomwareCampaignUse']], on='cve_id', how='left')
    j = x['t_pub_source'].str.startswith('json4')
    x['has_cvss_at_disclosure'] = np.where(j, x['v4_has_cvss'], x['disc_has_cvss_any']).astype(bool)
    x['has_cwe_at_disclosure'] = np.where(j, x['v4_has_cwe'], x['disc_has_cwe']).astype(bool)
    x[['scenario', 'cve_id', 't_pub', 't_kev', 'duration_days', 't_pub_source', 'period', 'disc_assigner_short_name', 'vendorProject', 'has_cvss_at_disclosure', 'has_cwe_at_disclosure', 'knownRansomwareCampaignUse']].to_csv(TABLES / f's04_pre_disclosure_kev_records_{SNAP}.csv', index=False)
    summ = {}
    for s, g in x.groupby('scenario'):
        summ[s] = {'n': len(g), 'by_period': g['period'].value_counts().to_dict(), 'duration_days_quantiles': g['duration_days'].quantile([0, 0.25, 0.5, 0.75, 1]).to_dict(), 'kev_vendor_project_top': g['vendorProject'].value_counts().head(8).to_dict(), 'cna_top': g['disc_assigner_short_name'].value_counts().head(8).to_dict(), 'has_cvss_at_disclosure': float(g['has_cvss_at_disclosure'].mean()), 'has_cwe_at_disclosure': float(g['has_cwe_at_disclosure'].mean()), 'ransomware_known': int((g['knownRansomwareCampaignUse'] == 'Known').sum()), 'share_of_all_kev_events': len(g) / (len(g) + int(c[(c['scenario'] == s) & c['in_cohort']]['event_freeze'].sum()))}
    write_json(METRICS / f's04_pre_disclosure_kev_summary_{SNAP}.json', summ)
    return x

def main(cmd: str) -> None:
    t0 = utc_now()
    if cmd in ('cohort', 'all'):
        c = build_cohorts()
        print(c.groupby(['scenario', 'in_cohort']).size())
    if cmd in ('json4bias', 'all'):
        print(json.dumps(json4bias(), indent=1, default=str, ensure_ascii=False)[:3000])
    if cmd in ('epss', 'all'):
        print(json.dumps(epss_run(), indent=1, default=str)[:3000])
    if cmd in ('edb', 'all'):
        print(json.dumps(edb_run(), indent=1, default=str)[:3000])
    if cmd in ('final', 'all'):
        print(epss_fair().T.to_string())
        print(json.dumps(json.load(open(METRICS / f's04_pre_disclosure_kev_summary_{SNAP}.json', encoding='utf-8')) if predisclosure() is not None else {}, indent=1, ensure_ascii=False))
    if cmd in ('table', 'epss', 'edb', 'all') and (TABLES / f's04_epss_common_subcohort_{SNAP}.csv').exists() and (TABLES / f's04_exploitdb_coverage_{SNAP}.csv').exists():
        feasibility_table()
    print('başlangıç', t0, 'bitiş', utc_now())
if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
