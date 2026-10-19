from __future__ import annotations
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kev_survival.events import scenarios as sc
from kev_survival.io.common import ROOT, load_project, raw_dir
from kev_survival.reconstruct import cvelist as cl
from kev_survival.reconstruct import json4 as j4
CFG = load_project()
SNAP = str(CFG['t_freeze']).replace('-', '')
T_FREEZE = dt.date.fromisoformat(str(CFG['t_freeze']))
INTERIM, TABLES, FIGS, METRICS = (ROOT / 'data' / 'interim', ROOT / 'results' / 'tables', ROOT / 'results' / 'figures', ROOT / 'results' / 'metrics')
PERIODS = {k: tuple((dt.date.fromisoformat(x) for x in v)) for k, v in CFG['periods'].items()}
RECON = INTERIM / f'first_published_records_s02_{SNAP}.parquet'
J4OUT = INTERIM / f'json4_public_pre2023-04_s03_{SNAP}.parquet'
SCEN_OUT = INTERIM / f's03_scenario_records_{SNAP}.parquet'
OLD_DAYS = 365
TRAIN_STARTS = [dt.date(2021, 11, 1), dt.date(2022, 1, 1), dt.date(2022, 4, 1), dt.date(2022, 7, 1), dt.date(2023, 4, 1)]
ALT_SPLIT = {'train': (dt.date(2023, 4, 1), dt.date(2023, 12, 31)), 'valid': (dt.date(2024, 1, 1), dt.date(2024, 6, 30)), 'test': (dt.date(2024, 7, 1), dt.date(2024, 12, 31))}

def load_kev() -> pd.DataFrame:
    k = pd.DataFrame(json.load(open(raw_dir() / 'cisa_kev' / 'known_exploited_vulnerabilities.json', encoding='utf-8'))['vulnerabilities'])
    k = k.rename(columns={'cveID': 'cve_id'})
    k['t_kev'] = pd.to_datetime(k['dateAdded']).dt.date
    return k[['cve_id', 't_kev']].sort_values('cve_id').reset_index(drop=True)

def json4() -> None:
    recon = pd.read_parquet(RECON)
    pre = recon[recon['t_pub'].notna() & (recon['t_pub'] >= sc.EARLIEST) & (recon['t_pub'] < sc.STRICT_START) & (recon['current_state'] == 'PUBLISHED')]
    found, bad = j4.first_public(raw_dir() / 'cvelist_json4', set(pre['cve_id']))
    df = pd.DataFrame([{'cve_id': c, **v} for c, v in found.items()]).sort_values('cve_id')
    df.to_parquet(J4OUT, index=False)
    v5repo = raw_dir() / 'cvelistV5'
    paths = cl.head_paths(v5repo)
    pr = pre.set_index('cve_id').loc[df['cve_id']]
    v5rows = {}
    for spec, data in cl.cat_file_batch(v5repo, (f'{c}:{paths[i]}' for i, c in zip(pr.index, pr['first_published_commit']))):
        v5rows[cl.CVE_PATH.search(spec).group(1)] = j4.fields_v5_na_excluded(cl.load_record(data))
    v5 = pd.DataFrame.from_dict(v5rows, orient='index').loc[df['cve_id']]
    m = df.merge(pre[['cve_id', 'cve_date_published', 'disclosure_fidelity']], on='cve_id')
    lag = (pd.to_datetime(m['json4_public_at'], utc=True) - pd.to_datetime(m['cve_date_published'], utc=True, format='ISO8601')).dt.total_seconds() / 3600
    rows = []
    mapping = {'description': ('v4_has_description', 'v5_has_description', 'description.description_data[].value ↔ containers.cna.descriptions[].value', 'eşlenebilir'), 'cvss': ('v4_has_cvss', 'v5_has_cvss', 'impact (serbest yapı; cvss/cvssv3/baseMetricV3) ↔ containers.cna.metrics[].cvssV3_x', "kısmen: JSON 4 impact yapısı CNA'ya göre değişir"), 'cwe': ('v4_has_cwe', 'v5_has_cwe', 'problemtype.problemtype_data[].description[].value ↔ problemTypes[].descriptions[].cweId', 'eşlenebilir (CWE metinden ayrıştırılır)'), 'vendor': ('v4_has_vendor', 'v5_has_vendor', 'affects.vendor.vendor_data[].vendor_name ↔ affected[].vendor', "eşlenebilir; JSON 4'te 'n/a' yaygın"), 'product': ('v4_has_product', 'v5_has_product', 'product_data[].product_name ↔ affected[].product', "eşlenebilir; 'n/a' yaygın"), 'version': ('v4_has_version', 'v5_has_version', 'version_data[].version_value ↔ affected[].versions[]', "kısmen: JSON 4'te sürüm aralığı/durum semantiği yok"), 'references': ('v4_has_references', 'v5_has_references', 'references.reference_data[].url ↔ references[].url', "eşlenebilir; etiketler JSON 4'te sınırlı")}
    for name, (c4, c5, rule, verdict) in mapping.items():
        rows.append({'field': name, 'json4_first_public_present': round(float(df[c4].mean()), 4), 'json4_present_when_within_24h': round(float(m.loc[lag.abs() <= 24, c4].mean()), 4), 'v5_first_record_present_same_cves_na_excluded': round(float(v5[c5].mean()), 4), 'both_present': round(float((df.set_index('cve_id')[c4].values & v5[c5].values).mean()), 4), 'mapping_rule': rule, 'mapping_verdict': verdict})
    pd.DataFrame(rows).to_csv(TABLES / 's03_json4_field_availability.csv', index=False)
    note = {'n_pre2023-04_published_in_scope': int(len(pre)), 'n_json4_public_found': int(len(df)), 'n_cves_with_unparseable_json4_versions': len(bad), 'json4_within_24h': round(float((lag.abs() <= 24).mean()), 4), 'json4_within_7d': round(float((lag.abs() <= 168).mean()), 4), 'json4_delay_hours_median': round(float(lag.median()), 3), 'json4_public_before_datePublished_gt24h': round(float((lag < -24).mean()), 4), 'json4_date_public_meta_present': round(float(df['v4_date_public_meta'].notna().mean()), 4)}
    (METRICS / f's03_json4_note_{SNAP}.json').write_text(json.dumps(note, indent=1), encoding='utf-8')
    print(json.dumps(note, indent=1))
    print(pd.DataFrame(rows).drop(columns=['mapping_rule']).to_string())

def kev_artifact() -> dict:
    kev = load_kev()
    v5 = raw_dir() / 'cvelistV5'
    head = subprocess.check_output(['git', '-C', str(v5), 'rev-parse', 'HEAD']).decode().strip()
    paths = cl.head_paths(v5)
    pubs = {}
    for spec, data in cl.cat_file_batch(v5, (f'{head}:{paths[c]}' for c in kev['cve_id'] if c in paths)):
        md = cl.load_record(data).get('cveMetadata', {})
        pubs[cl.CVE_PATH.search(spec).group(1)] = cl.parse_date_utc(md.get('datePublished'))
    kev['pub_date_descriptive'] = kev['cve_id'].map(lambda c: pubs[c].date() if pubs.get(c) else None)
    kev['pub_year_descriptive'] = kev['pub_date_descriptive'].map(lambda d: d.year if d else None)
    kev['age_days_at_addition'] = [(a - p).days if p else None for a, p in zip(kev['t_kev'], kev['pub_date_descriptive'])]
    kev['old_gt365d'] = kev['age_days_at_addition'] > OLD_DAYS
    d = kev.groupby('t_kev').agg(n=('cve_id', 'size'), n_old=('old_gt365d', 'sum'), median_age_days=('age_days_at_addition', 'median')).reset_index()
    d['old_share'] = (d['n_old'] / d['n']).round(4)
    d.to_csv(TABLES / 's03_kev_additions_daily.csv', index=False)
    t = pd.to_datetime(kev['t_kev'])
    w = kev.assign(week=t.dt.to_period('W-SUN').dt.start_time.dt.date).groupby('week').agg(n=('cve_id', 'size'), n_old=('old_gt365d', 'sum')).reset_index()
    w.to_csv(TABLES / 's03_kev_additions_weekly.csv', index=False)
    mth = kev.assign(month=t.dt.strftime('%Y-%m')).groupby('month').agg(n=('cve_id', 'size'), n_old=('old_gt365d', 'sum')).reset_index()
    mth['old_share'] = (mth['n_old'] / mth['n']).round(4)
    mth.to_csv(TABLES / 's03_kev_additions_monthly.csv', index=False)
    yr = kev.groupby(['t_kev', 'pub_year_descriptive']).size().rename('n').reset_index()
    yr.to_csv(TABLES / 's03_kev_additions_by_pub_year.csv', index=False)
    q1, q3 = d['n'].quantile([0.25, 0.75])
    spikes = d[d['n'] > q3 + 3 * (q3 - q1)]
    rules = {}
    for min_n in (10, 20):
        for share in (0.8, 0.9):
            bulk = d[(d['n'] >= min_n) & (d['old_share'] >= share)]
            last = bulk['t_kev'].max() if len(bulk) else None
            rules[f'R1_day_n>={min_n}_old>={share}'] = {'n_bulk_days': int(len(bulk)), 'last_bulk_day': str(last), 'n_cves_in_bulk_days': int(bulk['n'].sum()), 'bulk_days': [f'{r.t_kev}:{r.n}' for r in bulk.itertuples()]}
    hi = mth[mth['old_share'] >= 0.5]
    rules['R2_last_month_old_share>=0.5'] = str(hi['month'].max()) if len(hi) else None
    hi3 = mth[mth['n_old'] >= 20]
    rules['R3_last_month_n_old>=20'] = str(hi3['month'].max()) if len(hi3) else None
    bulk_days = d[(d['n'] >= 10) & (d['old_share'] >= 0.8)]
    bulk_days.to_csv(TABLES / 's03_kev_bulk_backfill_days.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    wk = w.set_index(pd.to_datetime(w['week']))
    ax.bar(wk.index, wk['n'] - wk['n_old'], width=6, color='#0072B2', label='Yayından ≤365 gün sonra')
    ax.bar(wk.index, wk['n_old'], width=6, bottom=wk['n'] - wk['n_old'], color='#E69F00', label='Yayından >365 gün sonra (geriye dönük)')
    for s0 in TRAIN_STARTS:
        ax.axvline(pd.Timestamp(s0), color='#555555', lw=0.7, ls='--')
    ax.set_ylabel('Haftalık KEV eklemesi')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / 'feasibility_kev_additions.png')
    res = {'kev_catalog_entries': int(len(kev)), 'kev_first_dateAdded': str(kev['t_kev'].min()), 'kev_last_dateAdded': str(kev['t_kev'].max()), 'kev_after_t_freeze': int((kev['t_kev'] > T_FREEZE).sum()), 'kev_without_cvelistV5_record': int(kev['pub_date_descriptive'].isna().sum()), 'kev_old_gt365d_share': round(float(kev['old_gt365d'].mean()), 4), 'daily_sum_eq_total': int(d['n'].sum()) == len(kev), 'weekly_sum_eq_total': int(w['n'].sum()) == len(kev), 'monthly_sum_eq_total': int(mth['n'].sum()) == len(kev), 'spike_threshold': float(q3 + 3 * (q3 - q1)), 'spike_days': [f'{r.t_kev}:{r.n}:{r.old_share}' for r in spikes.itertuples()], 'backfill_rules': rules, 'first_day_top10': d.sort_values('n', ascending=False).head(10)[['t_kev', 'n', 'old_share']].astype(str).to_dict(orient='records')}
    return res

def _neg_causes(neg: pd.DataFrame, recon: pd.DataFrame, nvd: pd.DataFrame, j4df: pd.DataFrame) -> pd.DataFrame:
    et = ZoneInfo('America/New_York')
    x = neg.merge(nvd, on='cve_id', how='left').merge(j4df[['cve_id', 'json4_public_at']], on='cve_id', how='left', suffixes=('', '_j4'))
    pub_raw = pd.to_datetime(x['cve_date_published'].where(~x['proxy_t_pub'], None), utc=True, format='ISO8601')
    pub_raw = pub_raw.fillna(pd.to_datetime(x['first_published_commit_at'], utc=True))
    x['flag_calendar_or_time_precision'] = pub_raw.dt.tz_convert(et).dt.date <= x['t_kev']
    cur = pd.to_datetime(x['current_date_published'], utc=True, format='ISO8601').dt.date
    x['flag_cve_date_field_changed'] = (cur != x['t_pub']) & cur.notna() & (cur <= x['t_kev'])
    nvdd = pd.to_datetime(x['nvd_published_raw'], utc=True, format='ISO8601').dt.date
    j4d = pd.to_datetime(x['json4_public_at'], utc=True).dt.date
    x['flag_cross_source_date_difference'] = nvdd.notna() & (nvdd <= x['t_kev']) | j4d.notna() & (j4d <= x['t_kev'])
    x['flag_bulk_import_artifact'] = x['t_pub_source'] == 'cvelistV5_bulk_import_snapshot'
    x['flag_missing_or_proxy_date'] = x['proxy_t_pub']
    order = ['calendar_or_time_precision', 'cve_date_field_changed', 'cross_source_date_difference', 'bulk_import_artifact', 'missing_or_proxy_date']
    tp = pd.to_datetime(x['t_pub'])
    x['flag_sources_consistent'] = nvdd.notna() & ((pd.to_datetime(nvdd) - tp).dt.days.abs() <= 1) & (j4d.isna() | ((pd.to_datetime(j4d) - tp).dt.days.abs() <= 1))
    x['negative_cause'] = 'unexplained_date_conflict'
    x.loc[x['flag_sources_consistent'], 'negative_cause'] = 'kev_before_cve_record_publication_sources_consistent'
    for c in reversed(order):
        x.loc[x[f'flag_{c}'], 'negative_cause'] = c
    x['plan_5_4_class'] = x['negative_cause'].map({'calendar_or_time_precision': 'same_day_tz', 'cve_date_field_changed': 'record_republished', 'cross_source_date_difference': 'source_mismatch'}).fillna('unexplained')
    return x

def scenarios(j4df: pd.DataFrame) -> dict:
    recon = pd.read_parquet(RECON)
    kev = load_kev()
    nvd = pd.read_parquet(INTERIM / f'nvd_current_s02_{SNAP}.parquet')[['cve_id', 'nvd_published_raw']]
    built = sc.build(recon, j4df)
    rows, per_rows, hz_rows, neg_all, recs = ([], [], [], [], [])
    for name, df in built.items():
        e = sc.add_events(df, kev, T_FREEZE)
        e['scenario'] = name
        e['period'] = [sc.period_of(d, PERIODS) for d in e['t_pub']]
        e['pub_year'] = [d.year for d in e['t_pub']]
        recs.append(e[['scenario', 'cve_id', 't_pub', 't_pub_source', 'proxy_t_pub', 'disclosure_fidelity', 't_kev', 'event_freeze', 'duration_days', 'duration_sign', 'zero_day_event', 'period', 'pub_year']])
        ev = e[e['event_freeze'] == 1]
        nonneg = ev[ev['duration_days'] >= 0]
        rows.append({'scenario': name, 'n_cve': len(e), 'n_kev_events': len(ev), 'event_rate': round(len(ev) / len(e), 5), 't_pub_min': str(e['t_pub'].min()), 't_pub_max': str(e['t_pub'].max()), 'max_followup_days': int((T_FREEZE - e['t_pub'].min()).days), 'min_followup_days': int((T_FREEZE - e['t_pub'].max()).days), 'events_positive': int((ev['duration_days'] > 0).sum()), 'events_zero': int((ev['duration_days'] == 0).sum()), 'events_negative': int((ev['duration_days'] < 0).sum()), 'events_le30d': int((nonneg['duration_days'] <= 30).sum()), 'events_le365d': int((nonneg['duration_days'] <= 365).sum()), 'n_proxy_t_pub': int(e['proxy_t_pub'].sum()), 'by_t_pub_source': json.dumps(e['t_pub_source'].value_counts().to_dict()), 'cve_by_year': json.dumps(e['pub_year'].value_counts().sort_index().to_dict()), 'kev_by_year': json.dumps(ev['pub_year'].value_counts().sort_index().to_dict())})
        v0, t0 = (PERIODS['valid'][0], PERIODS['test'][0])
        starts = TRAIN_STARTS if name in ('B', 'B7', 'C') else [sc.STRICT_START]
        for start in starts:
            for split_name, split in (('config', PERIODS), ('alt_2023-04_illustrative', ALT_SPLIT)):
                if split_name != 'config' and name not in ('A', 'B', 'D'):
                    continue
                win = dict(split)
                if split_name == 'config':
                    win['train'] = (start, PERIODS['train'][1])
                vs, ts = (win['valid'][0], win['test'][0])
                for per, (a, b) in win.items():
                    g = e[(e['t_pub'] >= a) & (e['t_pub'] <= b)]
                    gev = g[g['event_freeze'] == 1]
                    cut = {'train': vs, 'valid': ts, 'test': T_FREEZE}[per]
                    ok = gev[gev['t_kev'] <= cut]
                    nn = ok[ok['duration_days'] >= 0]
                    per_rows.append({'scenario': name, 'split': split_name, 'train_start': str(start), 'period': per, 'from': str(a), 'to': str(b), 'label_cutoff': str(cut), 'n_cve': len(g), 'events_at_freeze': len(gev), 'events_at_cutoff': len(ok), 'events_at_cutoff_nonneg': len(nn), 'events_30d_at_cutoff': int((nn['duration_days'] <= 30).sum()), 'events_365d_at_cutoff': int((nn['duration_days'] <= 365).sum()), 'zero_day_at_cutoff': int((ok['duration_days'] == 0).sum()), 'negative_at_cutoff': int((ok['duration_days'] < 0).sum()), 'event_rate_at_cutoff': round(len(ok) / len(g), 5) if len(g) else None})
                g = e[(e['t_pub'] >= win['train'][0]) & (e['t_pub'] <= win['valid'][1])]
                ok = g[(g['event_freeze'] == 1) & (g['t_kev'] <= ts)]
                nn = ok[ok['duration_days'] >= 0]
                per_rows.append({'scenario': name, 'split': split_name, 'train_start': str(start), 'period': 'train+valid_testmodel', 'from': str(win['train'][0]), 'to': str(win['valid'][1]), 'label_cutoff': str(ts), 'n_cve': len(g), 'events_at_freeze': int(g['event_freeze'].sum()), 'events_at_cutoff': len(ok), 'events_at_cutoff_nonneg': len(nn), 'events_30d_at_cutoff': int((nn['duration_days'] <= 30).sum()), 'events_365d_at_cutoff': int((nn['duration_days'] <= 365).sum()), 'zero_day_at_cutoff': int((ok['duration_days'] == 0).sum()), 'negative_at_cutoff': int((ok['duration_days'] < 0).sum()), 'event_rate_at_cutoff': round(len(ok) / len(g), 5) if len(g) else None})
        for h in (30, 365):
            hz_rows.append({'scenario': name, 'horizon_days': h, 'events_within': int(((ev['duration_days'] >= 0) & (ev['duration_days'] <= h)).sum()), 'zero_day_included': int((ev['duration_days'] == 0).sum())})
        neg = ev[ev['duration_days'] < 0].copy()
        if len(neg):
            neg_all.append(_neg_causes(neg, recon, nvd, j4df))
    comp = pd.DataFrame(rows)
    comp.to_csv(TABLES / 's03_scenario_comparison.csv', index=False)
    per = pd.DataFrame(per_rows)
    per.to_csv(TABLES / 'feasibility_cohort_counts.csv', index=False)
    pd.DataFrame(hz_rows).to_csv(TABLES / 's03_horizon_event_counts.csv', index=False)
    rec_all = pd.concat(recs).sort_values(['scenario', 'cve_id']).reset_index(drop=True)
    rec_all.to_parquet(SCEN_OUT, index=False)
    signs = rec_all[rec_all['event_freeze'] == 1].groupby(['scenario', 'duration_sign']).size().unstack(fill_value=0)
    signs.to_csv(TABLES / 's03_duration_signs.csv')
    negdf = pd.concat(neg_all) if neg_all else pd.DataFrame()
    cols = ['scenario', 'cve_id', 't_pub', 't_kev', 'duration_days', 't_pub_source', 'disclosure_fidelity', 'cve_date_published', 'current_date_published', 'nvd_published_raw', 'json4_public_at', 'flag_calendar_or_time_precision', 'flag_cve_date_field_changed', 'flag_cross_source_date_difference', 'flag_bulk_import_artifact', 'flag_missing_or_proxy_date', 'flag_sources_consistent', 'negative_cause', 'plan_5_4_class']
    if len(negdf):
        negdf[cols].to_csv(TABLES / 's03_negative_durations.csv', index=False)
        negdf[cols].to_parquet(INTERIM / f'date_quality_s03_{SNAP}.parquet', index=False)
        causes = negdf.groupby(['scenario', 'negative_cause']).size().unstack(fill_value=0)
        causes.to_csv(TABLES / 's03_negative_duration_causes.csv')
    feas = []
    thr = {'train': 150, 'valid': 40, 'test': 100, 'train+valid_testmodel': 150}
    for r in per.itertuples():
        feas.append({'scenario': r.scenario, 'split': r.split, 'train_start': r.train_start, 'period': r.period, 'events_at_cutoff': r.events_at_cutoff, 'events_30d': r.events_30d_at_cutoff, 'event_rate': r.event_rate_at_cutoff, 'events_per_feature_30': round(r.events_at_cutoff / 30, 2), 'events_per_feature_60': round(r.events_at_cutoff / 60, 2), 'min_events_threshold': thr[r.period], 'meets_threshold': r.events_at_cutoff >= thr[r.period], 'ratio_ge5_at_30': r.events_at_cutoff / 30 >= 5, 'ratio_ge5_at_60': r.events_at_cutoff / 60 >= 5})
    pd.DataFrame(feas).to_csv(TABLES / 's03_event_feature_feasibility.csv', index=False)
    return {'scenario_comparison': comp.drop(columns=['cve_by_year', 'kev_by_year']).to_dict(orient='records'), 'duration_signs': signs.to_dict(orient='index'), 'negative_causes': causes.to_dict(orient='index') if len(negdf) else {}, 'rejected_at_freeze_excluded': int(((recon['current_state'] == 'REJECTED') & recon['t_pub'].notna()).sum())}

def main(cmd: str) -> None:
    if cmd in ('json4', 'all'):
        json4()
    if cmd in ('kev', 'scenarios', 'all'):
        res = {'cvelistV5_head': subprocess.check_output(['git', '-C', str(raw_dir() / 'cvelistV5'), 'rev-parse', 'HEAD']).decode().strip(), 't_freeze': str(T_FREEZE)}
        res['kev_artifact'] = kev_artifact()
        res.update(scenarios(pd.read_parquet(J4OUT)))
        (METRICS / f's03_summary_{SNAP}.json').write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')
        print(json.dumps({k: v for k, v in res['kev_artifact'].items() if k != 'backfill_rules'}, indent=1, default=str))
        print(json.dumps({k: {x: v[x] for x in v if x != 'bulk_days'} if isinstance(v, dict) else v for k, v in res['kev_artifact']['backfill_rules'].items()}, indent=1, default=str))
        print(pd.DataFrame(res['scenario_comparison']).T.to_string())
        print(json.dumps({'signs': res['duration_signs'], 'causes': res['negative_causes'], 'rejected': res['rejected_at_freeze_excluded']}, indent=1, default=str))
if __name__ == '__main__':
    main(sys.argv[1])
