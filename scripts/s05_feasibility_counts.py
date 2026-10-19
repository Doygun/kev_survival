from __future__ import annotations
import datetime as dt
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kev_survival.events import scenarios as sc
from kev_survival.io.common import ROOT, load_project
CFG = load_project()
SNAP = str(CFG['t_freeze']).replace('-', '')
T_FREEZE = dt.date.fromisoformat(str(CFG['t_freeze']))
P = {k: tuple((dt.date.fromisoformat(x) for x in v)) for k, v in CFG['periods'].items()}
START = dt.date.fromisoformat(CFG['cohort_start'])
INTERIM, TABLES, METRICS = (ROOT / 'data' / 'interim', ROOT / 'results' / 'tables', ROOT / 'results' / 'metrics')
PRE_V1 = 'kev_before_cve_record_publication_sources_consistent'
N_TERMS = 30

def cohort(s: str) -> pd.DataFrame:
    r = pd.read_parquet(INTERIM / f's03_scenario_records_{SNAP}.parquet')
    dq = pd.read_parquet(INTERIM / f'date_quality_s03_{SNAP}.parquet')
    start = sc.STRICT_START if s == 'A' else START
    g = r[(r['scenario'] == s) & (r['t_pub'] >= start)].copy()
    q = dq[dq['scenario'] == s].set_index('cve_id')['negative_cause']
    cause = g['cve_id'].map(q)
    g = g[~cause.isin([PRE_V1]) & (cause.isna() | (cause == 'calendar_or_time_precision'))].copy()
    g.loc[g['cve_id'].map(q).eq('calendar_or_time_precision').fillna(False), 'duration_days'] = 0
    g['period'] = [sc.period_of(d, P) for d in g['t_pub']]
    return g

def count(g: pd.DataFrame, periods: list[str], cutoff: dt.date) -> dict:
    x = g[g['period'].isin(periods)]
    tk = pd.to_datetime(x['t_kev'])
    ev = x['t_kev'].notna() & (tk <= pd.Timestamp(cutoff))
    d = (tk - pd.to_datetime(x['t_pub'])).dt.days.where(ev)
    d = d.where(~(ev & (d < 0)), 0)
    return {'n_cve': len(x), 'events': int(ev.sum()), 'events_30d': int((ev & d.between(0, 30)).sum()), 'events_365d': int((ev & d.between(0, 365)).sum()), 'events_per_term_30': round(ev.sum() / N_TERMS, 2)}

def main() -> None:
    ctx = {'validmodel_train': (['train'], P['valid'][0]), 'testmodel_train_valid': (['train', 'valid'], P['test'][0]), 'valid_eval': (['valid'], P['test'][0]), 'test_eval': (['test'], T_FREEZE)}
    rows = []
    for s in ('B', 'A', 'B7'):
        g = cohort(s)
        for name, (per, cut) in ctx.items():
            rows.append({'scenario': s, 'context': name, 't_cutoff': str(cut), **count(g, per, cut)})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / f's05_event_counts_{SNAP}.csv', index=False)
    recon = pd.read_parquet(INTERIM / f'first_published_records_s02_{SNAP}.parquet')
    base = sc._base(recon)
    elig = base[base['t_pub'].notna() & (base['t_pub'] >= START) & (base['t_pub'] <= sc.END)]
    r = pd.read_parquet(INTERIM / f's03_scenario_records_{SNAP}.parquet')
    b = r[(r['scenario'] == 'B') & (r['t_pub'] >= START)]
    pre = elig['t_pub'] < sc.STRICT_START
    bpre = b['t_pub'] < sc.STRICT_START
    out = {'B_eligible': len(elig), 'B_reconstructed_24h': len(b), 'B_rate': len(b) / len(elig), 'B_pre2023_04_rate': int(bpre.sum()) / int(pre.sum()), 'B_post_rate': int((~bpre).sum()) / int((~pre).sum()), 'event_counts': rows}
    (METRICS / f's05_feasibility_counts_{SNAP}.json').write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
    print(t.to_string())
    print({k: v for k, v in out.items() if k != 'event_counts'})
if __name__ == '__main__':
    main()
