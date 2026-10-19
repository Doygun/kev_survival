from __future__ import annotations
import datetime as dt
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kev_survival.io.access import read_parquet, write_parquet
from kev_survival.io.common import ROOT, load_project, sha256_file
from kev_survival.split.labels import at_cutoff
CFG = load_project()
assert CFG['locked'] is True
SNAP = str(CFG['t_freeze']).replace('-', '')
T_FREEZE = dt.date.fromisoformat(str(CFG['t_freeze']))
P = {k: tuple((dt.date.fromisoformat(x) for x in v)) for k, v in CFG['periods'].items()}
OUT = ROOT / 'data' / 'processed' / 'labels'
CONTEXTS = {'validmodel': (['train'], P['valid'][0]), 'valid': (['valid'], P['test'][0]), 'testmodel': (['train', 'valid'], P['test'][0]), 'test': (['test'], T_FREEZE), 'finalmodel': (['train', 'valid', 'test'], T_FREEZE)}

def scenario_rows(c: pd.DataFrame, s: str) -> pd.DataFrame:
    ok = ~c['date_quality_class'].isin(['pre_disclosure_kev_listing', 'unexplained'])
    return c[(c[f'in_scenario_{s}'] == 1) & ok]

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = read_parquet(ROOT / 'data' / 'processed' / f'cohort_{SNAP}.parquet', 'S7', 'build')
    assert (c.loc[c['in_main_cohort'] == 1, 'in_scenario_B'] == 1).all()
    t = c[c['period'] == 'test']
    short = int(((pd.Timestamp(T_FREEZE) - pd.to_datetime(t['t_pub'])).dt.days < 365).sum())
    assert short == 0, 'test kaydında t_freeze − t_pub < 365'
    rows, files = ([], {})
    for s in ('B', 'A', 'B7'):
        sc = scenario_rows(c, s)
        for name, (per, cut) in CONTEXTS.items():
            x = sc[sc['period'].isin(per)].sort_values('cve_id')
            lab = at_cutoff(x[['cve_id', 't_pub', 't_kev', 'date_quality_class']], cut)
            lab.insert(1, 't_pub', x['t_pub'].values)
            lab.insert(2, 'period', x['period'].values)
            if cut == T_FREEZE:
                lab['t_kev'] = x['t_kev'].values
            p = OUT / f'labels_{name}_{s}_{SNAP}.parquet'
            write_parquet(lab, p, 'S7', 'build')
            files[p.name] = sha256_file(p)
            rows.append({'scenario': s, 'context': name, 't_cutoff': str(cut), 'periods': '+'.join(per), 'n_cve': len(lab), 'events': int(lab['event'].sum()), 'events_30d': int(((lab['event'] == 1) & (lab['duration_days'] <= 30)).sum()), 'max_duration': int(lab['duration_days'].max())})
    tab = pd.DataFrame(rows)
    tab.to_csv(ROOT / 'results' / 'tables' / f's07_label_contexts_{SNAP}.csv', index=False)
    (ROOT / 'results' / 'metrics' / f's07_summary_{SNAP}.json').write_text(json.dumps({'test_followup_lt_365': short, 'files_sha256': files}, indent=1), encoding='utf-8')
    print(tab.to_string())
if __name__ == '__main__':
    main()
