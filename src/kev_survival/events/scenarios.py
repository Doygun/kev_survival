from __future__ import annotations
import datetime as dt
import pandas as pd
STRICT_START = dt.date(2023, 4, 1)
EARLIEST = dt.date(2021, 11, 1)
END = dt.date(2024, 12, 31)
H24, H7D = (24.0, 168.0)

def _base(recon: pd.DataFrame) -> pd.DataFrame:
    return recon[(recon['reconstruction_status'] == 'reconstructed') & (recon['current_state'] == 'PUBLISHED')]

def _strict(b: pd.DataFrame) -> pd.DataFrame:
    s = b[(b['disclosure_fidelity'] == 'near_disclosure_le24h') & b['t_pub'].notna()]
    s = s[(s['t_pub'] >= STRICT_START) & (s['t_pub'] <= END)]
    return s.assign(t_pub_source='cvelistV5_first_published', proxy_t_pub=False)

def build(recon: pd.DataFrame, json4: pd.DataFrame) -> dict[str, pd.DataFrame]:
    b = _base(recon)
    a = _strict(b)
    pre = b[b['t_pub'].notna() & (b['t_pub'] >= EARLIEST) & (b['t_pub'] < STRICT_START)]
    j = pre.merge(json4, on='cve_id', how='inner')
    lag = (pd.to_datetime(j['json4_public_at'], utc=True) - pd.to_datetime(j['cve_date_published'], utc=True, format='ISO8601')).dt.total_seconds() / 3600
    j = j.assign(json4_delay_hours=lag, t_pub_source='json4_first_public+v5_datePublished', proxy_t_pub=False)
    out = {'A': a, 'B': pd.concat([j[lag.abs() <= H24], a]), 'B7': pd.concat([j[lag.abs() <= H7D], a]), 'C': pd.concat([pre.assign(t_pub_source=pre['disclosure_fidelity'].map({'bulk_import_snapshot': 'cvelistV5_bulk_import_snapshot'}).fillna('cvelistV5_first_published'), proxy_t_pub=False), a])}
    m = b[b['disclosure_fidelity'] == 'missing_date_published'].copy()
    m['t_pub'] = pd.to_datetime(m['first_published_commit_at'], utc=True).dt.date
    m = m[(m['t_pub'] >= STRICT_START) & (m['t_pub'] <= END)]
    out['D'] = pd.concat([a, m.assign(t_pub_source='proxy_first_published_commit', proxy_t_pub=True)])
    return {k: v.sort_values('cve_id').reset_index(drop=True) for k, v in out.items()}

def add_events(df: pd.DataFrame, kev: pd.DataFrame, t_freeze: dt.date) -> pd.DataFrame:
    k = kev[kev['t_kev'] <= t_freeze][['cve_id', 't_kev']]
    out = df.drop(columns=[c for c in ('t_kev',) if c in df]).merge(k, on='cve_id', how='left')
    out['event_freeze'] = out['t_kev'].notna().astype(int)
    out['duration_days'] = [(tk - tp).days if isinstance(tk, dt.date) and (not pd.isna(tk)) else (t_freeze - tp).days for tk, tp in zip(out['t_kev'], out['t_pub'])]
    out['duration_sign'] = 'censored'
    ev = out['event_freeze'] == 1
    out.loc[ev & (out['duration_days'] > 0), 'duration_sign'] = 'positive'
    out.loc[ev & (out['duration_days'] == 0), 'duration_sign'] = 'zero'
    out.loc[ev & (out['duration_days'] < 0), 'duration_sign'] = 'negative'
    out['zero_day_event'] = (out['duration_sign'] == 'zero').astype(int)
    return out

def period_of(d: dt.date, periods: dict[str, tuple[dt.date, dt.date]]) -> str:
    for k, (a, b) in periods.items():
        if a <= d <= b:
            return k
    return 'pre_candidate' if EARLIEST <= d < periods['train'][0] else 'outside'
