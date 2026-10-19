from __future__ import annotations
import datetime as dt
import gzip
import re
from pathlib import Path
import pandas as pd
TOLERANCES = (0, 1, 2, 3, 7)
FILE_RE = re.compile('epss_scores-(\\d{4}-\\d{2}-\\d{2})\\.csv\\.gz$')
VERSION_MAP = {'v2022.01.01': 'v2', 'v2023.03.01': 'v3', 'v2025.03.14': 'v4'}

def file_day(p: Path) -> dt.date:
    return dt.date.fromisoformat(FILE_RE.search(p.name).group(1))

def header_version(p: Path) -> str | None:
    with gzip.open(p, 'rt', encoding='utf-8') as fh:
        head = fh.readline()
    m = re.search('model_version:([^,\\s]+)', head) if head.startswith('#') else None
    return m.group(1) if m else None

def first_scores(files: list[Path], wanted: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    first: dict[str, tuple] = {}
    days = []
    for f in sorted(files, key=file_day):
        day, ver = (file_day(f), header_version(f))
        df = pd.read_csv(f, comment='#', usecols=['cve', 'epss', 'percentile'], dtype={'cve': str, 'epss': float, 'percentile': float})
        days.append({'day': day, 'n_rows': len(df), 'model_version_raw': ver, 'file': f.name})
        new = df[df['cve'].isin(wanted.difference(first))]
        for cve, s, pc in zip(new['cve'], new['epss'], new['percentile']):
            first[cve] = (day, s, pc, ver, f.name)
    e = pd.DataFrame([(k, *v) for k, v in sorted(first.items())], columns=['cve_id', 'first_epss_available_at', 'first_epss_score', 'first_epss_percentile', 'epss_model_version_raw', 'epss_source_file'])
    e['epss_model_version'] = e['epss_model_version_raw'].map(VERSION_MAP)
    return (e, pd.DataFrame(days))

def attach(cohort: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    out = cohort.merge(e, on='cve_id', how='left')
    d = (pd.to_datetime(out['first_epss_available_at']) - pd.to_datetime(out['t_pub'])).dt.days.astype('Int64')
    out['epss_delay_days'] = d
    for t in TOLERANCES:
        out[f'epss_available_within_{t}d'] = d.le(t).fillna(False).astype(bool)
    return out
