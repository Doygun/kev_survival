from __future__ import annotations
import re
import numpy as np
import pandas as pd
CAT = ['cvss_av', 'cvss_ac', 'cvss_pr', 'cvss_ui', 'cvss_s', 'cvss_c', 'cvss_i', 'cvss_a', 'cwe_top_group']
LOG1P = ['n_affected_products', 'vendor_cve_12m', 'vendor_kev_ever', 'vendor_kev_12m', 'desc_len_words', 'n_refs']
NUM = ['cvss_base_score', 'pub_year_month_index'] + LOG1P
BIN = ['cvss_missing', 'cvss_is_v4', 'cwe_missing', 'affected_present', 'flag_exploited_in_wild', 'flag_rce', 'flag_auth_bypass', 'flag_priv_esc', 'flag_info_disclosure', 'flag_dos']
TERMS = CAT + NUM + BIN
assert len(TERMS) == 27

def vendor_history(target: pd.DataFrame, universe: pd.DataFrame, kev: pd.DataFrame, current: bool, t_freeze=None) -> pd.DataFrame:
    u = universe.dropna(subset=['vendor_key', 't_pub'])
    k = kev.merge(u[['cve_id', 'vendor_key', 't_pub']], on='cve_id', how='inner')
    if current:
        k = k[k['t_kev'] <= t_freeze]
    up = {v: np.sort(pd.to_datetime(g['t_pub']).values) for v, g in u.groupby('vendor_key')}
    kt = {v: np.sort(pd.to_datetime(g['t_kev']).values) for v, g in k.groupby('vendor_key')}
    kp = {v: np.sort(pd.to_datetime(g['t_pub']).values) for v, g in k.groupby('vendor_key')}
    kset = set(k['cve_id'])
    out = np.zeros((len(target), 3), dtype=np.int64)
    tp = pd.to_datetime(target['t_pub']).values
    year = np.timedelta64(365, 'D')
    for i, (v, t, cid) in enumerate(zip(target['vendor_key'], tp, target['cve_id'])):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        a = up.get(v)
        if a is not None:
            out[i, 0] = np.searchsorted(a, t, 'left') - np.searchsorted(a, t - year, 'left')
        if current:
            self_in = int(cid in kset)
            b = kt.get(v)
            if b is not None:
                out[i, 1] = len(b) - self_in
            c = kp.get(v)
            if c is not None:
                out[i, 2] = np.searchsorted(c, t, 'left') - np.searchsorted(c, t - year, 'left')
        else:
            b = kt.get(v)
            if b is not None:
                n_before = np.searchsorted(b, t, 'left')
                out[i, 1] = n_before
                out[i, 2] = n_before - np.searchsorted(b, t - year, 'left')
    return pd.DataFrame(out, columns=['vendor_cve_12m', 'vendor_kev_ever', 'vendor_kev_12m'], index=target.index)

def raw_terms(f: pd.DataFrame, t_pub: pd.Series, flags: dict[str, str]) -> pd.DataFrame:
    x = pd.DataFrame(index=f.index)
    has = f['cvss_version'].notna()
    for c in ['cvss_av', 'cvss_ac', 'cvss_pr', 'cvss_ui', 'cvss_s', 'cvss_c', 'cvss_i', 'cvss_a']:
        x[c] = f[c].where(has) if c in f else None
    x['cvss_base_score'] = pd.to_numeric(f['cvss_base_score'], errors='coerce').where(has)
    x['cvss_missing'] = (~has).astype(int)
    x['cvss_is_v4'] = (f['cvss_version'] == '4.0').astype(int)
    x['cwe_top_group'] = f['cwe_id']
    x['cwe_missing'] = f['cwe_id'].isna().astype(int)
    x['affected_present'] = f['affected_present'].fillna(0).astype(int)
    x['n_affected_products'] = f['n_affected_products'].fillna(0).astype(int)
    desc = f['description'].fillna('')
    x['desc_len_words'] = desc.str.split().str.len().fillna(0).astype(int)
    for name, rx in flags.items():
        x[name] = desc.str.contains(rx, flags=re.IGNORECASE, regex=True).astype(int)
    x['n_refs'] = f['n_refs'].fillna(0).astype(int)
    tp = pd.to_datetime(t_pub)
    x['pub_year_month_index'] = ((tp.dt.year - 2022) * 12 + tp.dt.month).values
    return x

class Encoder:

    def __init__(self, cwe_levels: int=10):
        self.cwe_levels = cwe_levels

    def _expand(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        x['cwe_top_group'] = x['cwe_top_group'].where(x['cwe_top_group'].isin(self.top_cwe), np.where(x['cwe_top_group'].notna(), 'OTHER', None))
        cols = {}
        for c in CAT:
            for lv in self.levels[c]:
                cols[f'{c}={lv}'] = (x[c] == lv).astype(float)
        for c in NUM:
            v = pd.to_numeric(x[c], errors='coerce').astype(float)
            if c in LOG1P:
                v = np.log1p(v)
            cols[c] = v
        for c in BIN:
            cols[c] = x[c].astype(float)
        return pd.DataFrame(cols, index=x.index)

    def fit(self, x: pd.DataFrame) -> 'Encoder':
        self.top_cwe = list(x['cwe_top_group'].dropna().value_counts().sort_index().sort_values(ascending=False, kind='stable').head(self.cwe_levels).index)
        self.levels = {}
        for c in CAT:
            col = x[c] if c != 'cwe_top_group' else x[c].where(x[c].isin(self.top_cwe), np.where(x[c].notna(), 'OTHER', None))
            self.levels[c] = sorted(pd.Series(col).dropna().unique().tolist())
        e = self._expand(x)
        self.mean = e.mean()
        self.sd = e.std(ddof=0)
        self.columns = [c for c in e.columns if self.sd[c] > 0]
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        e = self._expand(x)[self.columns]
        e = e.fillna(self.mean[self.columns])
        return (e - self.mean[self.columns]) / self.sd[self.columns]

    def params(self) -> dict:
        return {'top_cwe': self.top_cwe, 'levels': self.levels, 'columns': self.columns, 'mean': self.mean[self.columns].round(10).to_dict(), 'sd': self.sd[self.columns].round(10).to_dict()}
