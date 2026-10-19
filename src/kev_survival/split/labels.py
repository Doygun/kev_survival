from __future__ import annotations
import datetime as dt
import pandas as pd

def at_cutoff(df: pd.DataFrame, cutoff: dt.date) -> pd.DataFrame:
    if (df['t_pub'] > cutoff).any():
        raise ValueError('t_pub > t_cutoff olan kayıt etiketlenemez')
    tk = pd.to_datetime(df['t_kev'])
    tp = pd.to_datetime(df['t_pub'])
    cut = pd.Timestamp(cutoff)
    ev = tk.notna() & (tk <= cut)
    dur = (tk.where(ev, cut) - tp).dt.days
    sd = df['date_quality_class'].eq('same_day_tz') & ev
    dur = dur.where(~sd, 0)
    if (dur < 0).any():
        raise ValueError('negatif süre etiket tablosunda olamaz (sınıflanmamış negatif)')
    out = df[['cve_id']].copy()
    out['t_cutoff'] = cutoff
    out['event'] = ev.astype(int).values
    out['duration_days'] = dur.astype(int).values
    return out
