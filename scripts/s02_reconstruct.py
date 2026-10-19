from __future__ import annotations
import datetime as dt
import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kev_survival.io.common import ROOT, load_project, raw_dir
from kev_survival.reconstruct import cvelist as cl
CFG = load_project()
SNAP = str(CFG['t_freeze']).replace('-', '')
INTERIM, TABLES, METRICS = (ROOT / 'data' / 'interim', ROOT / 'results' / 'tables', ROOT / 'results' / 'metrics')
V5 = raw_dir() / 'cvelistV5'
SCOPE = (dt.date(2021, 11, 1), dt.date(2024, 12, 31))
SELECT_BUFFER = dt.timedelta(days=31)
OUT_RECON = INTERIM / f'first_published_records_s02_{SNAP}.parquet'
OUT_FIELDS = INTERIM / f'field_status_s02_{SNAP}.parquet'
OUT_NVD = INTERIM / f'nvd_current_s02_{SNAP}.parquet'
OUT_EVENTS = INTERIM / f'nvd_change_events_s02_{SNAP}.parquet'

def period_of(d) -> str:
    if d is None or pd.isna(d):
        return 'outside'
    for k, (a, b) in CFG['periods'].items():
        if dt.date.fromisoformat(a) <= d <= dt.date.fromisoformat(b):
            return k
    return 'pre_candidate' if SCOPE[0] <= d < dt.date(2022, 1, 1) else 'outside'

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def v5() -> None:
    head = subprocess.check_output(['git', '-C', str(V5), 'rev-parse', 'HEAD']).decode().strip()
    head_at = subprocess.check_output(['git', '-C', str(V5), 'log', '-1', '--format=%cI']).decode().strip()
    paths = cl.head_paths(V5)
    sel = []
    for spec, data in cl.cat_file_batch(V5, (f'{head}:{p}' for p in paths.values())):
        md = cl.load_record(data).get('cveMetadata', {})
        d = cl.parse_date_utc(md.get('datePublished'))
        if d and SCOPE[0] - SELECT_BUFFER <= d.date() <= SCOPE[1] + SELECT_BUFFER:
            sel.append(cl.CVE_PATH.search(spec).group(1))
    print('HEAD', head, head_at, 'dosya', len(paths), 'seçilen', len(sel), flush=True)
    hist, added = cl.file_histories(V5, set(sel))
    rows, st = cl.reconstruct(V5, sel, head, hist, added, paths)
    df = pd.DataFrame(rows)
    bulk = df['reconstruction_notes'].fillna('').str.contains('bulk_import')
    lag = df['publication_commit_delay_hours']
    df['disclosure_fidelity'] = 'not_reconstructed'
    df.loc[df['reconstruction_status'].eq('reconstructed') & (lag > 24), 'disclosure_fidelity'] = 'delayed_gt24h'
    df.loc[df['reconstruction_status'].eq('reconstructed') & (lag <= 24), 'disclosure_fidelity'] = 'near_disclosure_le24h'
    df.loc[df['reconstruction_status'].eq('reconstructed') & bulk, 'disclosure_fidelity'] = 'bulk_import_snapshot'
    nodate = df['reconstruction_notes'].fillna('').str.contains('datePublished_missing')
    df.loc[df['reconstruction_status'].eq('reconstructed') & nodate, 'disclosure_fidelity'] = 'missing_date_published'
    cur_pub = pd.to_datetime(df['current_date_published'], utc=True, format='ISO8601').dt.date
    df['period_by_current_datePublished'] = [period_of(d) for d in cur_pub]
    df['period'] = [period_of(d) for d in df['t_pub']]
    df['in_scope'] = df['t_pub'].map(lambda d: d is not None and (not pd.isna(d)) and (SCOPE[0] <= d <= SCOPE[1]))
    df['snapshot_id'] = SNAP
    df['cvelistV5_head'] = head
    df.to_parquet(OUT_RECON, index=False)
    s = pd.DataFrame(st).merge(df[['cve_id', 'period', 'in_scope']], on='cve_id')
    s.to_parquet(OUT_FIELDS, index=False)
    print('yazıldı', OUT_RECON.name, len(df), OUT_FIELDS.name, len(s), flush=True)

def nvd() -> None:
    rows = []
    for p in sorted((raw_dir() / 'nvd_cves').glob('*.json.gz')):
        with gzip.open(p) as f:
            for v in json.load(f)['vulnerabilities']:
                c = v['cve']
                m = c.get('metrics', {})
                refs = c.get('references', [])
                cwes = sorted({d['value'] for w in c.get('weaknesses', []) for d in w.get('description', []) if d['value'].startswith('CWE-')})
                v3 = [x for k in ('cvssMetricV31', 'cvssMetricV30') for x in m.get(k, [])]
                rows.append({'cve_id': c['id'], 'nvd_published_raw': c.get('published'), 'nvd_last_modified_raw': c.get('lastModified'), 'nvd_status': c.get('vulnStatus'), 'nvd_source_identifier': c.get('sourceIdentifier'), 'nvd_has_cvss3': bool(v3), 'nvd_has_cvss3_primary': any((x.get('type') == 'Primary' for x in v3)), 'nvd_cvss3_vectors': json.dumps(sorted({x['cvssData']['vectorString'] for x in v3})), 'nvd_has_cvss_any': bool(m), 'nvd_has_cwe': bool(cwes), 'nvd_cwe_ids': json.dumps(cwes), 'nvd_has_cpe': bool(c.get('configurations')), 'nvd_ref_urls': json.dumps(sorted({r['url'] for r in refs})), 'nvd_n_refs': len(refs), 'nvd_ref_has_exploit_tag': any(('Exploit' in r.get('tags', []) for r in refs)), 'nvd_has_kev_fields': 'cisaExploitAdd' in c})
    df = pd.DataFrame(rows).drop_duplicates('cve_id').sort_values('cve_id')
    df.to_parquet(OUT_NVD, index=False)
    print('yazıldı', OUT_NVD.name, len(df))
URL_RE = re.compile('https?://\\S+')
CVSS_KEY = re.compile('\\b(AV|AC|PR|UI|S|C|I|A|AT|VC|VI|VA|SC|SI|SA|Au):([A-Z])')

def _vec(s: str | None) -> str | None:
    if not isinstance(s, str) or not s:
        return None
    comp = CVSS_KEY.findall(s)
    return '/'.join((f'{k}:{v}' for k, v in comp)) or None

def history() -> None:
    rows = []
    for p in sorted((raw_dir() / 'nvd_cvehistory').glob('*.json.gz')):
        with gzip.open(p) as f:
            d = json.load(f)
        for ch in d.get('cveChanges', []):
            c = ch['change']
            for i, det in enumerate(c.get('details', []) or [{}]):
                rows.append({'cve_id': c['cveId'], 'cve_change_id': c.get('cveChangeId'), 'event': c.get('eventName'), 'created': c.get('created'), 'source': c.get('sourceIdentifier'), 'detail_idx': i, 'action': det.get('action'), 'type': det.get('type'), 'old': None if det.get('oldValue') is None else str(det.get('oldValue')), 'new': None if det.get('newValue') is None else str(det.get('newValue'))})
    df = pd.DataFrame(rows).sort_values(['cve_id', 'created', 'cve_change_id', 'detail_idx'])
    df.to_parquet(OUT_EVENTS, index=False)
    print('yazıldı', OUT_EVENTS.name, len(df), df['cve_id'].nunique())

def _replay(g: pd.DataFrame, typ: str, extract) -> tuple[set, bool]:
    state, ok = (set(), True)
    for _, r in g[g['type'] == typ].iterrows():
        old, new = (extract(r['old']), extract(r['new']))
        if r['action'] == 'Added':
            state |= new
        elif r['action'] == 'Removed':
            if not old <= state:
                ok = False
            state -= old
        elif r['action'] == 'Changed':
            if not old <= state:
                ok = False
            state = state - old | new
    return (state, ok)

def _cwe(s):
    return set(cl.CWE_RE.findall(s if isinstance(s, str) else ''))

def _urls(s):
    return {u.rstrip(']),') for u in URL_RE.findall(s if isinstance(s, str) else '')}

def _vecs(s):
    v = _vec(s)
    return {v} if v else set()

def history_eval() -> tuple[pd.DataFrame, dict]:
    ev = pd.read_parquet(OUT_EVENTS)
    rec = pd.read_parquet(OUT_RECON)
    nvdc = pd.read_parquet(OUT_NVD).set_index('cve_id')
    sample = json.loads((INTERIM / 's01_nvd_history_sample.json').read_text(encoding='utf-8'))['sample']
    stratum = {c: k for k, ids in sample.items() for c in ids}
    first = {}
    paths = cl.head_paths(V5)
    rs = rec[rec['cve_id'].isin(stratum) & rec['first_published_commit'].notna()]
    specs = [f'{c}:{paths[i]}' for i, c in zip(rs['cve_id'], rs['first_published_commit']) if i in paths]
    for spec, data in cl.cat_file_batch(V5, specs):
        first[cl.CVE_PATH.search(spec).group(1)] = cl.field_values(cl.load_record(data))
    out = []
    for cve in sorted(stratum):
        g = ev[ev['cve_id'] == cve]
        r = {'cve_id': cve, 'stratum': stratum[cve], 'n_change_events': int(g['cve_change_id'].nunique())}
        recv = g[g['event'] == 'New CVE Received']
        r['has_new_cve_received'] = len(recv) > 0
        r['first_event_created'] = g['created'].min()
        r['first_event_name'] = g.sort_values('created')['event'].iloc[0] if len(g) else None
        ana = g[g['event'] == 'Initial Analysis']
        r['initial_analysis_created'] = ana['created'].min() if len(ana) else None
        for typ, key in (('CVSS V3.1', 'cvss31'), ('CVSS V3', 'cvss30'), ('CVSS V4.0', 'cvss40'), ('CWE', 'cwe'), ('Reference', 'reference'), ('Description', 'description'), ('CPE Configuration', 'cpe')):
            gg = g[g['type'] == typ]
            r[f'{key}_added_events'] = int((gg['action'] == 'Added').sum())
            r[f'{key}_changed_or_removed_events'] = int(gg['action'].isin(['Changed', 'Removed']).sum())
        r['has_kev_update_event'] = bool((g['event'] == 'CVE CISA KEV Update').any())
        r['has_ssvc_event'] = bool((g['type'] == 'SSVC').any())
        if cve in nvdc.index:
            n = nvdc.loc[cve]
            cwe_state, cwe_ok = _replay(g, 'CWE', _cwe)
            ref_state, ref_ok = _replay(g, 'Reference', _urls)
            v_state = set()
            v_ok = True
            for typ in ('CVSS V3.1', 'CVSS V3'):
                s_, ok_ = _replay(g, typ, _vecs)
                v_state |= s_
                v_ok &= ok_
            cur_v = {_vec(x) for x in json.loads(n['nvd_cvss3_vectors'])}
            r['replay_cwe_matches_current'] = cwe_ok and cwe_state == set(json.loads(n['nvd_cwe_ids']))
            r['replay_ref_matches_current'] = ref_ok and ref_state == set(json.loads(n['nvd_ref_urls']))
            r['replay_cvss3_matches_current'] = v_ok and v_state == cur_v
            r['nvd_published_raw'] = n['nvd_published_raw']
        fv = first.get(cve)
        x = rec[rec['cve_id'] == cve]
        if len(x):
            r['t_pub'] = x['t_pub'].iloc[0]
            r['cve_date_published'] = x['cve_date_published'].iloc[0]
        if fv is not None and len(recv):
            rv = recv
            nvd_cwe = _cwe(' '.join(rv[rv['type'] == 'CWE']['new'].fillna('')))
            v5_cwe = set(cl.CWE_RE.findall(fv['cwe_problemtypes'] or ''))
            nvd_vec = {_vec(s) for s in rv[rv['type'].isin(['CVSS V3.1', 'CVSS V3'])]['new'].fillna('')} - {None}
            v5_vec = {_vec(v) for k, v in json.loads(fv['cvss'] or '[]') if k in ('cvssV3_1', 'cvssV3_0')} - {None}
            nvd_urls = _urls(' '.join(rv[rv['type'] == 'Reference']['new'].fillna('')))
            v5_urls = set(json.loads(fv['references'] or '[]'))
            nvd_desc = rv[rv['type'] == 'Description']['new']
            r.update({'agree_cvss3_presence': bool(nvd_vec) == bool(v5_vec), 'agree_cvss3_vector': nvd_vec == v5_vec if nvd_vec and v5_vec else None, 'agree_cwe_presence': bool(nvd_cwe) == bool(v5_cwe), 'agree_cwe_value': nvd_cwe == v5_cwe if nvd_cwe and v5_cwe else None, 'agree_reference_urls': nvd_urls == v5_urls, 'agree_description': nvd_desc.iloc[0].strip() == (fv['description'] or '').strip() if len(nvd_desc) else None})
        out.append(r)
    s = pd.DataFrame(out)
    s.to_csv(TABLES / 's02_nvd_history_sample.csv', index=False)

    def rate(x):
        x = pd.Series(x).dropna()
        return round(float(x.astype(float).mean()), 4) if len(x) else None

    def days(a, b):
        return (pd.to_datetime(a, utc=True, format='ISO8601') - pd.to_datetime(b, utc=True, format='ISO8601')).dt.total_seconds() / 86400
    has_recv = s['has_new_cve_received']
    pub = pd.to_datetime(s['cve_date_published'], utc=True, format='ISO8601')
    first_ev_minus_nvdpub = days(s['first_event_created'], s['nvd_published_raw'])
    summ = {'n_sample': int(len(s)), 'by_stratum': s['stratum'].value_counts().to_dict(), 'q1_share_with_new_cve_received': rate(has_recv), 'q1_share_with_new_cve_received_by_stratum': s.groupby('stratum')['has_new_cve_received'].mean().round(4).to_dict(), 'q1_first_event_minus_nvd_published_days_median': round(float(first_ev_minus_nvdpub.median()), 3), 'q1_first_event_within_1d_of_nvd_published': rate(first_ev_minus_nvdpub.abs() <= 1), 'q1_new_cve_received_by_nvd_published_year': s.assign(y=pd.to_datetime(s['nvd_published_raw']).dt.year).groupby('y')['has_new_cve_received'].mean().round(3).to_dict(), 'q4_q7_replay_cvss3_matches_current': rate(s.get('replay_cvss3_matches_current')), 'q5_q7_replay_cwe_matches_current': rate(s.get('replay_cwe_matches_current')), 'q6_q7_replay_reference_matches_current': rate(s.get('replay_ref_matches_current')), 'q7_replay_all_three_match': rate(s[['replay_cvss3_matches_current', 'replay_cwe_matches_current', 'replay_ref_matches_current']].all(axis=1)), 'q8_n_with_v5_and_received': int(s['agree_cwe_presence'].notna().sum()), 'q8_agree_cvss3_presence': rate(s.get('agree_cvss3_presence')), 'q8_agree_cvss3_vector_when_both': rate(s.get('agree_cvss3_vector')), 'q8_agree_cwe_presence': rate(s.get('agree_cwe_presence')), 'q8_agree_cwe_value_when_both': rate(s.get('agree_cwe_value')), 'q8_agree_reference_urls': rate(s.get('agree_reference_urls')), 'q8_agree_description': rate(s.get('agree_description')), 'q9_share_with_initial_analysis': rate(s['initial_analysis_created'].notna()), 'q9_initial_analysis_minus_tpub_days_median': round(float(days(s['initial_analysis_created'], pub).median()), 2), 'q9_initial_analysis_minus_tpub_days_p90': round(float(days(s['initial_analysis_created'], pub).quantile(0.9)), 2), 'q9_share_cpe_added': rate(s['cpe_added_events'] > 0), 'q9_share_cvss_changed_or_removed': rate(s['cvss31_changed_or_removed_events'] + s['cvss30_changed_or_removed_events'] > 0), 'q9_share_cwe_changed_or_removed': rate(s['cwe_changed_or_removed_events'] > 0), 'q9_share_reference_added_events': rate(s['reference_added_events'] > 0), 'leakage_share_kev_update_event': rate(s['has_kev_update_event']), 'leakage_share_ssvc_event': rate(s['has_ssvc_event'])}
    (METRICS / f's02_nvd_history_adequacy_{SNAP}.json').write_text(json.dumps(summ, indent=1, default=str), encoding='utf-8')
    print(json.dumps(summ, indent=1, default=str))
    return (s, summ)

def json4() -> None:
    repo = raw_dir() / 'cvelist_json4'
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
    root = subprocess.check_output(['git', '-C', str(repo), 'rev-list', '--max-parents=0', 'HEAD']).decode().split()[0]
    first = subprocess.check_output(['git', '-C', str(repo), 'log', '--format=%cI', '-1', root]).decode()
    last = subprocess.check_output(['git', '-C', str(repo), 'log', '--format=%cI', '-1']).decode()
    hist, _ = cl.file_histories(repo, None, pathspec='.')
    fa = {c: v[0][1] for c, v in hist.items()}
    months = pd.to_datetime(pd.Series(fa), unit='s', utc=True).dt.strftime('%Y-%m').value_counts().sort_index()
    res = {'head': head, 'first_commit': first.strip(), 'last_commit': last.strip(), 'n_files': len(fa), 'first_added_by_month': months.to_dict()}
    final_json = METRICS / f's02_json4_coverage_{SNAP}.json'
    final_pq = INTERIM / f'json4_first_added_s02_{SNAP}.parquet'
    new_json = final_json.with_name(final_json.stem + '_rerun.json')
    new_pq = final_pq.with_name(final_pq.stem + '_rerun.parquet')
    new_json.write_text(json.dumps(res, indent=1), encoding='utf-8')
    pd.Series(fa, name='json4_first_added_unix').rename_axis('cve_id').reset_index().sort_values('cve_id').to_parquet(new_pq, index=False)
    print(json.dumps({'rerun': {new_json.name: sha(new_json), new_pq.name: sha(new_pq)}, 'early': {final_json.name: sha(final_json) if final_json.exists() else None, final_pq.name: sha(final_pq) if final_pq.exists() else None}}, indent=1))

def json4_public() -> None:
    repo = raw_dir() / 'cvelist_json4'
    rec = pd.read_parquet(OUT_RECON)
    target = rec[rec['in_scope'] & (rec['disclosure_fidelity'] == 'bulk_import_snapshot')]
    want = set(target['cve_id'])
    hist, _ = cl.file_histories(repo, want, pathspec='.')
    paths = cl.head_paths(repo, pathspec='.')
    found: dict[str, tuple] = {}
    unparseable: dict[str, int] = {}
    pending = {c: 0 for c in sorted(want) if hist.get(c) and c in paths}
    while pending:
        specs = {c: f'{hist[c][i][0]}:{paths[c]}' for c, i in pending.items() if hist[c][i][2] != 'D'}
        contents = dict(cl.cat_file_batch(repo, specs.values()))
        nxt = {}
        for c, i in pending.items():
            data = contents.get(specs.get(c))
            try:
                r = json.loads(data.decode('utf-8', 'replace'), strict=False) if data else {}
            except json.JSONDecodeError:
                r, unparseable[c] = ({}, unparseable.get(c, 0) + 1)
            if not isinstance(r, dict) or not isinstance(r.get('CVE_data_meta'), dict):
                r = {}
            if r.get('CVE_data_meta', {}).get('STATE') == 'PUBLIC':
                found[c] = (hist[c][i][1], r)
            elif i + 1 < len(hist[c]):
                nxt[c] = i + 1
        pending = nxt
    rows = []
    for c in sorted(want):
        f = found.get(c)
        r = f[1] if f else {}
        pt = json.dumps(r.get('problemtype', {}))
        rows.append({'cve_id': c, 'json4_in_history': c in hist, 'json4_public_found': f is not None, 'json4_first_public_commit_at': dt.datetime.fromtimestamp(f[0], dt.timezone.utc) if f else None, 'json4_has_impact_cvss': bool(r.get('impact')), 'json4_has_cwe': 'CWE-' in pt, 'json4_has_affects': bool(r.get('affects', {}).get('vendor', {}).get('vendor_data')), 'json4_n_refs': len(r.get('references', {}).get('reference_data', []))})
    j = pd.DataFrame(rows).merge(target[['cve_id', 'period', 'cve_date_published', 'current_date_published']], on='cve_id')
    for col in ('cve_date_published', 'current_date_published'):
        j[f'json4_delay_h_vs_{col}'] = (pd.to_datetime(j['json4_first_public_commit_at'], utc=True) - pd.to_datetime(j[col], utc=True, format='ISO8601')).dt.total_seconds() / 3600
    j.to_parquet(INTERIM / f'json4_public_s02_{SNAP}.parquet', index=False)
    lag = j['json4_delay_h_vs_current_date_published']
    note = {'n_bulk_import_cves_in_scope': int(len(j)), 'json4_public_found': _rate(j['json4_public_found']), 'n_cves_with_unparseable_json4_versions': len(unparseable), 'delay_vs_current_datePublished_le24h': _rate((lag.abs() <= 24)[lag.notna()]), 'delay_vs_current_datePublished_le7d': _rate((lag.abs() <= 168)[lag.notna()]), 'delay_vs_first_record_datePublished_le24h': _rate((j['json4_delay_h_vs_cve_date_published'].abs() <= 24)[lag.notna()]), 'delay_h_median': round(float(lag.median()), 2), 'json4_has_impact_cvss': _rate(j.loc[j['json4_public_found'], 'json4_has_impact_cvss']), 'json4_has_cwe': _rate(j.loc[j['json4_public_found'], 'json4_has_cwe']), 'json4_has_affects': _rate(j.loc[j['json4_public_found'], 'json4_has_affects']), 'by_period_le24h': j.assign(ok=lag.abs() <= 24).groupby('period')['ok'].mean().round(4).to_dict()}
    (METRICS / f's02_json4_public_note_{SNAP}.json').write_text(json.dumps(note, indent=1, default=str), encoding='utf-8')
    print(json.dumps(note, indent=1, default=str))

def _rate(s) -> float | None:
    s = pd.Series(s).dropna()
    return round(float(s.astype(float).mean()), 4) if len(s) else None

def _first_reliable_month(sc: pd.DataFrame) -> str | None:
    m = sc.assign(month=pd.to_datetime(sc['t_pub']).dt.to_period('M').astype(str)).groupby('month').agg(n=('cve_id', 'size'), le24h=('disclosure_fidelity', lambda x: (x == 'near_disclosure_le24h').mean()), bulk=('disclosure_fidelity', lambda x: (x == 'bulk_import_snapshot').mean()))
    m.round(4).to_csv(TABLES / 's02_fidelity_by_month.csv')
    ok = (m['le24h'] >= 0.99)[::-1].cummin()[::-1]
    return ok[ok].index.min() if ok.any() else None

def tables() -> None:
    df = pd.read_parquet(OUT_RECON)
    fs = pd.read_parquet(OUT_FIELDS)
    nvdc = pd.read_parquet(OUT_NVD)
    sc = df[df['in_scope']]
    groups = [('all_2021-11_2024-12', sc)] + [(p, sc[sc['period'] == p]) for p in ('pre_candidate', 'train', 'valid', 'test')]
    rows = []
    for name, g in groups:
        rec = g[g['reconstruction_status'] == 'reconstructed']
        lag = rec['publication_commit_delay_hours']
        live = rec[(rec['current_state'] == 'PUBLISHED') & (rec['disclosure_fidelity'] == 'near_disclosure_le24h')]
        snap = rec[(rec['current_state'] == 'PUBLISHED') & (rec['disclosure_fidelity'] == 'bulk_import_snapshot')]
        rows.append({'group': name, 'n_cve_selected': len(g), 'n_reconstructed': len(rec), 'reconstruction_rate': _rate(g['reconstruction_status'] == 'reconstructed'), 'disclosure_fidelity_le24h_rate': _rate(g['disclosure_fidelity'] == 'near_disclosure_le24h'), 'bulk_import_snapshot_rate': _rate(g['disclosure_fidelity'] == 'bulk_import_snapshot'), 'delayed_gt24h_rate': _rate(g['disclosure_fidelity'] == 'delayed_gt24h'), 'n_disclosure_le24h_published': len(live), 'snapshot_cvss_any': _rate(snap['disc_has_cvss_any']), 'snapshot_cwe': _rate(snap['disc_has_cwe']), 'n_rejected_at_freeze': int((g['current_state'] == 'REJECTED').sum()), 'n_published_at_freeze': int((rec['current_state'] == 'PUBLISHED').sum()), 'first_commit_is_bulk_import': _rate(rec['reconstruction_notes'].fillna('').str.contains('bulk_import')), 'delay_h_median': round(float(lag.median()), 3), 'delay_h_p90': round(float(lag.quantile(0.9)), 3), 'delay_h_p99': round(float(lag.quantile(0.99)), 3), 'delay_le_1h': _rate(lag <= 1), 'delay_le_24h': _rate(lag <= 24), 'delay_le_7d': _rate(lag <= 168), 'delay_negative': _rate(lag < 0), 'disc_cvss_any': _rate(live['disc_has_cvss_any']), 'disc_cvss3': _rate(live['disc_has_cvss3']), 'disc_cvss4': _rate(live['disc_has_cvss4']), 'disc_cwe': _rate(live['disc_has_cwe']), 'disc_problemtype_any': _rate(live['disc_has_problemtype_any']), 'disc_affected_product': _rate(live['disc_has_affected_product']), 'disc_affected_versions': _rate(live['disc_has_affected_versions']), 'disc_references': _rate(live['disc_n_refs'] > 0), 'disc_description': _rate(live['disc_has_description']), 'disc_adp_present': _rate(live['disc_has_adp']), 'cur_cvss_any': _rate(live['cur_has_cvss_any']), 'cur_cwe': _rate(live['cur_has_cwe']), 'cur_affected_product': _rate(live['cur_has_affected_product']), 'cur_adp_present': _rate(live['cur_has_adp'])})
    tab = pd.DataFrame(rows)
    tab.to_csv(TABLES / 'feasibility_reconstruction.csv', index=False)
    fsc = fs[fs['in_scope']].merge(df[['cve_id', 'disclosure_fidelity']], on='cve_id')
    keys = ['disclosure_fidelity', 'period', 'field']
    fa = fsc.groupby(keys + ['status']).size().rename('n').reset_index()
    fa['share'] = fa['n'] / fa.groupby(keys)['n'].transform('sum')
    allp = fsc.groupby(['disclosure_fidelity', 'field', 'status']).size().rename('n').reset_index().assign(period='all_2021-11_2024-12')
    allp['share'] = allp['n'] / allp.groupby(['disclosure_fidelity', 'field'])['n'].transform('sum')
    fa = pd.concat([allp, fa])[keys + ['status', 'n', 'share']]
    fa['share'] = fa['share'].round(4)
    fa.to_csv(TABLES / 's02_field_availability.csv', index=False)
    wide = fa[(fa['period'] == 'all_2021-11_2024-12') & (fa['disclosure_fidelity'] == 'near_disclosure_le24h')].pivot(index='field', columns='status', values='share').fillna(0)
    wide.round(4).to_csv(TABLES / 's02_first_vs_current_changes.csv')
    fail = df[df['reconstruction_status'] != 'reconstructed'][['cve_id', 'reconstruction_status', 'state_at_first_commit', 'states_before_first_published', 'current_state', 'current_date_published', 'history_commit_count', 'reconstruction_notes']]
    fail.to_csv(TABLES / 's02_reconstruction_failures.csv', index=False)
    m = sc.merge(nvdc[['cve_id', 'nvd_published_raw']], on='cve_id', how='left')
    v5pub = pd.to_datetime(m['cve_date_published'], utc=True, format='ISO8601')
    nvdpub = pd.to_datetime(m['nvd_published_raw'], utc=True, format='ISO8601')
    curpub = pd.to_datetime(m['current_date_published'], utc=True, format='ISO8601')
    m['nvd_minus_v5_pub_hours'] = (nvdpub - v5pub).dt.total_seconds() / 3600
    m['t_pub_nvd'] = nvdpub.dt.date
    m['date_diff_v5_vs_nvd_days'] = (pd.to_datetime(m['t_pub_nvd']) - pd.to_datetime(m['t_pub'])).dt.days
    m['datePublished_changed'] = (curpub != v5pub) & curpub.notna()
    disc = m[(m['date_diff_v5_vs_nvd_days'].abs() >= 1) | m['datePublished_changed'] | m['nvd_published_raw'].isna()]
    disc[['cve_id', 'period', 'cve_date_published', 'current_date_published', 'nvd_published_raw', 'nvd_minus_v5_pub_hours', 'date_diff_v5_vs_nvd_days', 'datePublished_changed', 'current_state']].to_csv(TABLES / 's02_date_discrepancies.csv', index=False)
    d = m['date_diff_v5_vs_nvd_days']
    summ = {'cvelistV5_head': df['cvelistV5_head'].iloc[0], 'n_selected': int(len(df)), 'n_in_scope': int(len(sc)), 'reconstruction_status_in_scope': sc['reconstruction_status'].value_counts().to_dict(), 'reconstruction_status_selected_all': df['reconstruction_status'].value_counts().to_dict(), 'disclosure_fidelity_in_scope': sc['disclosure_fidelity'].value_counts().to_dict(), 'missing_date_published_in_first_record_by_current_period': df.loc[df['disclosure_fidelity'] == 'missing_date_published', 'period_by_current_datePublished'].value_counts().to_dict(), 'missing_date_published_first_commit_is_bulk': _rate(df.loc[df['disclosure_fidelity'] == 'missing_date_published', 'first_commit_added_files'] >= cl.BULK_COMMIT_MIN_FILES), 'disclosure_fidelity_by_period': sc.groupby('period')['disclosure_fidelity'].value_counts(normalize=True).round(4).unstack(fill_value=0).to_dict(orient='index'), 'bulk_import_commits': sc.loc[sc['disclosure_fidelity'] == 'bulk_import_snapshot'].groupby('first_published_commit').agg(n=('cve_id', 'size'), at=('first_published_commit_at', 'first'), files=('first_commit_added_files', 'first')).reset_index().to_dict(orient='records'), 'first_month_from_which_all_months_le24h_ge_0.99': _first_reliable_month(sc), 'state_at_first_commit_in_scope': sc['state_at_first_commit'].value_counts(dropna=False).to_dict(), 'current_state_in_scope': sc['current_state'].value_counts(dropna=False).to_dict(), 'date_v5_vs_nvd': {'nvd_matched': _rate(m['nvd_published_raw'].notna()), 'same_utc_day': _rate((d == 0)[d.notna()]), 'nvd_later_by_1d': _rate((d == 1)[d.notna()]), 'nvd_earlier': _rate((d < 0)[d.notna()]), 'abs_ge_2d': _rate((d.abs() >= 2)[d.notna()]), 'nvd_minus_v5_hours_median': round(float(m['nvd_minus_v5_pub_hours'].median()), 3)}, 'datePublished_changed_after_first_published': _rate(m['datePublished_changed']), 'max_commit_time_used': str(max(df['first_published_commit_at'].max(), df['current_record_commit_at'].max())), 't_freeze': str(CFG['t_freeze'])}
    (METRICS / f's02_reconstruction_summary_{SNAP}.json').write_text(json.dumps(summ, indent=1, default=str), encoding='utf-8')
    print(tab.T.to_string())
    print(wide.round(4).to_string())
    print(json.dumps(summ, indent=1, default=str))
if __name__ == '__main__':
    {'v5': v5, 'nvd': nvd, 'history': history, 'history_eval': history_eval, 'json4': json4, 'json4_public': json4_public, 'tables': tables}[sys.argv[1]]()
