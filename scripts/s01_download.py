from __future__ import annotations
import datetime as dt
import gzip
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kev_survival.io.common import ROOT, append_jsonl, http_get, load_project, raw_dir, utc_now, write_new
LOGS = ROOT / 'logs'
NVD = 'https://services.nvd.nist.gov/rest/json'
NVD_SLEEP = 6.5

def status(source: str, **kw) -> None:
    append_jsonl(LOGS / 'download_status.jsonl', {'time_utc': utc_now(), 'source': source, **kw})

def kev() -> None:
    out = raw_dir() / 'cisa_kev' / 'known_exploited_vulnerabilities.json'
    if out.exists():
        print('KEV zaten var')
        return
    url = 'https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json'
    r = http_get(url, None, LOGS / 'download_kev.jsonl')
    write_new(out, r.content)
    d = json.loads(r.content)
    status('cisa_kev', ok=True, catalogVersion=d.get('catalogVersion'), dateReleased=d.get('dateReleased'), count=d.get('count'), last_modified=r.headers.get('Last-Modified'))
    print('KEV', d.get('catalogVersion'), d.get('count'))

def edb() -> None:
    base = raw_dir() / 'exploitdb'
    log = LOGS / 'download_exploitdb.jsonl'
    meta = http_get('https://gitlab.com/api/v4/projects/exploit-database%2Fexploitdb/repository/commits/main', None, log).json()
    sha = meta['id']
    out = base / f'files_exploits_{sha[:12]}.csv'
    if not out.exists():
        r = http_get(f'https://gitlab.com/exploit-database/exploitdb/-/raw/{sha}/files_exploits.csv', None, log)
        write_new(out, r.content)
    readme = base / f'README_{sha[:12]}.md'
    if not readme.exists():
        r = http_get(f'https://gitlab.com/exploit-database/exploitdb/-/raw/{sha}/README.md', None, log)
        write_new(readme, r.content)
    status('exploitdb', ok=True, commit=sha, committed_date=meta['committed_date'])
    print('EDB', sha)

def epss() -> None:
    cfg = load_project()
    start = dt.date(2021, 11, 1)
    end = dt.date.fromisoformat(cfg['periods']['test'][1]) + dt.timedelta(days=7)
    base = raw_dir() / 'epss'
    log = LOGS / 'download_epss.jsonl'
    test_start = dt.date.fromisoformat(cfg['periods']['test'][0])
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d >= test_start] + [d for d in days if d < test_start]
    missing = []
    for d in days:
        out = base / f'epss_scores-{d.isoformat()}.csv.gz'
        if not out.exists():
            try:
                r = http_get(f'https://epss.empiricalsecurity.com/epss_scores-{d.isoformat()}.csv.gz', None, log, timeout=120)
                write_new(out, r.content)
            except Exception as exc:
                missing.append(d.isoformat())
                append_jsonl(log, {'date': d.isoformat(), 'error': repr(exc)})
    status('epss', ok=not missing, start=start.isoformat(), end=end.isoformat(), missing_days=missing)
    print('EPSS eksik gün:', missing)

def _nvd_windows(start: dt.date, end: dt.date, days: int=119):
    s = start
    while s <= end:
        e = min(s + dt.timedelta(days=days), end + dt.timedelta(days=1))
        yield (s, e)
        s = e

def nvd_cves() -> None:
    cfg = load_project()
    end = dt.date.fromisoformat(cfg['periods']['test'][1])
    base = raw_dir() / 'nvd_cves'
    log = LOGS / 'download_nvd.jsonl'
    for s, e in _nvd_windows(dt.date(2020, 11, 1), end):
        idx, total = (0, None)
        while total is None or idx < total:
            out = base / f'nvd_cves_pub_{s}_{e}_start{idx:06d}.json.gz'
            if out.exists():
                with gzip.open(out) as f:
                    d = json.load(f)
            else:
                params = {'pubStartDate': f'{s}T00:00:00.000', 'pubEndDate': f'{e}T00:00:00.000', 'resultsPerPage': 2000, 'startIndex': idx}
                r = http_get(f'{NVD}/cves/2.0', params, log, backoff=30)
                write_new(out, gzip.compress(r.content))
                d = r.json()
                time.sleep(NVD_SLEEP)
            total = d['totalResults']
            idx += d['resultsPerPage'] or 2000
            print(s, e, idx, total, flush=True)
    status('nvd_cves', ok=True, range=['2020-11-01', str(end)])

def nvd_history(n_per_stratum: int=75) -> None:
    cfg = load_project()
    base = raw_dir()
    kev_ids = {v['cveID'] for v in json.load(open(base / 'cisa_kev' / 'known_exploited_vulnerabilities.json', encoding='utf-8'))['vulnerabilities']}
    periods = {k: tuple((dt.date.fromisoformat(x) for x in v)) for k, v in cfg['periods'].items()}
    strata: dict[str, list[str]] = {k: [] for k in periods} | {'kev': []}
    for p in sorted((base / 'nvd_cves').glob('*.json.gz')):
        with gzip.open(p) as f:
            for v in json.load(f)['vulnerabilities']:
                c = v['cve']
                if c.get('vulnStatus') == 'Rejected':
                    continue
                pub = dt.date.fromisoformat(c['published'][:10])
                for k, (a, b) in periods.items():
                    if a <= pub <= b:
                        strata['kev' if c['id'] in kev_ids else k].append(c['id'])
    rng = random.Random(cfg['seeds'][0])
    sample = {k: rng.sample(sorted(set(v)), min(n_per_stratum, len(set(v)))) for k, v in strata.items()}
    sample_path = ROOT / 'data' / 'interim' / 's01_nvd_history_sample.json'
    if not sample_path.exists():
        sample_path.write_text(json.dumps({'seed': cfg['seeds'][0], 'n_per_stratum': n_per_stratum, 'population_sizes': {k: len(set(v)) for k, v in strata.items()}, 'sample': sample}, indent=1), encoding='utf-8')
    sample = json.loads(sample_path.read_text(encoding='utf-8'))['sample']
    out_dir = base / 'nvd_cvehistory'
    log = LOGS / 'download_nvd.jsonl'
    for k, ids in sample.items():
        for cve in ids:
            out = out_dir / f'{cve}.json.gz'
            if out.exists():
                continue
            r = http_get(f'{NVD}/cvehistory/2.0', {'cveId': cve, 'resultsPerPage': 5000}, log, backoff=30)
            write_new(out, gzip.compress(r.content))
            time.sleep(NVD_SLEEP)
        print('history', k, 'tamam', flush=True)
    status('nvd_cvehistory', ok=True, n=sum((len(v) for v in sample.values())))
if __name__ == '__main__':
    {'kev': kev, 'edb': edb, 'epss': epss, 'nvd_cves': nvd_cves, 'nvd_history': nvd_history}[sys.argv[1]]()
