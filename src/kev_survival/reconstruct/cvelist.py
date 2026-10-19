from __future__ import annotations
import datetime as dt
import json
import re
import subprocess
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
CVE_PATH = re.compile('(CVE-\\d{4}-\\d{4,})\\.json$')
CWE_RE = re.compile('CWE-\\d+')
BULK_COMMIT_MIN_FILES = 1000
FIELDS = ['description', 'cvss', 'cwe_problemtypes', 'affected_vendor', 'affected_product', 'affected_versions', 'references', 'cna_id', 'state', 'date_published', 'date_updated']
CONTENT_FIELDS = FIELDS[:7]

def git_lines(repo: Path, *args: str) -> Iterator[str]:
    p = subprocess.Popen(['git', '-C', str(repo), *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for raw in p.stdout:
        yield raw.decode('utf-8', 'replace').rstrip('\n')
    p.wait()

def head_paths(repo: Path, rev: str='HEAD', pathspec: str='cves') -> dict[str, str]:
    out = {}
    for line in git_lines(repo, 'ls-tree', '-r', '--name-only', rev, pathspec):
        m = CVE_PATH.search(line)
        if m:
            out[m.group(1)] = line
    return out

def file_histories(repo: Path, wanted: set[str] | None=None, pathspec: str='cves') -> tuple[dict[str, list[tuple[str, int, str]]], dict[str, int]]:
    hist: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    added_per_commit: dict[str, int] = defaultdict(int)
    commit, ct = ('', 0)
    for line in git_lines(repo, 'log', '--format=@@%H %ct', '--name-status', '--no-renames', '--', pathspec):
        if line.startswith('@@'):
            commit, ts = line[2:].split()
            ct = int(ts)
        elif line:
            status, _, path = line.partition('\t')
            m = CVE_PATH.search(path)
            if not m:
                continue
            if status.startswith('A'):
                added_per_commit[commit] += 1
            if wanted is None or m.group(1) in wanted:
                hist[m.group(1)].append((commit, ct, status[:1]))
    for v in hist.values():
        v.reverse()
    return (dict(hist), dict(added_per_commit))

def cat_file_batch(repo: Path, specs: Iterable[str]) -> Iterator[tuple[str, bytes | None]]:
    specs = list(specs)
    p = subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def feed():
        for s in specs:
            p.stdin.write((s + '\n').encode())
        p.stdin.close()
    t = threading.Thread(target=feed, daemon=True)
    t.start()
    for s in specs:
        header = p.stdout.readline().decode().split()
        if len(header) == 2 and header[1] == 'missing':
            yield (s, None)
            continue
        size = int(header[2])
        data = p.stdout.read(size)
        p.stdout.read(1)
        yield (s, data)
    t.join()
    p.wait()

def _canon(x) -> str | None:
    if x in (None, '', [], {}):
        return None
    return json.dumps(x, sort_keys=True, ensure_ascii=False)

def field_values(rec: dict) -> dict[str, str | None]:
    md = rec.get('cveMetadata', {})
    cna = rec.get('containers', {}).get('cna', {})
    descs = cna.get('descriptions', [])
    desc = next((d.get('value') for d in descs if str(d.get('lang', '')).startswith('en')), descs[0].get('value') if descs else None)
    cvss = sorted(((k, v.get('vectorString') or json.dumps(v, sort_keys=True)) for m in cna.get('metrics', []) for k, v in m.items() if k.startswith('cvss')))
    pts = [d for pt in cna.get('problemTypes', []) for d in pt.get('descriptions', [])]
    cwes = sorted({d['cweId'] for d in pts if d.get('cweId')} | {c for d in pts if not d.get('cweId') for c in CWE_RE.findall(d.get('description', ''))})
    pt_text = sorted({d.get('description', '') for d in pts if d.get('description')})
    aff = cna.get('affected', [])
    return {'description': desc or None, 'cvss': _canon(cvss), 'cwe_problemtypes': _canon(cwes or pt_text), 'affected_vendor': _canon(sorted({a.get('vendor') for a in aff if a.get('vendor')})), 'affected_product': _canon(sorted({f"{a.get('vendor')}|{a.get('product')}" for a in aff if a.get('product') or a.get('vendor')})), 'affected_versions': _canon(sorted((_canon([a.get('vendor'), a.get('product'), a.get('versions'), a.get('defaultStatus')]) for a in aff if a.get('versions') or a.get('defaultStatus')))), 'references': _canon(sorted({r.get('url') for r in cna.get('references', []) if r.get('url')})), 'cna_id': md.get('assignerOrgId'), 'state': md.get('state'), 'date_published': _iso_utc(md.get('datePublished')), 'date_updated': _iso_utc(md.get('dateUpdated'))}

def _iso_utc(raw: str | None) -> str | None:
    d = parse_date_utc(raw)
    return d.isoformat() if d else None

def summary_values(rec: dict) -> dict:
    md = rec.get('cveMetadata', {})
    containers = rec.get('containers', {})
    cna = containers.get('cna', {})
    metrics = cna.get('metrics', [])
    keys = {k for m in metrics for k in m}
    fv = field_values(rec)
    return {'state': md.get('state'), 'date_published_raw': md.get('datePublished'), 'date_reserved_raw': md.get('dateReserved'), 'date_updated_raw': md.get('dateUpdated'), 'date_rejected_raw': md.get('dateRejected'), 'assigner_org_id': md.get('assignerOrgId'), 'assigner_short_name': md.get('assignerShortName'), 'has_cvss_any': any((k.startswith('cvss') for k in keys)), 'has_cvss3': bool(keys & {'cvssV3_1', 'cvssV3_0'}), 'has_cvss4': 'cvssV4_0' in keys, 'has_cwe': bool(fv['cwe_problemtypes'] and 'CWE-' in fv['cwe_problemtypes']), 'has_problemtype_any': fv['cwe_problemtypes'] is not None, 'n_affected': len(cna.get('affected', [])), 'has_affected_product': fv['affected_product'] is not None, 'has_affected_versions': fv['affected_versions'] is not None, 'n_refs': len(cna.get('references', [])), 'has_description': bool(fv['description']), 'has_adp': 'adp' in containers, 'n_adp': len(containers.get('adp', []))}

def field_status(first: str | None, cur: str | None, *, failed: bool, rejected_now: bool, field: str) -> str:
    if failed:
        return 'reconstruction_failed'
    if rejected_now and field in CONTENT_FIELDS:
        return 'not_applicable'
    if first is None and cur is None:
        return 'absent_at_disclosure'
    if first is None:
        return 'added_after_disclosure'
    if cur is None:
        return 'removed_after_disclosure'
    return 'available_at_disclosure' if first == cur else 'modified_after_disclosure'

def parse_date_utc(raw: str | None) -> dt.datetime | None:
    if not raw:
        return None
    d = dt.datetime.fromisoformat(raw.replace('Z', '+00:00'))
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)

def load_record(data: bytes) -> dict:
    return json.loads(data.decode('utf-8'))

def reconstruct(repo: Path, cves: Iterable[str], head: str, hist: dict, added_per_commit: dict, head_path: dict[str, str]) -> tuple[list[dict], list[dict]]:
    cves = sorted(cves)
    cur = {}
    for spec, data in cat_file_batch(repo, (f'{head}:{head_path[c]}' for c in cves if c in head_path)):
        cur[CVE_PATH.search(spec).group(1)] = load_record(data) if data else None
    first: dict[str, tuple[str, int, dict] | None] = {}
    states_before: dict[str, list[str]] = defaultdict(list)
    pending = {c: 0 for c in cves if hist.get(c)}
    while pending:
        specs = []
        for c, i in pending.items():
            commit, _, status = hist[c][i]
            path = head_path.get(c)
            specs.append(f'{commit}:{path}' if status != 'D' and path else None)
        nxt = {}
        lookup = dict(zip(pending, specs))
        valid = [s for s in specs if s]
        contents = dict(cat_file_batch(repo, valid))
        for c, i in pending.items():
            s = lookup[c]
            data = contents.get(s) if s else None
            rec = load_record(data) if data else None
            st = rec.get('cveMetadata', {}).get('state') if rec else 'MISSING'
            if st == 'PUBLISHED':
                first[c] = (hist[c][i][0], hist[c][i][1], rec)
            else:
                states_before[c].append(st)
                if i + 1 < len(hist[c]):
                    nxt[c] = i + 1
                else:
                    first[c] = None
        pending = nxt
    rows, status_rows = ([], [])
    for c in cves:
        h = hist.get(c, [])
        f = first.get(c)
        cur_rec = cur.get(c)
        notes = []
        if not h:
            rstatus = 'no_history'
        elif f is None:
            rstatus = 'never_published_in_history'
        else:
            rstatus = 'reconstructed'
        frec = f[2] if f else None
        fs = summary_values(frec) if frec else {}
        cs = summary_values(cur_rec) if cur_rec else {}
        pub = parse_date_utc(fs.get('date_published_raw'))
        commit_at = dt.datetime.fromtimestamp(f[1], dt.timezone.utc) if f else None
        delay = (commit_at - pub).total_seconds() / 3600 if commit_at and pub else None
        if f and added_per_commit.get(f[0], 0) >= BULK_COMMIT_MIN_FILES:
            notes.append(f'first_published_commit_is_bulk_import({added_per_commit[f[0]]}_files)')
        if states_before.get(c):
            notes.append('states_before_first_published=' + ','.join(states_before[c]))
        if f and (not fs.get('date_published_raw')):
            notes.append('datePublished_missing_in_first_published_record')
        if f and cs.get('date_published_raw') and (fs.get('date_published_raw') != cs.get('date_published_raw')):
            notes.append('datePublished_changed_after_first_published')
        rows.append({'cve_id': c, 'reconstruction_status': rstatus, 'reconstruction_source': 'cvelistV5_git_first_published' if f else None, 'first_published_commit': f[0] if f else None, 'first_published_commit_at': commit_at, 'state_at_first_commit': states_before[c][0] if states_before.get(c) else 'PUBLISHED' if f else None, 'states_before_first_published': ','.join(states_before.get(c, [])) or None, 'cve_date_published': fs.get('date_published_raw'), 't_pub': pub.date() if pub else None, 'cve_date_reserved': fs.get('date_reserved_raw'), 'publication_commit_delay_hours': delay, 'first_commit_added_files': added_per_commit.get(f[0]) if f else None, 'current_record_commit': h[-1][0] if h else None, 'current_record_commit_at': dt.datetime.fromtimestamp(h[-1][1], dt.timezone.utc) if h else None, 'current_record_modified_at': cs.get('date_updated_raw'), 'current_state': cs.get('state'), 'current_date_published': cs.get('date_published_raw'), 'current_date_rejected': cs.get('date_rejected_raw'), 'history_commit_count': len(h), 'reconstruction_notes': ';'.join(notes) or None, **{f'disc_{k}': v for k, v in fs.items() if k.startswith(('has_', 'n_'))}, **{f'cur_{k}': v for k, v in cs.items() if k.startswith(('has_', 'n_'))}, 'disc_assigner_short_name': fs.get('assigner_short_name')})
        fv = field_values(frec) if frec else {}
        cv = field_values(cur_rec) if cur_rec else {}
        rejected_now = cs.get('state') == 'REJECTED'
        for fld in FIELDS:
            status_rows.append({'cve_id': c, 'field': fld, 'status': field_status(fv.get(fld), cv.get(fld), failed=frec is None, rejected_now=rejected_now, field=fld)})
    return (rows, status_rows)
