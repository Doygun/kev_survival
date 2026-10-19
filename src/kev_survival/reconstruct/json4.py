from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
from kev_survival.reconstruct import cvelist as cl
NA = {'', 'n/a', 'N/A', 'na', '-', 'unspecified'}

def _ok(v) -> bool:
    return isinstance(v, str) and v.strip() not in NA

def fields_v4(r: dict) -> dict:
    desc = [d.get('value', '') for d in r.get('description', {}).get('description_data', [])]
    impact = r.get('impact')
    pts = json.dumps(r.get('problemtype', {}))
    vendors = r.get('affects', {}).get('vendor', {}).get('vendor_data', []) or []
    prods = [p for v in vendors for p in v.get('product', {}).get('product_data', []) or []]
    vers = [x for p in prods for x in p.get('version', {}).get('version_data', []) or []]
    refs = r.get('references', {}).get('reference_data', []) or []
    has_cvss = False
    if impact:
        s = json.dumps(impact)
        has_cvss = 'vectorString' in s or 'baseScore' in s or 'CVSS:' in s
    return {'v4_has_description': any((_ok(d) and '** RESERVED **' not in d for d in desc)), 'v4_has_cvss': has_cvss, 'v4_has_cwe': 'CWE-' in pts, 'v4_has_vendor': any((_ok(v.get('vendor_name')) for v in vendors)), 'v4_has_product': any((_ok(p.get('product_name')) for p in prods)), 'v4_has_version': any((_ok(x.get('version_value')) for x in vers)), 'v4_has_references': any((_ok(x.get('url')) for x in refs)), 'v4_date_public_meta': r.get('CVE_data_meta', {}).get('DATE_PUBLIC')}

def fields_v5_na_excluded(r: dict) -> dict:
    cna = r.get('containers', {}).get('cna', {})
    aff = cna.get('affected', []) or []
    vers = [v for a in aff for v in a.get('versions') or []]
    pts = json.dumps(cna.get('problemTypes', []))
    return {'v5_has_description': any((_ok(d.get('value')) for d in cna.get('descriptions', []))), 'v5_has_cvss': any((k.startswith('cvss') for m in cna.get('metrics', []) for k in m)), 'v5_has_cwe': 'CWE-' in pts, 'v5_has_vendor': any((_ok(a.get('vendor')) for a in aff)), 'v5_has_product': any((_ok(a.get('product')) for a in aff)), 'v5_has_version': any((_ok(v.get('version')) for v in vers)), 'v5_has_references': any((_ok(x.get('url')) for x in cna.get('references', [])))}

def first_public(repo: Path, cves: set[str]) -> tuple[dict[str, dict], dict[str, int]]:
    hist, _ = cl.file_histories(repo, cves, pathspec='.')
    paths = cl.head_paths(repo, pathspec='.')
    found: dict[str, dict] = {}
    bad: dict[str, int] = {}
    pending = {c: 0 for c in sorted(cves) if hist.get(c) and c in paths}
    while pending:
        specs = {c: f'{hist[c][i][0]}:{paths[c]}' for c, i in pending.items() if hist[c][i][2] != 'D'}
        contents = dict(cl.cat_file_batch(repo, specs.values()))
        nxt = {}
        for c, i in pending.items():
            data = contents.get(specs.get(c))
            try:
                r = json.loads(data.decode('utf-8', 'replace'), strict=False) if data else {}
            except json.JSONDecodeError:
                r, bad[c] = ({}, bad.get(c, 0) + 1)
            if not isinstance(r, dict) or not isinstance(r.get('CVE_data_meta'), dict):
                r = {}
            if r.get('CVE_data_meta', {}).get('STATE') == 'PUBLIC':
                commit, ct, _ = hist[c][i]
                found[c] = {'json4_commit': commit, 'json4_public_at': dt.datetime.fromtimestamp(ct, dt.timezone.utc), **fields_v4(r)}
            elif i + 1 < len(hist[c]):
                nxt[c] = i + 1
        pending = nxt
    return (found, bad)

def _find_base_scores(x) -> list[float]:
    out: list[float] = []
    if isinstance(x, dict):
        for k, v in x.items():
            if k == 'baseScore':
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    pass
            else:
                out += _find_base_scores(v)
    elif isinstance(x, list):
        for v in x:
            out += _find_base_scores(v)
    return out

def detail_v4(r: dict) -> dict:
    desc = [d.get('value', '') for d in r.get('description', {}).get('description_data', []) or [] if isinstance(d, dict)]
    text = ' '.join((d for d in desc if _ok(d)))
    refs = r.get('references', {}).get('reference_data', []) or []
    scores = _find_base_scores(r.get('impact'))
    return {'v4_desc_len_words': len(text.split()), 'v4_cvss_base_score': max(scores) if scores else None, 'v4_n_refs': sum((1 for x in refs if isinstance(x, dict) and _ok(x.get('url')))), 'v4_assigner': r.get('CVE_data_meta', {}).get('ASSIGNER')}
