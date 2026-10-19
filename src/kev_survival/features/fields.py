from __future__ import annotations
import json
import re
NA = {'', 'n/a', 'na', '-', 'unspecified', 'unknown', 'n/a n/a', 'none'}
CWE_RE = re.compile('(?<![A-Za-z-])CWE-(\\d+)')
V3_UI = {'N': 'N', 'R': 'R'}
V4_UI = {'N': 'N', 'P': 'R', 'A': 'R'}

def _ok(v) -> bool:
    return isinstance(v, str) and v.strip().lower() not in NA

def vendor_key(v) -> str | None:
    if not _ok(v):
        return None
    k = re.sub('[^a-z0-9]', '', v.lower())
    return k or None

def _vectors(x, out: list) -> list:
    if isinstance(x, dict):
        vs = x.get('vectorString')
        if isinstance(vs, str) and vs.startswith(('CVSS:3.1/', 'CVSS:3.0/', 'CVSS:4.0/')):
            out.append((vs, x.get('baseScore')))
        for v in x.values():
            _vectors(v, out)
    elif isinstance(x, list):
        for v in x:
            _vectors(v, out)
    return out

def cvss_from(obj) -> dict:
    found = _vectors(obj, [])
    for pref in ('CVSS:3.1/', 'CVSS:3.0/', 'CVSS:4.0/'):
        for vs, score in found:
            if not vs.startswith(pref):
                continue
            m = dict((p.split(':', 1) for p in vs.split('/')[1:] if ':' in p))
            try:
                score = float(score) if score is not None else None
            except (TypeError, ValueError):
                score = None
            if pref == 'CVSS:4.0/':
                d = {'cvss_av': m.get('AV'), 'cvss_ac': m.get('AC'), 'cvss_pr': m.get('PR'), 'cvss_ui': V4_UI.get(m.get('UI')), 'cvss_s': 'NOT_APPLICABLE_V4', 'cvss_c': m.get('VC'), 'cvss_i': m.get('VI'), 'cvss_a': m.get('VA')}
            else:
                d = {'cvss_av': m.get('AV'), 'cvss_ac': m.get('AC'), 'cvss_pr': m.get('PR'), 'cvss_ui': V3_UI.get(m.get('UI')), 'cvss_s': m.get('S'), 'cvss_c': m.get('C'), 'cvss_i': m.get('I'), 'cvss_a': m.get('A')}
            return {**d, 'cvss_base_score': score, 'cvss_version': pref[5:8], 'cvss_vector': vs}
    return {'cvss_version': None, 'cvss_vector': None, 'cvss_base_score': None}

def first_cwe(obj) -> str | None:
    m = CWE_RE.search(json.dumps(obj or []))
    return f'CWE-{m.group(1)}' if m else None

def parse_v5_cna(rec: dict) -> dict:
    cna = (rec.get('containers') or {}).get('cna') or {}
    aff = [a for a in cna.get('affected') or [] if isinstance(a, dict)]
    vendors = [a.get('vendor') for a in aff if _ok(a.get('vendor'))]
    prods = {a.get('product').strip().lower() for a in aff if _ok(a.get('product'))}
    descs = [d for d in cna.get('descriptions') or [] if isinstance(d, dict) and _ok(d.get('value'))]
    en = [d['value'] for d in descs if str(d.get('lang', '')).lower().startswith('en')] or [d['value'] for d in descs]
    refs = [r for r in cna.get('references') or [] if isinstance(r, dict) and _ok(r.get('url'))]
    return {'vendor_raw': vendors[0] if vendors else None, 'vendor_key': vendor_key(vendors[0]) if vendors else None, 'affected_present': int(bool(vendors or prods)), 'n_affected_products': len(prods), **cvss_from(cna.get('metrics') or []), 'cwe_id': first_cwe(cna.get('problemTypes')), 'description': en[0] if en else '', 'n_refs': len(refs)}

def parse_v4(rec: dict) -> dict:
    vd = [v for v in ((rec.get('affects') or {}).get('vendor') or {}).get('vendor_data') or [] if isinstance(v, dict)]
    vendors = [v.get('vendor_name') for v in vd if _ok(v.get('vendor_name'))]
    prods = {p.get('product_name').strip().lower() for v in vd for p in (v.get('product') or {}).get('product_data') or [] if isinstance(p, dict) and _ok(p.get('product_name'))}
    dd = [d for d in (rec.get('description') or {}).get('description_data') or [] if isinstance(d, dict)]
    texts = [d.get('value') for d in dd if _ok(d.get('value')) and '** RESERVED **' not in d.get('value')]
    en = [d.get('value') for d in dd if str(d.get('lang', '')).lower().startswith('en') and d.get('value') in texts] or texts
    refs = [r for r in (rec.get('references') or {}).get('reference_data') or [] if isinstance(r, dict) and _ok(r.get('url'))]
    return {'vendor_raw': vendors[0] if vendors else None, 'vendor_key': vendor_key(vendors[0]) if vendors else None, 'affected_present': int(bool(vendors or prods)), 'n_affected_products': len(prods), **cvss_from(rec.get('impact') or {}), 'cwe_id': first_cwe(rec.get('problemtype')), 'description': en[0] if en else '', 'n_refs': len(refs)}

def parse_nvd(cve: dict) -> dict:
    metrics = cve.get('metrics') or {}
    cv = {}
    for key, pref in (('cvssMetricV31', 'CVSS:3.1/'), ('cvssMetricV30', 'CVSS:3.0/'), ('cvssMetricV40', 'CVSS:4.0/')):
        ms = metrics.get(key) or []
        ms = sorted(ms, key=lambda m: m.get('type') != 'Primary')
        if ms:
            cv = cvss_from([m.get('cvssData', {}) for m in ms[:1]])
            if cv.get('cvss_version'):
                break
    if not cv.get('cvss_version'):
        cv = {'cvss_version': None, 'cvss_vector': None, 'cvss_base_score': None}
    cwe = first_cwe([w.get('description') for w in cve.get('weaknesses') or []])
    cpes = set()
    for conf in cve.get('configurations') or []:
        for node in conf.get('nodes') or []:
            for m in node.get('cpeMatch') or []:
                parts = str(m.get('criteria', '')).split(':')
                if len(parts) > 5 and m.get('vulnerable', True):
                    cpes.add((parts[3], parts[4]))
    descs = [d.get('value') for d in cve.get('descriptions') or [] if d.get('lang') == 'en']
    return {'affected_present': int(bool(cpes)), 'n_affected_products': len({p for _, p in cpes}), **cv, 'cwe_id': cwe, 'description': descs[0] if descs else '', 'n_refs': len(cve.get('references') or [])}
