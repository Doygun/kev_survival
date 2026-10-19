from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kev_survival.io.common import ROOT, load_project, raw_dir, sha256_file, utc_now
SOURCES = {'cvelistV5': {'official_url': 'https://github.com/CVEProject/cvelistV5', 'format': 'git repository, CVE JSON 5.x', 'method': 'git clone --no-checkout (tam geçmiş, sığ değil)', 'license': 'CVE Program Terms of Use (https://www.cve.org/Legal/TermsOfUse)'}, 'cvelist_json4': {'official_url': 'https://github.com/CVEProject/cvelist', 'format': 'git repository, CVE JSON 4.0', 'method': 'git clone --no-checkout (tam geçmiş); yalnızca kapsam denetimi', 'license': 'CVE Program Terms of Use'}, 'cisa_kev': {'official_url': 'https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json', 'format': 'JSON', 'method': 'HTTPS GET (tek dosya)', 'license': 'CC0 1.0 (cisagov/kev-data LICENSE)'}, 'nvd_cves': {'official_url': 'https://services.nvd.nist.gov/rest/json/cves/2.0', 'format': 'JSON (API 2.0), sayfa başına gzip ile saklandı', 'method': 'HTTPS GET, anahtarsız, istekler arası 6,5 s, pubStartDate/pubEndDate 119 günlük pencereler, resultsPerPage=2000', 'license': 'NVD kullanım koşulları (kamu malı; atıf notu istenir)'}, 'nvd_cvehistory': {'official_url': 'https://services.nvd.nist.gov/rest/json/cvehistory/2.0', 'format': 'JSON (API 2.0), CVE başına gzip ile saklandı', 'method': 'HTTPS GET, cveId ile temsil edici örneklem (data/interim/s01_nvd_history_sample.json), anahtarsız, 6,5 s', 'license': 'NVD kullanım koşulları'}, 'epss': {'official_url': 'https://epss.empiricalsecurity.com/epss_scores-YYYY-MM-DD.csv.gz', 'format': 'günlük CSV (gzip); v2+ dosyalarında #model_version,score_date başlığı', 'method': 'HTTPS GET, günlük dosyalar 2021-11-01 – 2025-01-07', 'license': 'FIRST EPSS SIG; skorlar topluluğa ücretsiz yayımlanır (açık lisans metni bulunamadı)'}, 'exploitdb': {'official_url': 'https://gitlab.com/exploit-database/exploitdb', 'format': 'files_exploits.csv (yalnızca metadata) ve README', 'method': "Belirli commit'ten raw dosya indirme; exploit kodu indirilmedi (kullanıcı talimatı)", 'license': 'GPL-2.0-or-later (GitLab proje lisansı)'}}
EARLY_S2_OUTPUTS = {'json4_first_added_s02_20261006.parquet', 's02_json4_coverage_20261006.json'}

def git_info(repo: Path) -> dict:

    def g(*a):
        return subprocess.check_output(['git', '-C', str(repo), *a]).decode().strip()
    packs = sorted((repo / '.git' / 'objects' / 'pack').glob('*.pack'))
    return {'head_commit': g('rev-parse', 'HEAD'), 'head_commit_date': g('log', '-1', '--format=%cI'), 'first_commit_date': g('log', '--format=%cI', '-1', g('rev-list', '--max-parents=0', 'HEAD').split()[0]), 'n_commits': int(g('rev-list', '--count', 'HEAD')), 'pack_files': [{'path': str(p.relative_to(ROOT)).replace('\\', '/'), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in packs]}

def main() -> None:
    cfg = load_project()
    base = raw_dir()
    status = {}
    log = ROOT / 'logs' / 'download_status.jsonl'
    if log.exists():
        for line in log.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            status[r['source']] = r
    m = {'t_freeze': str(cfg['t_freeze']), 'manifest_created_utc': utc_now(), 'sources': {}}
    for name, meta in SOURCES.items():
        d = base / name
        entry = {**meta, 'local_dir': str(d.relative_to(ROOT)).replace('\\', '/'), 'present': d.exists()}
        if (d / '.git').exists():
            entry.update(git_info(d))
            entry['total_bytes'] = sum((p['bytes'] for p in entry['pack_files']))
        elif d.exists():
            files = sorted((p for p in d.rglob('*') if p.is_file()))
            entry['files'] = [{'path': str(p.relative_to(ROOT)).replace('\\', '/'), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in files]
            entry['n_files'] = len(files)
            entry['total_bytes'] = sum((f['bytes'] for f in entry['files']))
        if name in status:
            entry['download_status'] = status[name]
        if name == 'epss' and 'epss_github_archive_check' in status:
            entry['missing_days_final'] = status['epss'].get('missing_days', [])
            entry['official_archive_check'] = status['epss_github_archive_check']
        m['sources'][name] = entry
    m['total_raw_bytes'] = sum((s.get('total_bytes', 0) for s in m['sources'].values()))
    derived = []
    for sub in ('data/interim', 'data/processed', 'results/tables', 'results/metrics', 'results/figures'):
        for p in sorted((ROOT / sub).rglob('*')):
            if p.is_file() and p.name != '.gitkeep':
                rec = {'path': str(p.relative_to(ROOT)).replace('\\', '/'), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                if p.name in EARLY_S2_OUTPUTS:
                    rec['status'] = "early_unverified_s2_output (S1 sırasında üretildi; geçerli S2 sonucu değildir; S2'de *_rerun adıyla yeniden üretildi, bkz. reports/stage_S2_report.md)"
                derived.append(rec)
    m['derived_files'] = derived
    (ROOT / 'data' / 'manifest.json').write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding='utf-8')
    print({k: (v.get('present'), v.get('total_bytes'), v.get('n_files'), v.get('head_commit')) for k, v in m['sources'].items()}, m['total_raw_bytes'])
if __name__ == '__main__':
    main()
