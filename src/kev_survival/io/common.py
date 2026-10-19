from __future__ import annotations
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
import requests
import yaml
ROOT = Path(__file__).resolve().parents[3]
USER_AGENT = 'kev_survival-research/0.1 (academic feasibility study)'

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def load_project() -> dict:
    with open(ROOT / 'configs' / 'project.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)

def raw_dir() -> Path:
    t_freeze = load_project()['t_freeze']
    if not t_freeze:
        raise RuntimeError('t_freeze configs/project.yaml içinde tanımlı değil')
    return ROOT / 'data' / 'raw' / str(t_freeze)

def sha256_file(path: Path, chunk: int=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while (block := f.read(chunk)):
            h.update(block)
    return h.hexdigest()

def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

def http_get(url: str, params: dict | None, log_path: Path, *, retries: int=3, backoff: float=10.0, timeout: float=180.0, stream: bool=False) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        t0 = utc_now()
        try:
            r = requests.get(url, params=params, timeout=timeout, stream=stream, headers={'User-Agent': USER_AGENT})
            append_jsonl(log_path, {'time_utc': t0, 'url': url, 'params': params, 'status': r.status_code, 'attempt': attempt, 'message': r.headers.get('message')})
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff * 2 ** attempt)
                continue
            r.raise_for_status()
            raise requests.HTTPError(f"HTTP {r.status_code}: {r.headers.get('message')}")
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            append_jsonl(log_path, {'time_utc': t0, 'url': url, 'params': params, 'status': None, 'attempt': attempt, 'error': repr(exc)})
            if attempt < retries:
                time.sleep(backoff * 2 ** attempt)
                continue
    raise RuntimeError(f'İndirme başarısız: {url} {params}') from last_exc

def write_new(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + '.part')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    tmp.rename(path)
