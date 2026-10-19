from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
from kev_survival.io.common import ROOT, utc_now
LOG = ROOT / 'logs' / 'file_access.jsonl'

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def log(path: Path, mode: str, stage: str, phase: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    rec = {'time': utc_now(), 'stage': stage, 'phase': phase, 'mode': mode, 'run_context': os.environ.get('KEV_RUN_CONTEXT', 'primary'), 'path': str(Path(path).resolve().relative_to(ROOT)).replace('\\', '/'), 'sha256': _sha(Path(path))}
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')

def read_parquet(path: Path, stage: str, phase: str, **kw) -> pd.DataFrame:
    df = pd.read_parquet(path, **kw)
    log(path, 'read', stage, phase)
    return df

def write_parquet(df: pd.DataFrame, path: Path, stage: str, phase: str) -> None:
    df.to_parquet(path, index=False)
    log(path, 'write', stage, phase)
