from __future__ import annotations
import json
import pickle
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sksurv.linear_model import CoxnetSurvivalAnalysis
from kev_survival.eval import metrics as M
from kev_survival.features.matrix import TERMS, Encoder
from kev_survival.io.access import read_parquet
from kev_survival.io.common import ROOT, load_project, sha256_file, utc_now
warnings.filterwarnings('ignore', category=FutureWarning)
CFG = load_project()
assert CFG['locked'] is True
FCFG = yaml.safe_load(open(ROOT / 'configs' / 'features.yaml', encoding='utf-8'))
MCFG = yaml.safe_load(open(ROOT / 'configs' / 'models.yaml', encoding='utf-8'))['cox']
SNAP = str(CFG['t_freeze']).replace('-', '')
PROC, LAB = (ROOT / 'data' / 'processed', ROOT / 'data' / 'processed' / 'labels')
OUT = ROOT / 'results' / 'models' / 'final_artifact'
STAGE, PHASE = ('S14', 'final_artifact')
README = '# Nihai yeniden eğitilmiş model (dağıtım artefaktı)\n\nBu dizindeki model, ana kohortun tamamı (eğitim + doğrulama + test dönemleri, etiket kesimi t_freeze)\nüzerinde, S9\'da kilitlenen hiperparametrelerle yeniden eğitilmiş elastic-net Cox modelidir.\n\n**Test performansı bu modelden üretilmemiştir.** Makalede bildirilen bütün sonuçlar, yalnızca eğitim ve\ndoğrulama dönemleriyle eğitilen `results/models/cox_disclosure_B_<snap>.pkl` modeliyle 2024 test kümesinde\nelde edilmiştir. Bu dizindeki model test verisini gördüğü için hiçbir değerlendirme ölçütü için kullanılamaz.\n\nDosyalar:\n- `cox_disclosure_final_B_<snap>.pkl`: {"model", "alpha", "columns", "encoder", "hyperparameters_from", "note"}\n- `final_model_<snap>.json`: eğitim kümesi sayıları, sıfırdan farklı katsayı sayısı, SHA-256 özetleri\n- Katsayılar: `results/tables/s14_final_model_coefficients_<snap>.csv`\n\nKullanım: ham 27 terim (`kev_survival.features.matrix.raw_terms`) → `encoder` parametreleriyle kodlama →\n`model.predict(X, alpha=alpha)` doğrusal öngörücü; `model.predict_survival_function(X, alpha=alpha)` hayatta kalma.\n'

def main() -> None:
    t0 = utc_now()
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.load(open(ROOT / 'results' / 'metrics' / f'cox_selection_{SNAP}.json', encoding='utf-8'))['selection']['disclosure']
    raw = read_parquet(PROC / f'features_raw_disclosure_{SNAP}.parquet', STAGE, PHASE).set_index('cve_id')
    y = read_parquet(LAB / f'labels_finalmodel_B_{SNAP}.parquet', STAGE, PHASE)
    ids = y['cve_id']
    enc = Encoder(FCFG['encoding']['cwe_top_group_levels']).fit(raw.loc[ids, TERMS])
    X = enc.transform(raw.loc[ids, TERMS])
    m = CoxnetSurvivalAnalysis(l1_ratio=sel['l1_ratio'], alphas=sel['alpha_path'], fit_baseline_model=True, max_iter=MCFG['max_iter']).fit(X.values, M.surv_y(y['event'], y['duration_days']))
    j = int(np.argmin(np.abs(np.asarray(m.alphas_) - sel['alpha'])))
    coef = pd.DataFrame({'model': 'cox_disclosure_final_B', 'column': X.columns, 'coef': m.coef_[:, j]})
    coef.to_csv(ROOT / 'results' / 'tables' / f's14_final_model_coefficients_{SNAP}.csv', index=False)
    pkl = OUT / f'cox_disclosure_final_B_{SNAP}.pkl'
    with open(pkl, 'wb') as f:
        pickle.dump({'model': m, 'alpha': sel['alpha'], 'columns': list(X.columns), 'encoder': enc.params(), 'hyperparameters_from': f'cox_selection_{SNAP}.json#selection.disclosure (B)', 'note': 'Dağıtım artefaktı; test performansı bu modelden üretilmemiştir.'}, f)
    (OUT / 'README.md').write_text(README.replace('<snap>', SNAP), encoding='utf-8')
    meta = {'started': t0, 'finished': utc_now(), 'snapshot_id': SNAP, 'training_labels': f'labels_finalmodel_B_{SNAP}.parquet', 't_cutoff': str(CFG['t_freeze']), 'n_train': int(len(y)), 'events_train': int(y['event'].sum()), 'l1_ratio': sel['l1_ratio'], 'alpha': sel['alpha'], 'n_columns': int(X.shape[1]), 'n_nonzero': int((coef['coef'] != 0).sum()), 'zero_time_offset_applied': False, 'test_performance_from_this_model': False, 'model_sha256': sha256_file(pkl)}
    (OUT / f'final_model_{SNAP}.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    (ROOT / 'results' / 'metrics' / f's14_final_model_{SNAP}.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in meta.items() if k not in ('started', 'finished')}, indent=1))
if __name__ == '__main__':
    main()
