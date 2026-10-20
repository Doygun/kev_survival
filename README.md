# kev_survival

Code for the study *Predicting Time to CISA KEV Listing from Information Available at CVE Disclosure: A Temporally Gated Survival Analysis*.

The pipeline estimates, from the information available on the day a CVE was published, the time until the CVE is added to the CISA Known Exploited Vulnerabilities (KEV) catalog. The event of interest is KEV listing (the KEV `dateAdded` field), not first exploitation.

## Layout

| Path | Content |
|---|---|
| `src/kev_survival/io` | Download, logging and manifest helpers |
| `src/kev_survival/reconstruct` | Disclosure-time CVE record reconstruction from the cvelistV5 Git history and the JSON 4 archive |
| `src/kev_survival/events`, `split` | Event table, censoring rules, temporal splits |
| `src/kev_survival/features` | Disclosure-time and current-record feature matrices with provenance |
| `src/kev_survival/models`, `eval` | Elastic-net Cox and logistic models; Uno C, IPCW Brier/IBS, calibration, recall at budget, paired bootstrap |
| `scripts/s01_*.py` … `scripts/s14_*.py` | Stage scripts, run in order: download and freeze, reconstruction, cohort, splits, features, models, evaluation, sensitivity analyses, final model, reproducibility audit |

## Data sources

Raw data are not included. The scripts download them from the public sources below; users must comply with each source's terms.

| Source | Role |
|---|---|
| CVE Program: cvelistV5 and cvelist (JSON 4) repositories | Disclosure-time record and publication date |
| NVD CVE API 2.0 and Change History API | Current record (leakage check) and date checks. This product uses the NVD API but is not endorsed or certified by the NVD. |
| CISA Known Exploited Vulnerabilities catalog (CC0 1.0) | Event date |
| FIRST EPSS daily archive | External comparison only |
| Exploit-DB `files_exploits.csv` metadata | Feasibility check only |

Derived data files and result tables are available from the author on request.

## Running

Python 3.13 and [uv](https://github.com/astral-sh/uv):

```bash
uv sync
.venv/Scripts/python scripts/s01_download.py
```

Then run the remaining stage scripts in numerical order. Configuration files are not part of this repository; the scripts expect them under `configs/` (contact the author).

## License

MIT (`LICENSE`).
