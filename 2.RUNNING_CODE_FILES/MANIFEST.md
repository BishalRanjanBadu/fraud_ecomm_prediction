# fraud_nb01-07_v2 — package manifest

**Marker:** `# MARKER: fraud_nb01-07_v2 :: <notebook>` is the first line of every setup cell.
This package supersedes `fraud_nb05-07_v1` and its separate PATCH_02; the fix now lives inside notebook 02.

```
fraud_nb01-07_v2/
├── MANIFEST.md
└── notebooks/
    ├── 01_Data_Loading_and_First_Look.ipynb     34 cells
    ├── 02_Data_Cleaning.ipynb                   24 cells
    ├── 03_Missing_Values_and_Outliers.ipynb     19 cells
    ├── 04_Statistics_and_EDA.ipynb              24 cells
    ├── 05_Hypothesis_Testing.ipynb              11 cells
    ├── 06_Correlation_VIF_Encoding.ipynb        11 cells
    └── 07_Model_Building_and_Evaluation.ipynb   15 cells
```

## Run order (Colab, Secrets `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`)

Run **01 → 02 → 03 → 04 → 05 → 06 → 07**, top to bottom, **one notebook per Colab session**.

**Memory.** Projected peak memory on the real data is about 4–5.5 GB per notebook, which fits the free tier only when notebooks do not share a runtime.

**Stop conditions:**
* **05:** the last line prints `SENTINELS:`, and every value must be `False`. Notebook 06 refuses to run otherwise.
* **07:** the search takes roughly 30–60 minutes. If the runtime disconnects, run all again: finished model families are reused.

**Send back:** all seven executed `.ipynb` files (with outputs), `reports/07_run_record.json`, and the `model_card.md` from the printed `models/v_...` prefix.

## What changed from your uploaded 01–04

| Notebook | Fix | Class |
|---|---|---|
| all | Shared setup: flat S3 layout baked in (the `dev/` auto-detection is removed), `boto3==1.43.95` pinned, unpinned `pyarrow>=14` install removed, library versions and drift recorded in every run record, landing objects checked before any work | config / provenance |
| all | Parquet writes verify a byte-level round trip (columns, dtypes, categories) before upload | silent corruption |
| 01 | Conflicting-PK resolution uses a stable sort, so the surviving row no longer depends on quicksort tie order | reproducibility |
| 01 | Label-table ids are whitespace-stripped before matching; the notebook now prints that no audited payment was alerted | correctness check |
| 01 | Text corrected: the audit samples **non-alerted** payments only; group overlap across the cut is the production condition, and the controls are chronological CV plus no identifier features | documentation was wrong |
| 02 | **`device_n_customers` / `ip_n_customers` are now point-in-time**: an as-of distinct count at or before each payment. v1 used full-period counts, a live leak that also spanned test rows. `velocity_fix` in the run record reports how many rows changed | **leakage (live)** |
| 02 | `home_region_code` comes from the authoritative customer column (non-null per contract) instead of a nullable pincode prefix. Agreement between the two is printed | data quality |
| 02 | Explicit canonical spellings for `iPhone`, `iOS`, `macOS`, `WebView` and `RuPay`. Title-case had produced `Iphone`/`Ios`/`Rupay`, the same bug class 02 already warned about for UPI | canonicalisation |
| 02 | Level-set assertions for device, OS, browser, network, city, email and shipping speed. New spelling drift fails the notebook instead of creating a new category | fail-closed |
| 02 | Diagnostic crosstab for `ip_region_code == 99`. Values are unchanged pending evidence of what 99 means | verify before changing |
| 02 | Markdown claim "nothing crosses rows" corrected | documentation was wrong |
| 03 | The `-1` region fill is recorded under `sentinel_fill`, not `global_medians` | mislabelled artifact |
| 04 | Hand-off notes now describe what notebook 05 actually does | documentation drift |

Notebooks 05–07 are unchanged in logic from v1; only their markers and wording were updated.

## One decision still open

**The notebook 07 gate is design-weighted.** The alert-review stratum and the audit stratum are each weighted by their inverse sampling fraction, and audit-only PR-AUC is reported as a supporting line. If you want audit-only instead, say so **before** running 07.

## Verification performed before packaging

**Build checks**
* Every source line except each cell's last ends with `\n`, checked on disk.
* `nbformat.validate` passes.
* On-disk source is identical to the executed source.
* No bare `except`, no warning suppression, no explicit keys, no stale v1 names.

**Execution**
* All seven notebooks were executed from these `.ipynb` files, one process per notebook, in the Colab-matched stack: Python 3.13, pandas 2.2.3, numpy 2.1.3, pyarrow 23.0.1, scipy 1.16.3, statsmodels 0.15.0, scikit-learn 1.6.1, lightgbm 4.6.0, xgboost 3.4.1, boto3 1.43.95.
* The input was a raw landing mock of all 12 CSVs with your contract's exact columns and these defects: four timestamp formats, sentinels, padding, exact and PK duplicates, `PAYR` retries, `_DUP` identities, bad and test amounts, spelling drift, the 255 overflow, orphan devices, and the two-stratum review design.
* Notebook 01 reproduces the real 33-column stage layout, and 04 the 110-column layout.

**Failure paths**
* All three model families were forced through selection, explanation and persistence.
* CV resume was exercised.
* With full-period velocity columns, 05 flags them and 06 stops.

**Expected warnings (harmless)**
* A scipy `disp`/`iprint` DeprecationWarning raised inside scikit-learn's LogisticRegression.
* Matplotlib deprecation notices from pandas' date-axis plotting in 04, depending on Colab's Matplotlib version.
