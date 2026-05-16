# EDA notebook

## auto-EDA seed (ydata-profiling, minimal mode)
- profiled rows=5000 cols=16 duplicates=0 (full train rows=5000, target='PitNextLap')
### Alerts (ydata flags)
  - UNIQUE on id
  - ZEROS on PitStop
  - ZEROS on LapTime_Delta
  - ZEROS on Cumulative_Degradation
  - ZEROS on Position_Change
  - ZEROS on PitNextLap
### Target ('PitNextLap')
  - type=Numeric n_unique=? mean=0.1978 min=0 max=1 std=0.3984 skewness=1.518 kurtosis=0.3037
### Highest-cardinality columns
  - id: type=Numeric n_unique=5000 missing=0.0%
  - Cumulative_Degradation: type=Numeric n_unique=4356 missing=0.0%
  - LapTime (s): type=Numeric n_unique=3426 missing=0.0%
  - LapTime_Delta: type=Numeric n_unique=3049 missing=0.0%
  - RaceProgress: type=Numeric n_unique=244 missing=0.0%
  - Driver: type=Text n_unique=65 missing=0.0%
  - TyreLife: type=Numeric n_unique=5 missing=0.0%
  - Position_Change: type=Numeric n_unique=3 missing=0.0%
  - LapNumber: type=Numeric n_unique=2 missing=0.0%
  - Stint: type=Numeric n_unique=1 missing=0.0%

---
## Agent observations

- [iter 0] The 'id' column is a unique identifier and should be dropped to prevent leakage.

- [iter 2] The 'id' column is a unique identifier and should be dropped to prevent leakage.
