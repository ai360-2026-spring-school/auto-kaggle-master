# EDA notebook

## auto-EDA seed (ydata-profiling, minimal mode)
- profiled rows=50000 cols=16 duplicates=0 (full train rows=50000, target='PitNextLap')
### Alerts (ydata flags)
  - SKEWED on LapTime_Delta
  - UNIQUE on id
  - ZEROS on PitStop
  - ZEROS on LapTime_Delta
  - ZEROS on Cumulative_Degradation
  - ZEROS on Position_Change
  - ZEROS on PitNextLap
### Target ('PitNextLap')
  - type=Numeric n_unique=? mean=0.2006 min=0 max=1 std=0.4005 skewness=1.495 kurtosis=0.2357
### Highest-cardinality columns
  - id: type=Numeric n_unique=50000 missing=0.0%
  - Cumulative_Degradation: type=Numeric n_unique=26500 missing=0.0%
  - LapTime_Delta: type=Numeric n_unique=12321 missing=0.0%
  - LapTime (s): type=Numeric n_unique=8582 missing=0.0%
  - RaceProgress: type=Numeric n_unique=207 missing=0.0%
  - Driver: type=Text n_unique=79 missing=0.0%
  - TyreLife: type=Numeric n_unique=5 missing=0.0%
  - Compound: type=Text n_unique=0 missing=0.0%
  - Race: type=Text n_unique=0 missing=0.0%
  - Year: type=Numeric n_unique=0 missing=0.0%

---
## Agent observations
