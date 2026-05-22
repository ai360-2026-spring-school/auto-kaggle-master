# EDA notebook

## auto-EDA seed (ydata-profiling, minimal mode)
- profiled rows=100000 cols=16 duplicates=0 (full train rows=439140, target='PitNextLap')
### Alerts (ydata flags)
  - SKEWED on LapTime (s)
  - SKEWED on LapTime_Delta
  - UNIQUE on id
  - ZEROS on PitStop
  - ZEROS on LapTime_Delta
  - ZEROS on Cumulative_Degradation
  - ZEROS on Position_Change
  - ZEROS on PitNextLap
### Target ('PitNextLap')
  - type=Numeric n_unique=? mean=0.2001 min=0 max=1 std=0.4001 skewness=1.499 kurtosis=0.2468
### Highest-cardinality columns
  - id: type=Numeric n_unique=100000 missing=0.0%
  - Cumulative_Degradation: type=Numeric n_unique=39800 missing=0.0%
  - LapTime_Delta: type=Numeric n_unique=15525 missing=0.0%
  - LapTime (s): type=Numeric n_unique=8991 missing=0.0%
  - RaceProgress: type=Numeric n_unique=225 missing=0.0%
  - Driver: type=Text n_unique=61 missing=0.0%
  - LapNumber: type=Numeric n_unique=1 missing=0.0%
  - TyreLife: type=Numeric n_unique=1 missing=0.0%
  - Position_Change: type=Numeric n_unique=1 missing=0.0%
  - Compound: type=Text n_unique=0 missing=0.0%

---
## Agent observations

- [iter 2] Smoothed target encoding for Driver, Race, Compound (alpha=10) may capture driver/race-specific pit propensity beyond frequency. Will add numeric TE features alongside existing encodings.
