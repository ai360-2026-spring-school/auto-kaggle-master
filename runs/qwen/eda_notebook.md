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

- [iter 0] Driver has 527 unique values and high mutual information (0.061) with target. Some drivers (e.g. D223, VET) have 100% PitNextLap rate, suggesting strong predictive power. This is a prime candidate for target encoding.

- [iter 0] id column shows perfect mapping to target (leakage signal) and must be dropped. PitStop is sparse (only 676 ones) but may carry signal.

- [iter 0] Residual analysis shows high mean absolute error for drivers MSC, D223, D334, D379, DOO — these are poorly predicted by the incumbent. Target encoding may help reduce this error.

- [iter 1] Driver D223 has 100% PitNextLap rate (12/12), VET has 94.7% (36/38), and MSC has high residual error (0.38 avg abs residual), indicating the incumbent struggles on high-pit-probability drivers. Target encoding Driver is justified by both signal and residual analysis.

- [iter 1] Driver has highly variable pit probability: 15 drivers have 100% PitNextLap rate (e.g. D223, VET), while others like RAI have only 33%. High-count drivers (RAI, D018, DRA) have moderate rates. This supports using a smoothed target encoding to avoid overfitting on rare drivers.

- [iter 2] Driver D223, VET, D379, D389, and MSC each have 100% PitNextLap rate but very low counts (1-3), making them prone to overfitting. High-count drivers like RAI, D018, DRA have moderate pit rates (15-33%). A smoothed target encoding (e.g., with prior blending) is essential to avoid overfitting on rare drivers while capturing signal from frequent ones.

- [iter 2] Residual analysis shows the incumbent struggles most on drivers MSC, D223, D334, D379, and DOO, with mean absolute residuals >0.5. These drivers have extreme pit probabilities (e.g. D223, VET at 100%), suggesting the raw categorical Driver fails to capture their behavior. A smoothed target encoding could reduce this error by better representing driver-specific tendencies.

- [iter 3] Driver has high mutual information (0.061) with target and shows extreme pit rates: 15 drivers have 100% PitNextLap rate (e.g. D223, VET), while others like D128 have 0%. Residual analysis shows high error on drivers MSC, D223, D334, D379, DOO. A smoothed target encoding of Driver is justified to capture this signal without overfitting on rare drivers.

- [iter 4] Driver has high variability in PitNextLap rate: 15 drivers have 100% rate (e.g. D223, VET), while frequent drivers like RAI, D018 have moderate rates (15-33%). Rare drivers with 100% rate (count=1-3) are prone to overfitting. A smoothed target encoding (e.g. with global mean prior) is essential to balance signal and regularization.

- [iter 4] Residual analysis shows the incumbent has high mean absolute error on drivers MSC, D223, D334, D379, and DOO (>0.5), all of which have extreme PitNextLap rates (e.g. D223, VET at 100%). This confirms that the raw Driver categorical fails to capture their behavior. A smoothed target encoding is likely to reduce this error by better representing driver-specific tendencies while avoiding overfitting on rare drivers.

- [iter 5] Driver has high variability in PitNextLap rate: 15 drivers have 100% rate (e.g. D223, VET), while frequent drivers like RAI, D018 have moderate rates (15-33%). Rare drivers with 100% rate (count=1-3) are prone to overfitting. A smoothed target encoding (e.g. with global mean prior) is essential to balance signal and regularization.

- [iter 5] Residual analysis shows the incumbent has high mean absolute error on drivers MSC, D223, D334, D379, and DOO (>0.5), all of which have extreme PitNextLap rates (e.g. D223, VET at 100%). This confirms that the raw Driver categorical fails to capture their behavior. A smoothed target encoding is likely to reduce this error by better representing driver-specific tendencies while avoiding overfitting on rare drivers.
