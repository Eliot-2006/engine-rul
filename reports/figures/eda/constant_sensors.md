# Dropped columns (EDA-07)

Measured on all 100 FD001 training engines. Re-derived on train-split rows only at fit time (PREP-01).

| Column | Distinct values | Std | Reason |
|---|---|---|---|
| `op3` (operational setting) | 1 | 0.00e+00 | constant |
| `s1` (T2 – total temperature at fan inlet (°R)) | 1 | 0.00e+00 | constant |
| `s5` (P2 – pressure at fan inlet (psia)) | 1 | 5.33e-15 | constant |
| `s10` (epr – engine pressure ratio (P50/P2)) | 1 | 0.00e+00 | constant |
| `s16` (farB – burner fuel-air ratio) | 1 | 3.47e-18 | constant |
| `s18` (Nf_dmd – demanded fan speed (rpm)) | 1 | 0.00e+00 | constant |
| `s19` (PCNfR_dmd – demanded corrected fan speed (rpm)) | 1 | 0.00e+00 | constant |
| `s6` (P15 – total pressure in bypass duct (psia)) | 2 | 1.39e-03 | only two distinct values (quantization noise) |
| `op1` (operational setting) | 158 | 2.19e-03 | noise around FD001's single operating condition; dropped by config (ADR-02) |
| `op2` (operational setting) | 13 | 2.93e-04 | noise around FD001's single operating condition; dropped by config (ADR-02) |

Kept: 14 sensors: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21.
