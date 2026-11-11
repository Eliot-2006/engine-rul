# External reference results (not reproduced here)

> **Read this before comparing.** EngineRUL's numbers are **not** ranked against the papers below.
> Published C-MAPSS results differ in target cap, whether test labels are capped, window length,
> normalization, validation protocol, and sometimes in the scoring formula. Our headline metric
> uses **uncapped** official test labels (see `docs/experiments.md`), which is stricter than the
> capped-truth convention used by many papers. Our secondary, capped-truth table in
> `reports/report.md` is the closer (but still not identical) point of comparison.

Per EVAL-41, numbers may only be entered after checking the original paper (table and page). None
of the primary sources below were available to verify during this implementation (they are
behind publisher paywalls), so every numeric cell is **not verified** and intentionally left empty.

| Paper | Model | Subset | Train-target cap | Test labels capped | Window | Normalization | Validation protocol | Score formula | Reported RMSE | Reported score | Citation (table/page) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Heimes (2008), PHM08 | RNN | PHM08 challenge data | piecewise-linear (value not verified) | unstated | not verified | not verified | not verified | PHM08 | not verified | not verified | — |
| Babu, Zhao & Li (2016), DASFAA | CNN | FD001–FD004 | not verified | not verified | not verified | not verified | not verified | not verified | not verified | not verified | — |
| Zheng et al. (2017), IEEE ICPHM | LSTM | FD001–FD004 | not verified | not verified | not verified | not verified | not verified | not verified | not verified | not verified | — |
| Li, Ding & Sun (2018), *Reliab. Eng. Syst. Saf.* | Deep CNN | FD001–FD004 | 125 (commonly cited; not verified) | not verified | 30 for FD001 (commonly cited; not verified) | not verified | not verified | not verified | not verified | not verified | — |

## How to fill a row

1. Obtain the paper itself (not a blog, survey table, or citing paper).
2. Copy the FD001 numbers exactly, with the table number and page.
3. Fill every protocol column; write "unstated" if the paper does not say.
4. Only if every protocol column matches EngineRUL's capped-truth protocol may the README make a
   direct comparison (EVAL-42); otherwise use wording like "under a different protocol".
