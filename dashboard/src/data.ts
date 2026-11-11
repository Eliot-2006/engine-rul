// Typed loaders for the JSON written by `engine-rul export-dashboard`. Every file carries a
// `protocol` tag; a chart asks for the protocol it is allowed to show and gets an error otherwise,
// so official-test and validation-only numbers cannot be mixed by accident.

export type FamilyId = "constant" | "life_prior" | "ridge" | "hgb" | "lstm";
export type Interval = [number, number];
export interface MeanStd { mean: number; std: number }

export interface Provenance {
  experiment: string;
  created_at_utc: string;
  git_commit: string | null;
  git_dirty: boolean | null;
  source_snapshot: string | null;
  official_test_set_used: false;
}

interface Base<P extends string> {
  protocol: P;
  schema_version: number;
  title: string;
  sources: string[];
}

export interface Meta {
  protocol: "meta";
  schema_version: number;
  project: string;
  repository: string;
  families: { id: FamilyId; label: string; kind: "baseline" | "model" }[];
  official_test_evaluated_at: string;
  disclaimer: string;
}

export interface OfficialTest extends Base<"official_test"> {
  dataset: string;
  split: string;
  target: string;
  unit: string;
  n_engines: number;
  n_true_rul_above_cap: number;
  training_target_cap: number;
  ci_method: string;
  evaluated_at: string;
  git_commit: string;
  headline: {
    best_family: FamilyId;
    best_rmse: number;
    baseline_family: FamilyId;
    baseline_rmse: number;
    rmse_reduction_pct: number;
  };
  families: {
    id: FamilyId;
    seeds: number;
    seeds_identical: boolean;
    rmse: MeanStd;
    rmse_ci95: Interval;
    mae: MeanStd;
    mae_ci95: Interval;
    nasa_score: MeanStd;
    bias: MeanStd;
    late_fraction: number;
    rmse_capped_truth: number;
  }[];
  paired: Paired[];
  stages: Record<FamilyId, { bin: string; n: number; rmse: number; bias: number }[]>;
  engines: { unit: number; last_cycle: number; y_true: number; pred: Record<FamilyId, number> }[];
}

export interface Paired { a: FamilyId; b: FamilyId; delta_rmse: number; ci95: Interval; ci_excludes_0: boolean }

export interface Phase1Validation extends Base<"phase1_validation"> {
  split: string;
  target: string;
  selection_rmse: Record<FamilyId, number>;
  selection_note: string;
  trajectories: {
    unit: number;
    failure_cycle: number;
    cycle: number[];
    y_true: number[];
    pred: Record<FamilyId, number[]>;
  }[];
}

export interface CV extends Base<"nested_cv"> {
  split: string;
  target: string;
  endpoints: string;
  primary_strategy: string;
  n_outer_folds: number;
  ci_method: string;
  provenance: Provenance;
  strategies: {
    id: string;
    label: string;
    n_points: number;
    families: {
      id: FamilyId;
      pooled_rmse: number;
      pooled_rmse_ci95: Interval;
      pooled_mae: number;
      fold_rmse_mean: number;
      fold_rmse_std: number;
      mean_rank: number;
      rank1_frac: number;
      bootstrap_rank1_frac: number;
    }[];
    paired: Paired[];
    folds: { family: FamilyId; repeat: number; fold: number; rmse: number; mae: number }[];
  }[];
  rank_agreement: { a: string; b: string; kendall_tau: number }[];
  selection_agreement_with_primary: Record<string, Record<string, number>>;
}

export interface Ablation extends Base<"nested_cv_ablation"> {
  split: string;
  target: string;
  delta: string;
  provenance: Provenance;
  determinism_check_max_abs_diff: Record<string, number>;
  rows: {
    family: FamilyId;
    variant: string;
    group: string;
    description: string;
    pooled_rmse: number;
    delta_rmse: number;
    ci95: Interval;
    ci_excludes_0: boolean;
    repeats: number;
    n_folds: number;
  }[];
}

export interface ConformalRow {
  family: FamilyId;
  strategy: string;
  method: string;
  level: number;
  coverage: number;
  coverage_fold_min?: number;
  coverage_fold_max?: number;
  mean_width: number;
  n_points: number;
  stage?: string;
}

export interface Conformal extends Base<"nested_cv_conformal"> {
  split: string;
  target: string;
  inner_fractions: Record<string, number>;
  calibration_strategy: string;
  levels: number[];
  methods: { id: string; label: string; guarantee: "finite_sample" | "empirical"; note: string }[];
  caveat: string;
  provenance: Provenance;
  summary: ConformalRow[];
  per_stage: ConformalRow[];
}

export interface Benchmarks extends Base<"benchmark"> {
  environment: Record<string, string | number | null>;
  scope_definitions: Record<"model" | "end_to_end", string>;
  method: string;
  memory_note: string;
  process_max_rss_mb: number;
  provenance: Provenance;
  rows: {
    family: FamilyId;
    device: string;
    scope: "model" | "end_to_end";
    unit: string;
    n: number;
    latency_ms_median: number;
    latency_ms_p95: number;
    throughput_per_s: number;
    python_peak_alloc_mb: number;
  }[];
  training: {
    family: FamilyId;
    device: string;
    train_time_s: number;
    epochs: number | null;
    n_params: number | null;
    artifact_bytes: number;
  }[];
}

interface Files {
  "meta.json": Meta;
  "official_test.json": OfficialTest;
  "phase1_validation.json": Phase1Validation;
  "cv.json": CV;
  "ablation.json": Ablation;
  "conformal.json": Conformal;
  "benchmarks.json": Benchmarks;
}

const PROTOCOL: { [K in keyof Files]: Files[K]["protocol"] } = {
  "meta.json": "meta",
  "official_test.json": "official_test",
  "phase1_validation.json": "phase1_validation",
  "cv.json": "nested_cv",
  "ablation.json": "nested_cv_ablation",
  "conformal.json": "nested_cv_conformal",
  "benchmarks.json": "benchmark",
};

export const SCHEMA_VERSION = 1;

export async function load<K extends keyof Files>(file: K): Promise<Files[K]> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/${file}`);
  if (!res.ok) throw new Error(`could not load ${file} (HTTP ${res.status})`);
  const data = (await res.json()) as Files[K];
  if (data.protocol !== PROTOCOL[file]) {
    throw new Error(`${file}: expected protocol "${PROTOCOL[file]}", got "${data.protocol}"`);
  }
  if (data.schema_version !== SCHEMA_VERSION) {
    throw new Error(`${file}: unsupported schema version ${data.schema_version}`);
  }
  return data;
}
