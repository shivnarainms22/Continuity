// Types mirror continuity/api's JSON shapes exactly -- see
// continuity/api/ground_truth.py, continuity/api/incidents_severity.py and
// continuity/api/report_schema.py for the Python side of every field below.

export interface IncidentSummary {
  id: string
  window: { start: string; end: string }
  predicate: Record<string, string>
  kind: string | null
  is_decoy: boolean
}

export interface IncidentSeverity {
  id: string
  affected_subscribers: number
  arr_at_risk_low: number
  arr_at_risk_expected: number
  arr_at_risk_high: number
  sql: string | null
}

export interface Predicate {
  dimension: string
  value: string
}

// --- SSE stage frames --------------------------------------------------------

export interface AnomalyWindowRef {
  start: string
  end: string
  peak_z: number
}

export interface WalkRef {
  start: string
  end: string
  blast_radius: Predicate[]
  stop_reason: string
}

export interface RefinedRef {
  used_fallback: boolean
  span: { start: string; end: string }
  typical_multiple: number
}

export interface CorrelateQuantifyDetail {
  blast_radius: Predicate[]
  top_cause: string | null
  affected_subscribers: number
  arr_at_risk_expected: number
}

export interface StagePayloads {
  session_startup: Record<string, never>
  detect: {
    total_buckets: number
    anomalous_buckets: number
    windows_found: number
    windows: AnomalyWindowRef[]
  }
  walk: {
    anomaly_windows: number
    incidents_after_merge: number
    walks: WalkRef[]
  }
  refine: {
    incidents_refined: number
    refined: RefinedRef[]
  }
  correlate_and_quantify: {
    incidents: number
    detail: CorrelateQuantifyDetail[]
  }
}

export type StageName = keyof StagePayloads

export const STAGE_ORDER: StageName[] = [
  'session_startup',
  'detect',
  'walk',
  'refine',
  'correlate_and_quantify',
]

// A genuine discriminated union (one concrete object type per stage name, joined with
// `|`), not a generic interface -- only this form lets TypeScript narrow `.payload`'s
// type from a `switch` on `.stage` elsewhere in the app.
export type StageFrame = {
  [K in StageName]: { stage: K; elapsed_ms: number; payload: StagePayloads[K] }
}[StageName]

// --- the final, structured report --------------------------------------------

export interface DetectionSummary {
  total_buckets: number
  anomalous_buckets: number
  unknown_buckets: number
  unknown_fraction: number
  windows_found: number
  sql: string
}

export interface WhatHappened {
  metric_label: string
  population_span: { start: string; end: string }
  population_anomalous_buckets: number
  population_span_buckets: number
  population_burst_count: number
  population_peak_z: number
  used_fallback: boolean
  fallback_reason: string | null
  refined_span: { start: string; end: string } | null
  refined_peak_z: number | null
  refined_anomalous_buckets: number | null
  refined_span_buckets: number | null
  peak_value: number
  expected_at_peak: number
  typical_multiple: number
  peak_multiple: number
  severity_sql: string
}

export interface DrillDownStep {
  dimension: string
  value: string
  share_of_deviation: number
  lift: number
  weight: number
  sql: string
  baseline_sql: string
}

export interface WhoAffected {
  predicates: Predicate[]
  drill_down: DrillDownStep[]
  stop_reason: string
  stop_detail: string
  peak_window: { start: string; end: string }
}

export interface DisconfirmingEvidence {
  sibling_dimension: string | null
  note: string
  siblings_checked: number
  siblings_degraded: number
  siblings_not_degraded: number
}

export interface Candidate {
  change_id: number
  changed_at: string
  change_type: string
  component: string
  description: string
  dimension_key: string
  dimension_value: string
  score: number
  temporal_delta_hours: number
  dimensional_overlap: boolean
  disconfirming_evidence: DisconfirmingEvidence
  sql: string
}

export interface RejectedCandidate {
  change_id: number
  changed_at: string
  description: string
  reason: string
}

export interface ProbableCause {
  top: Candidate | null
  others: Candidate[]
  rejected: RejectedCandidate[]
  sql: string
}

export interface ImpactMethodology {
  base_monthly_churn: number
  base_churn_variation: number
  qoe_delta_ratio: number
  peak_deviation_ratio: number
  notes: string
}

export interface Impact {
  affected_subscribers: number
  arr_at_risk_low: number
  arr_at_risk_expected: number
  arr_at_risk_high: number
  methodology: ImpactMethodology
  sql: string
}

export interface RecommendedAction {
  has_candidate: boolean
  change_id: number | null
  component: string | null
  description: string | null
}

export type BucketStatus = 'anomalous' | 'normal' | 'unknown'

export interface SeriesPoint {
  bucket: string
  value: number | null
  expected: number | null
  lower: number | null
  upper: number | null
  status: BucketStatus
}

export interface IncidentSeries {
  points: SeriesPoint[]
  sql: string
  metric: string
  anomaly_windows: { start: string; end: string }[]
}

export interface IncidentReport {
  what_happened: WhatHappened
  who_affected: WhoAffected
  probable_cause: ProbableCause
  impact: Impact
  recommended_action: RecommendedAction
  series: IncidentSeries
}

export interface PerformanceEntry {
  stage: string
  elapsed_ms: number
  queries: number
}

export interface Report {
  metric: string
  metric_label: string
  description: string
  window: { start: string; end: string }
  detection: DetectionSummary
  incidents: IncidentReport[]
  performance: PerformanceEntry[]
  total_elapsed_ms: number
}

export interface DoneFrame {
  total_elapsed_ms: number
  brief: string
  report: Report
}

// --- Agent stream (GET /api/investigate/{id}/agent-stream) --------------------
// The Gemini arm. Where the walker reports one frame per pipeline STAGE, the agent
// reports one per MEASUREMENT, so the view can show the work as it happens rather
// than a spinner over a ~45s investigation.

export interface AgentDetectFrame {
  description: string
  metric: string
  windows_found: number
  span: { start: string; end: string }
  peak_z: number
  sql: string
  elapsed_ms: number
}

export interface AgentStageFrame {
  stage: 'investigate' | 'correlate' | 'quantify' | 'brief'
  label: string
}

/** One tool call the model chose to make. `audit_index` is the same value a brief
 * claim cites, which is what lets a figure in the brief link back to the measurement
 * that produced it. `sql` is the full query text from the audit log. */
export interface AgentToolCallFrame {
  audit_index: number
  tool: string
  arguments: Record<string, unknown>
  sql: string | null
  result: Record<string, unknown>
  elapsed_ms: number
}

export interface AgentSliceDimension {
  dimension: string
  value: string
}

export interface AgentDoneFrame {
  detected: boolean
  message?: string
  total_elapsed_ms: number
  tool_calls?: number
  investigation?: {
    hypothesis: string
    final_slice: AgentSliceDimension[]
    final_lift: number | null
    stop_reason: string
    reasoning: string
  }
  correlation?: {
    confidence: string
    corroborated: boolean
    top_candidate_change_id: string | null
    disconfirming_evidence: string
    reasoning: string
  }
  quantify?: {
    affected_subscribers: number
    arr_at_risk_low: string
    arr_at_risk_expected: string
    arr_at_risk_high: string
    methodology_caveat: string
  }
  brief?: {
    summary: string
    claims: { text: string; source: { tool_name: string; audit_index: number } }[]
    recommended_action: string
    methodology_notes: string
    unresolved: boolean
  }
  citations_verified?: boolean
  citation_error?: string | null
}
