// Presentation-only helpers: turning structured facts the API sends (dimension/value
// pairs, enums, numbers) into prose. Deliberately kept out of continuity/api -- see
// continuity/api/report_schema.py's own module docstring for why that split was made.

import type { Predicate, StageName } from '../types'

export const STAGE_LABELS: Record<StageName, string> = {
  session_startup: 'Connect',
  detect: 'Detect',
  walk: 'Localize',
  refine: 'Refine',
  correlate_and_quantify: 'Correlate & quantify',
}

// Mirrors continuity/analysis/cli.py's _DIMENSION_PHRASES exactly, so a blast radius
// reads the same way here as it does in the plain-text brief.
const DIMENSION_PHRASES: Record<string, (value: string) => string> = {
  device_type: (v) => `${v} devices`,
  app_version: (v) => `app ${v}`,
  os_version: (v) => `OS ${v}`,
  cdn: (v) => `CDN ${v}`,
  pop: (v) => `PoP ${v}`,
  isp: (v) => `ISP ${v}`,
  country: (v) => `country ${v}`,
  region: (v) => `region ${v}`,
  title_id: (v) => `title ${v}`,
}

function phraseFor({ dimension, value }: Predicate): string {
  const fn = DIMENSION_PHRASES[dimension]
  return fn ? fn(value) : `${dimension}=${value}`
}

function joinPhrases(phrases: string[]): string {
  if (phrases.length === 0) return ''
  if (phrases.length === 1) return phrases[0]
  if (phrases.length === 2) return `${phrases[0]} and ${phrases[1]}`
  return `${phrases.slice(0, -1).join(', ')}, and ${phrases[phrases.length - 1]}`
}

/** "roku devices and app 8.2.0", or `wholePopulationText` when there is no predicate. */
export function humanizeBlastRadius(
  predicates: Predicate[],
  wholePopulationText = 'the whole population',
): string {
  if (predicates.length === 0) return wholePopulationText
  return joinPhrases(predicates.map(phraseFor))
}

/** Same as `humanizeBlastRadius`, from a plain `{dimension: value}` record (the shape
 * `/api/incidents` returns) -- relies on the record's own key order, which
 * data/ground_truth.json already writes coarse-to-fine. */
export function humanizeBlastRadiusRecord(
  predicate: Record<string, string>,
  wholePopulationText = 'All users',
): string {
  const entries = Object.entries(predicate).map(([dimension, value]) => ({ dimension, value }))
  return humanizeBlastRadius(entries, wholePopulationText)
}

// Mirrors continuity/analysis/cli.py's _STOP_REASON_TEXT.
const STOP_REASON_TEXT: Record<string, string> = {
  low_lift:
    'the best-explaining value only explained its own share of the population -- big, not causal',
  low_share: 'no remaining dimension explained enough of the deviation to justify descending further',
  single_value: 'every remaining dimension had only one usable value -- nothing left to compare',
  max_depth: 'reached the maximum drill-down depth',
  too_small: 'the next candidate slice was too small a share of the population to trust its ratio',
  dimensions_exhausted: 'every candidate dimension is already fixed in this slice',
}

export function humanizeStopReason(reason: string): string {
  return STOP_REASON_TEXT[reason] ?? reason
}

// Matches the `kind` values continuity/data/generator.py plants in ground_truth.json.
const KIND_TITLES: Record<string, string> = {
  device_app_fault: 'Device / app fault',
  pop_fault: 'CDN PoP fault',
  encode_fault: 'Encode pipeline fault',
  decoy_premiere: 'Content premiere (no fault)',
}

export function humanizeKind(kind: string | null): string {
  if (kind === null) return 'Unclassified incident'
  return KIND_TITLES[kind] ?? kind
}

const usdFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
})

export function formatUsd(value: number): string {
  return usdFormatter.format(value)
}

export function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

export function formatPercent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`
}

export function formatMultiple(value: number): string {
  return `${value.toFixed(1)}x`
}

/** A robust z-score: a COUNT of deviations from baseline, not a multiple of anything.
 * Rendering it through `formatMultiple` produced "90.5x robust deviations", which reads
 * as a ratio and is simply the wrong unit -- the sort of error a careful reader spots
 * immediately and rightly distrusts the rest of the page for. */
export function formatZScore(value: number): string {
  return `${value.toFixed(1)}σ`
}

// Every naive ISO-8601 timestamp in this dataset is UTC by this project's own
// convention (see continuity/config.py) -- the backend never appends an offset.
// `new Date("2026-02-12T18:10:00")` parses a naive string as the *browser's local
// time*, not UTC, so every downstream render silently shifts by the host's UTC
// offset (this bit the data loader once already; see CLAUDE.md). `parseUtc` is the
// ONE place a naive-UTC string becomes a `Date` -- every component and every helper
// below routes through it instead of calling `new Date(...)` directly.
const HAS_TZ_OFFSET = /(Z|[+-]\d{2}:?\d{2})$/

export function parseUtc(iso: string): Date {
  return new Date(HAS_TZ_OFFSET.test(iso) ? iso : `${iso}Z`)
}

const dateTimeFormatter = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  day: 'numeric',
  year: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
  timeZone: 'UTC',
})

/** Naive ISO timestamps throughout this dataset are UTC (see continuity/config.py) --
 * pinned to the UTC timeZone above so this never silently shifts by the browser's own
 * local offset. */
export function formatDateTime(iso: string): string {
  return `${dateTimeFormatter.format(parseUtc(iso))} UTC`
}

const timeOnlyFormatter = new Intl.DateTimeFormat('en-US', {
  hour: 'numeric',
  minute: '2-digit',
  timeZone: 'UTC',
})

/** Full "Feb 13, 2026, 3:40 AM – 9:05 AM UTC" when both ends fall on the same UTC
 * calendar day; the date is repeated only when the range actually spans days. */
export function formatDateRange(startIso: string, endIso: string): string {
  const start = parseUtc(startIso)
  const end = parseUtc(endIso)
  const sameDay =
    start.getUTCFullYear() === end.getUTCFullYear() &&
    start.getUTCMonth() === end.getUTCMonth() &&
    start.getUTCDate() === end.getUTCDate()
  if (sameDay) {
    return `${dateTimeFormatter.format(start)} – ${timeOnlyFormatter.format(end)} UTC`
  }
  return `${formatDateTime(startIso)} – ${formatDateTime(endIso)}`
}

export function formatDuration(startIso: string, endIso: string): string {
  const ms = parseUtc(endIso).getTime() - parseUtc(startIso).getTime()
  const hours = ms / (1000 * 60 * 60)
  if (hours < 1) return `${Math.round(ms / (1000 * 60))}m`
  if (hours < 48) return `${hours % 1 === 0 ? hours : hours.toFixed(1)}h`
  return `${(hours / 24).toFixed(1)}d`
}

export function formatMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

/** Value formatting per continuity/analysis/metrics.py's own unit for each metric --
 * rebuffer/errors are ratios best read as a percentage, startup is milliseconds,
 * bitrate is kbps. */
export function formatMetricValue(metric: string, value: number): string {
  if (metric === 'rebuffer' || metric === 'errors') return formatPercent(value, 2)
  if (metric === 'startup') return `${value.toFixed(0)}ms`
  if (metric === 'bitrate') return `${value.toFixed(0)}kbps`
  return value.toFixed(3)
}

export function formatTime(iso: string): string {
  return new Intl.DateTimeFormat('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    timeZone: 'UTC',
  }).format(parseUtc(iso))
}

/** The human label for one of the agent's tools, for the live measurement trail. */
export function humanizeToolName(tool: string): string {
  const labels: Record<string, string> = {
    detect_anomalies: 'Scanned for anomalies',
    measure_slice: 'Measured slice',
    split_on_dimension: 'Split on dimension',
    split_all_dimensions: 'Split across dimensions',
    refine_incident_span: 'Refined incident span',
    find_changes: 'Searched the change log',
    quantify_impact: 'Quantified impact',
  }
  return labels[tool] ?? tool.replace(/_/g, ' ')
}

/** A one-line finding for a tool call, read from fields the tool actually returned.
 *
 * Never computes or infers a figure -- it only selects and formats what is already in
 * `result`, which is the same rule the agent itself works under. An unrecognised tool
 * (or a result shape that has moved) yields `null` and the row renders without a
 * headline rather than with a confident wrong one. */
export function summarizeToolCall(
  tool: string,
  result: Record<string, unknown>,
): string | null {
  if (result.error) return String(result.error)

  if (tool === 'split_all_dimensions' || tool === 'split_on_dimension') {
    const dims = (result.dimensions ?? result.values) as
      | { dimension?: string; top_value?: string; lift?: number; meets_lift_gate?: boolean }[]
      | undefined
    const best = dims?.find((d) => d.meets_lift_gate) ?? dims?.[0]
    if (!best?.dimension) return null
    const lift = typeof best.lift === 'number' ? `${formatMultiple(best.lift)} its size` : null
    const where = best.top_value ? `${best.dimension}=${best.top_value}` : best.dimension
    return lift ? `${where} explains ${lift}` : where
  }

  if (tool === 'measure_slice') {
    const z = result.z
    const value = result.value
    if (typeof z !== 'number' || typeof value !== 'number') return null
    return `${value.toPrecision(3)} · ${formatZScore(Math.abs(z))} from its own baseline`
  }

  if (tool === 'find_changes') {
    const candidates = result.candidates as { change_id?: string | number }[] | undefined
    const rejected = result.rejected as unknown[] | undefined
    if (!candidates?.length) return `no plausible change found · ${rejected?.length ?? 0} rejected`
    return `${candidates.length} candidate${candidates.length === 1 ? '' : 's'}, ${rejected?.length ?? 0} rejected`
  }

  if (tool === 'quantify_impact') {
    const subs = result.affected_subscribers
    const arr = result.arr_at_risk_expected
    if (typeof subs !== 'number') return null
    return arr ? `${formatCompactNumber(subs)} subscribers · $${arr} ARR at risk` : `${formatCompactNumber(subs)} subscribers`
  }

  if (tool === 'refine_incident_span') {
    const start = result.window_start
    const end = result.window_end
    if (typeof start !== 'string' || typeof end !== 'string') return null
    return `true span ${formatDateRange(start, end)}`
  }

  if (tool === 'detect_anomalies') {
    const windows = result.windows as unknown[] | undefined
    if (!windows) return null
    return `${windows.length} anomaly window${windows.length === 1 ? '' : 's'}`
  }

  return null
}
