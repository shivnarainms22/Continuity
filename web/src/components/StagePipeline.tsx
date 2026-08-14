import type { StageArrival } from '../lib/api'
import { STAGE_LABELS, formatMs, formatUsd, humanizeBlastRadius, humanizeStopReason } from '../lib/format'
import { STAGE_ORDER } from '../types'

function summarize(stage: StageArrival): string {
  switch (stage.stage) {
    case 'session_startup':
      return 'ClickHouse session ready'
    case 'detect': {
      const p = stage.payload
      return `${p.windows_found} anomaly window(s) across ${p.total_buckets} buckets (${p.anomalous_buckets} anomalous)`
    }
    case 'walk': {
      const p = stage.payload
      return `${p.incidents_after_merge} incident(s) after merging ${p.anomaly_windows} window(s)`
    }
    case 'refine': {
      const p = stage.payload
      return `${p.incidents_refined} incident(s) re-examined against its own blast radius`
    }
    case 'correlate_and_quantify': {
      const p = stage.payload
      return `${p.incidents} incident(s) correlated to a probable cause and quantified`
    }
  }
}

function StageDetail({ stage }: { stage: StageArrival }) {
  switch (stage.stage) {
    case 'session_startup':
      return null
    case 'detect':
      return (
        <ul className="mt-2 space-y-1 font-mono text-[11px] text-muted">
          {stage.payload.windows.map((w, i) => (
            <li key={i}>
              {w.start.slice(11, 16)}–{w.end.slice(11, 16)} · peak z {w.peak_z.toFixed(1)}σ
            </li>
          ))}
        </ul>
      )
    case 'walk':
      return (
        <ul className="mt-2 space-y-1 text-[11px] text-muted">
          {stage.payload.walks.map((w, i) => (
            <li key={i}>
              <span className="font-mono text-faint">
                {w.start.slice(11, 16)}–{w.end.slice(11, 16)}
              </span>{' '}
              → {humanizeBlastRadius(w.blast_radius)}{' '}
              <span className="text-faint">({humanizeStopReason(w.stop_reason)})</span>
            </li>
          ))}
        </ul>
      )
    case 'refine':
      return (
        <ul className="mt-2 space-y-1 text-[11px] text-muted">
          {stage.payload.refined.map((r, i) => (
            <li key={i}>
              <span className="font-mono text-faint">
                {r.span.start.slice(0, 16)}–{r.span.end.slice(11, 16)}
              </span>{' '}
              · {r.typical_multiple.toFixed(1)}x typical baseline
              {r.used_fallback && <span className="text-faint"> (population-level fallback)</span>}
            </li>
          ))}
        </ul>
      )
    case 'correlate_and_quantify':
      return (
        <ul className="mt-2 space-y-1 text-[11px] text-muted">
          {stage.payload.detail.map((d, i) => (
            <li key={i}>
              {humanizeBlastRadius(d.blast_radius)} ·{' '}
              <span className="font-mono text-fg">{formatUsd(d.arr_at_risk_expected)}</span> ARR at risk
              {d.top_cause && <span className="text-faint"> — {d.top_cause}</span>}
            </li>
          ))}
        </ul>
      )
  }
}

function Dot({ state }: { state: 'done' | 'active' | 'pending' }) {
  if (state === 'done') {
    return <span className="h-2 w-2 shrink-0 rounded-full bg-accent" />
  }
  if (state === 'active') {
    return (
      <span className="relative flex h-2 w-2 shrink-0">
        <span className="absolute h-2 w-2 animate-ping rounded-full bg-accent opacity-60" />
        <span className="h-2 w-2 rounded-full bg-accent" />
      </span>
    )
  }
  return <span className="h-2 w-2 shrink-0 rounded-full border border-hairline-strong" />
}

/** The "watch it think" pipeline: one card per stage, filling in with real timings as
 * SSE frames arrive, each expandable to the evidence it actually produced. */
export function StagePipeline({ stages, running }: { stages: StageArrival[]; running: boolean }) {
  const byName = new Map(stages.map((s) => [s.stage, s]))

  return (
    <ol className="space-y-2">
      {STAGE_ORDER.map((name, i) => {
        const stage = byName.get(name)
        const isNext = !stage && running && i === stages.length
        const state: 'done' | 'active' | 'pending' = stage ? 'done' : isNext ? 'active' : 'pending'

        return (
          <li
            key={name}
            className={`rounded border border-hairline bg-surface px-3 py-2.5 transition-opacity duration-300 ${
              state === 'pending' ? 'opacity-40' : 'opacity-100'
            }`}
          >
            <div className="flex items-center gap-3">
              <Dot state={state} />
              <span className="w-40 shrink-0 text-sm font-medium text-fg">{STAGE_LABELS[name]}</span>
              <span className="min-w-0 flex-1 truncate text-xs text-muted">
                {stage ? summarize(stage) : state === 'active' ? 'running…' : ''}
              </span>
              {stage && (
                <span className="shrink-0 font-mono text-xs text-faint">{formatMs(stage.elapsed_ms)}</span>
              )}
            </div>
            {stage && stage.stage !== 'session_startup' && (
              <details>
                <summary className="mt-1.5 cursor-pointer pl-5 text-[11px] font-medium text-faint hover:text-accent">
                  evidence
                </summary>
                <div className="pl-5">
                  <StageDetail stage={stage} />
                </div>
              </details>
            )}
          </li>
        )
      })}
    </ol>
  )
}
