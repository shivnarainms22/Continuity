import { ApprovalGate } from './ApprovalGate'
import { HeroChart } from './HeroChart'
import { Sql } from './Sql'
import {
  formatDateRange,
  formatMultiple,
  formatPercent,
  formatUsd,
  humanizeBlastRadius,
  humanizeStopReason,
} from '../lib/format'
import type { IncidentReport, PerformanceEntry } from '../types'

function WhatHappened({ report }: { report: IncidentReport }) {
  const h = report.what_happened
  return (
    <section>
      <h3 className="mb-2 text-sm font-semibold text-fg">What happened</h3>
      <p className="text-sm leading-relaxed text-muted">
        Detected at population level between{' '}
        <span className="text-fg">{formatDateRange(h.population_span.start, h.population_span.end)}</span>,
        where the diluted signal breached threshold across {h.population_burst_count} separate burst
        {h.population_burst_count === 1 ? '' : 's'} (peak {h.population_peak_z.toFixed(1)}σ).{' '}
        {h.used_fallback ? (
          <>Re-examined against the isolated blast radius: {h.fallback_reason}.</>
        ) : (
          <>
            Re-examined against the isolated blast radius, the fault actually ran{' '}
            <span className="text-fg">{formatDateRange(h.refined_span!.start, h.refined_span!.end)}</span>{' '}
            (peak {h.refined_peak_z!.toFixed(1)}σ, versus {h.population_peak_z.toFixed(1)}σ at population
            level).
          </>
        )}
      </p>
      <p className="mt-2 text-sm leading-relaxed text-muted">
        Typical degradation across the span was{' '}
        <span className="font-mono font-medium text-fg">{formatMultiple(h.typical_multiple)}</span>{' '}
        baseline (median across every anomalous bucket, not the worst one); the single worst bucket
        reached <span className="font-mono text-fg">{formatMultiple(h.peak_multiple)}</span> baseline.
        Impact below is quantified from the typical figure — a single unlucky bucket must not set the
        churn multiplier for the whole incident.
      </p>
      <Sql sql={h.severity_sql} label="View severity SQL" />
    </section>
  )
}

function WhoAffected({ report }: { report: IncidentReport }) {
  const w = report.who_affected
  return (
    <section>
      <h3 className="mb-2 text-sm font-semibold text-fg">Who was affected</h3>
      <p className="text-sm text-fg">{humanizeBlastRadius(w.predicates)}</p>
      {w.drill_down.length > 0 && (
        <ol className="mt-3 space-y-2">
          {w.drill_down.map((step, i) => (
            <li key={i} className="text-sm text-muted">
              <span className="font-mono text-faint">{i + 1}.</span>{' '}
              <span className="text-fg">
                {step.dimension} = {step.value}
              </span>{' '}
              — {formatPercent(Math.min(step.share_of_deviation, 1))} of the deviation, lift{' '}
              {step.lift.toFixed(1)}x
              <Sql sql={step.sql} label="View drill-down SQL" />
            </li>
          ))}
        </ol>
      )}
      <p className="mt-3 text-xs text-faint">Stopped because {humanizeStopReason(w.stop_reason)}.</p>
    </section>
  )
}

function ProbableCause({ report }: { report: IncidentReport }) {
  const pc = report.probable_cause
  return (
    <section>
      <h3 className="mb-2 text-sm font-semibold text-fg">
        Probable cause <span className="font-normal text-faint">— with disconfirming evidence</span>
      </h3>
      {pc.top ? (
        <div className="text-sm leading-relaxed text-muted">
          <p>
            <span className="font-mono text-fg">[change #{pc.top.change_id}]</span>{' '}
            <span className="text-fg">{pc.top.component}</span>: {pc.top.description}
          </p>
          <p className="mt-1 text-xs text-faint">
            Changed {pc.top.temporal_delta_hours.toFixed(1)}h before the incident's true onset ·
            confidence {pc.top.score.toFixed(2)}
          </p>
          <p className="mt-2 rounded border border-hairline bg-bg p-2.5 text-xs text-muted">
            <span className="font-medium text-fg">Disconfirming evidence checked: </span>
            {pc.top.disconfirming_evidence.note}
          </p>
          <Sql sql={pc.top.sql} label="View correlation SQL" />
          {(pc.others.length > 0 || pc.rejected.length > 0) && (
            <details className="mt-2">
              <summary className="cursor-pointer text-xs font-medium text-faint hover:text-accent">
                {pc.others.length + pc.rejected.length} other change(s) considered
              </summary>
              <ul className="mt-1.5 space-y-1 pl-3 text-xs text-faint">
                {pc.others.map((c) => (
                  <li key={c.change_id}>
                    #{c.change_id} {c.description} — score {c.score.toFixed(2)}
                  </li>
                ))}
                {pc.rejected.map((r) => (
                  <li key={r.change_id}>
                    #{r.change_id} {r.description} — {r.reason}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      ) : (
        <p className="text-sm text-muted">
          No change_log entry correlates with this incident, temporally or dimensionally. No probable
          cause identified from recorded changes.
        </p>
      )}
    </section>
  )
}

function Impact({ report }: { report: IncidentReport }) {
  const im = report.impact
  const m = im.methodology
  return (
    <section>
      <h3 className="mb-2 text-sm font-semibold text-fg">Subscribers affected and ARR at risk</h3>
      <p className="text-sm text-muted">
        <span className="font-mono text-fg">{im.affected_subscribers.toLocaleString()}</span> affected
        subscribers
      </p>
      <p className="mt-1 text-2xl font-semibold text-fg">
        {formatUsd(im.arr_at_risk_low)} <span className="text-faint">–</span> {formatUsd(im.arr_at_risk_high)}
      </p>
      <p className="text-xs text-faint">expected {formatUsd(im.arr_at_risk_expected)}</p>
      <details className="mt-2">
        <summary className="cursor-pointer text-xs font-medium text-faint hover:text-accent">
          Methodology
        </summary>
        <p className="mt-1.5 text-xs leading-relaxed text-muted">
          churn_risk = base_monthly_churn ({m.base_monthly_churn} ±{formatPercent(m.base_churn_variation)}
          , assumption) × tenure_multiplier × severity_multiplier, capped at 1.0. Driven by the typical
          deviation ratio ({m.qoe_delta_ratio.toFixed(2)}), deliberately not the peak ratio (
          {m.peak_deviation_ratio.toFixed(2)}). arr_at_risk sums churn_risk × monthly ARPU × 12 over every
          affected subscriber. {m.notes}
        </p>
      </details>
      <Sql sql={im.sql} label="View impact SQL" />
    </section>
  )
}

export function IncidentBrief({
  report,
  metricLabel,
  performance,
  index,
  total,
}: {
  report: IncidentReport
  metricLabel: string
  performance: PerformanceEntry[]
  index: number
  total: number
}) {
  const blastRadiusText = humanizeBlastRadius(report.who_affected.predicates)

  return (
    <article className="space-y-6 rounded border border-hairline bg-surface p-5">
      {total > 1 && (
        <p className="text-[11px] font-medium tracking-wide text-faint uppercase">
          Incident {index + 1} of {total}
        </p>
      )}

      <WhatHappened report={report} />

      <div className="rounded border border-hairline bg-bg p-3">
        <HeroChart series={report.series} metricLabel={metricLabel} />
      </div>

      <WhoAffected report={report} />
      <ProbableCause report={report} />
      <Impact report={report} />

      <ApprovalGate
        action={report.recommended_action}
        blastRadiusText={blastRadiusText}
        performance={performance}
      />
    </article>
  )
}
