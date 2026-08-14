import { useInvestigation } from '../lib/api'
import { formatDateRange, formatMs, humanizeBlastRadiusRecord } from '../lib/format'
import type { IncidentSummary } from '../types'
import { IncidentBrief } from './IncidentBrief'
import { StagePipeline } from './StagePipeline'

export function InvestigationView({
  incident,
  onBack,
}: {
  incident: IncidentSummary
  onBack: () => void
}) {
  const { stages, done, error, running } = useInvestigation(incident.id)

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <button
        type="button"
        onClick={onBack}
        className="mb-6 text-xs font-medium text-faint transition-colors hover:text-fg"
      >
        ← Back to feed
      </button>

      <header className="mb-8">
        <div className="mb-1 flex items-center gap-2">
          <h1 className="font-mono text-lg font-semibold text-fg">{incident.id}</h1>
          {running && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-accent-dim px-2 py-0.5 text-[11px] font-medium text-accent">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute h-1.5 w-1.5 animate-ping rounded-full bg-accent opacity-70" />
                <span className="h-1.5 w-1.5 rounded-full bg-accent" />
              </span>
              investigating
            </span>
          )}
        </div>
        <p className="text-sm text-muted">
          {humanizeBlastRadiusRecord(incident.predicate)} · {formatDateRange(incident.window.start, incident.window.end)}
        </p>
      </header>

      <section className="mb-8">
        <h2 className="mb-3 text-xs font-semibold tracking-wide text-faint uppercase">
          Investigation pipeline
        </h2>
        <StagePipeline stages={stages} running={running} />
      </section>

      {error && (
        <div className="mb-6 rounded border border-danger-dim bg-danger-dim/40 p-3 text-sm text-danger">
          {error}
        </div>
      )}

      {done && (
        <section>
          <div className="mb-4 flex items-baseline justify-between">
            <h2 className="text-xs font-semibold tracking-wide text-faint uppercase">Incident brief</h2>
            <span className="font-mono text-[11px] text-faint">
              completed in {formatMs(done.total_elapsed_ms)}
            </span>
          </div>

          {done.report.incidents.length === 0 ? (
            <div className="rounded border border-hairline bg-surface p-5 text-sm text-muted">
              <p className="font-medium text-fg">No anomalies detected.</p>
              <p className="mt-1">
                This is a normal, healthy outcome, not a failure: {done.report.metric_label.toLowerCase()}{' '}
                stayed within its seasonality-aware expected range for the whole window checked.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {done.report.incidents.map((ir, i) => (
                <IncidentBrief
                  key={i}
                  report={ir}
                  metricLabel={done.report.metric_label}
                  performance={done.report.performance}
                  index={i}
                  total={done.report.incidents.length}
                />
              ))}
            </div>
          )}

          <details className="mt-6">
            <summary className="cursor-pointer text-xs font-medium text-faint hover:text-accent">
              View plain-text brief
            </summary>
            <pre className="mt-2 max-h-96 overflow-auto rounded border border-hairline bg-surface p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-muted">
              {done.brief}
            </pre>
          </details>
        </section>
      )}
    </div>
  )
}
