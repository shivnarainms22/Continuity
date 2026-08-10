import { useEffect, useState } from 'react'
import { fetchIncidentSeverity, fetchIncidents } from '../lib/api'
import {
  formatDateRange,
  formatDuration,
  formatUsd,
  humanizeBlastRadiusRecord,
  humanizeKind,
} from '../lib/format'
import type { IncidentSeverity, IncidentSummary } from '../types'

type SeverityState = IncidentSeverity | 'loading' | 'error'

function MoneyAtRisk({ state }: { state: SeverityState | undefined }) {
  if (!state || state === 'loading') {
    return <div className="h-6 w-24 animate-pulse rounded bg-surface-2" />
  }
  if (state === 'error') {
    return <span className="text-sm text-faint">unavailable</span>
  }
  if (state.affected_subscribers === 0) {
    return <span className="text-sm text-faint">no measurable impact</span>
  }
  return (
    <div>
      <div className="font-mono text-lg font-semibold text-fg">{formatUsd(state.arr_at_risk_expected)}</div>
      <div className="text-[11px] text-faint">
        {formatUsd(state.arr_at_risk_low)}–{formatUsd(state.arr_at_risk_high)} ARR at risk
      </div>
    </div>
  )
}

function rank(state: SeverityState | undefined): number {
  if (!state || state === 'loading' || state === 'error') return -1
  return state.arr_at_risk_expected
}

export function IncidentFeed({ onSelect }: { onSelect: (incident: IncidentSummary) => void }) {
  const [incidents, setIncidents] = useState<IncidentSummary[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [severities, setSeverities] = useState<Record<string, SeverityState>>({})

  useEffect(() => {
    fetchIncidents()
      .then(setIncidents)
      .catch((err: unknown) => setLoadError(String(err)))
  }, [])

  useEffect(() => {
    if (!incidents) return
    setSeverities(Object.fromEntries(incidents.map((inc) => [inc.id, 'loading'])))
    for (const incident of incidents) {
      fetchIncidentSeverity(incident.id)
        .then((severity) => setSeverities((prev) => ({ ...prev, [incident.id]: severity })))
        .catch(() => setSeverities((prev) => ({ ...prev, [incident.id]: 'error' })))
    }
  }, [incidents])

  const sorted = incidents
    ? [...incidents].sort((a, b) => rank(severities[b.id]) - rank(severities[a.id]))
    : null

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <header className="mb-10">
        <h1 className="text-lg font-semibold text-fg">Continuity</h1>
        <p className="mt-1 text-sm text-muted">
          Streaming QoE incident investigation. Deterministic detection, no model calls, every number
          traceable to the query that produced it.
        </p>
      </header>

      <h2 className="mb-3 text-xs font-semibold tracking-wide text-faint uppercase">
        Incidents{incidents ? ` (${incidents.length})` : ''}
      </h2>

      {loadError && (
        <div className="rounded border border-danger-dim bg-danger-dim/40 p-3 text-sm text-danger">
          Failed to load incidents: {loadError}
        </div>
      )}

      {!incidents && !loadError && (
        <div className="space-y-2">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-24 animate-pulse rounded border border-hairline bg-surface" />
          ))}
        </div>
      )}

      <ul className="space-y-2">
        {sorted?.map((incident) => (
          <li key={incident.id}>
            <button
              type="button"
              onClick={() => onSelect(incident)}
              className="group flex w-full items-center justify-between gap-6 rounded border border-hairline bg-surface p-4 text-left transition-colors hover:border-hairline-strong hover:bg-surface-2"
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-fg">{humanizeKind(incident.kind)}</span>
                  {incident.is_decoy && (
                    <span className="rounded bg-surface-2 px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-faint uppercase">
                      decoy — no fault
                    </span>
                  )}
                </div>
                <p className="mt-1 truncate text-sm text-muted">
                  {humanizeBlastRadiusRecord(incident.predicate)}
                </p>
                <p className="mt-1 font-mono text-[11px] text-faint">
                  {formatDateRange(incident.window.start, incident.window.end)} ·{' '}
                  {formatDuration(incident.window.start, incident.window.end)} · {incident.id}
                </p>
              </div>
              <div className="shrink-0 text-right">
                <MoneyAtRisk state={severities[incident.id]} />
              </div>
              <span className="shrink-0 text-sm font-medium text-faint transition-colors group-hover:text-accent">
                Investigate →
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
