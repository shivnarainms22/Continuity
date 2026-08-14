import { useAgentInvestigation } from '../lib/api'
import {
  formatDateRange,
  formatMs,
  formatMultiple,
  formatZScore,
  humanizeBlastRadiusRecord,
  humanizeToolName,
  summarizeToolCall,
} from '../lib/format'
import type { AgentSliceDimension, AgentToolCallFrame, IncidentSummary } from '../types'
import { Sql } from './Sql'

/** The slice a tool call was asked about, as the agent phrased it. `slice_json` arrives
 * either as an object or as a JSON string (the model is free to send either, and the
 * tool layer accepts both), so both are handled rather than one rendering as `[object]`. */
function argumentSlice(args: Record<string, unknown>): Record<string, string> | null {
  const raw = args.slice_json
  if (!raw) return null
  if (typeof raw === 'string') {
    try {
      const parsed = JSON.parse(raw) as Record<string, string>
      return Object.keys(parsed).length ? parsed : null
    } catch {
      return null
    }
  }
  if (typeof raw === 'object') {
    const record = raw as Record<string, string>
    return Object.keys(record).length ? record : null
  }
  return null
}

function Measurement({ call, index }: { call: AgentToolCallFrame; index: number }) {
  const slice = argumentSlice(call.arguments)
  const summary = summarizeToolCall(call.tool, call.result)

  return (
    <li className="relative pl-8">
      <span className="absolute left-0 top-0.5 flex h-5 w-5 items-center justify-center rounded-full border border-hairline bg-surface font-mono text-[10px] text-faint">
        {index + 1}
      </span>
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-sm font-medium text-fg">
          {humanizeToolName(call.tool)}
          <span className="ml-2 font-normal text-muted">
            {slice ? humanizeBlastRadiusRecord(slice) : 'whole population'}
          </span>
        </p>
        <span className="shrink-0 font-mono text-[11px] text-faint">
          +{formatMs(call.elapsed_ms)}
        </span>
      </div>
      {summary && <p className="mt-0.5 text-sm text-accent">{summary}</p>}
      {call.sql && <Sql sql={call.sql} label={`View the query · audit_index ${call.audit_index}`} />}
    </li>
  )
}

function Slice({ dimensions }: { dimensions: AgentSliceDimension[] }) {
  if (!dimensions.length) return <span className="text-muted">the whole population</span>
  return (
    <span className="font-mono text-fg">
      {dimensions.map((d) => `${d.dimension}=${d.value}`).join(' · ')}
    </span>
  )
}

export function AgentInvestigationView({
  incident,
  onBack,
}: {
  incident: IncidentSummary
  onBack: () => void
}) {
  const { detect, stage, toolCalls, done, error, running } = useAgentInvestigation(incident.id)
  const investigation = done?.investigation
  const correlation = done?.correlation
  const quantify = done?.quantify
  const brief = done?.brief

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
              {stage ? stage.label.toLowerCase() : 'starting'}
            </span>
          )}
        </div>
        <p className="text-sm text-muted">
          Gemini investigating · {humanizeBlastRadiusRecord(incident.predicate)} is the planted
          answer, which the agent is not told
        </p>
      </header>

      {detect && (
        <section className="mb-8 rounded border border-hairline bg-surface p-4">
          <div className="mb-1 flex items-baseline justify-between">
            <h2 className="text-xs font-semibold tracking-wide text-faint uppercase">
              Detect · deterministic, no model
            </h2>
            <span className="font-mono text-[11px] text-faint">{formatMs(detect.elapsed_ms)}</span>
          </div>
          <p className="text-sm text-fg">
            {detect.windows_found} anomaly window{detect.windows_found === 1 ? '' : 's'} · peak{' '}
            {formatZScore(detect.peak_z)} from baseline ·{' '}
            {formatDateRange(detect.span.start, detect.span.end)}
          </p>
          <Sql sql={detect.sql} />
        </section>
      )}

      <section className="mb-8">
        <h2 className="mb-4 text-xs font-semibold tracking-wide text-faint uppercase">
          What the agent measured
          {toolCalls.length > 0 && (
            <span className="ml-2 font-mono normal-case text-faint">{toolCalls.length} queries</span>
          )}
        </h2>
        {toolCalls.length === 0 && running && (
          <p className="text-sm text-muted">Forming a hypothesis…</p>
        )}
        <ol className="space-y-5 border-l border-hairline pl-0">
          {toolCalls.map((call, i) => (
            <Measurement key={call.audit_index} call={call} index={i} />
          ))}
        </ol>
      </section>

      {error && (
        <div className="mb-6 rounded border border-danger-dim bg-danger-dim/40 p-3 text-sm text-danger">
          {error}
        </div>
      )}

      {done && !done.detected && (
        <div className="rounded border border-hairline bg-surface p-5 text-sm text-muted">
          <p className="font-medium text-fg">No anomalies detected.</p>
          <p className="mt-1">
            {done.message} Nothing was handed to the model, so this cost no tokens — the correct
            outcome for a healthy window.
          </p>
        </div>
      )}

      {done?.detected && (
        <section className="space-y-6">
          <div className="flex items-baseline justify-between">
            <h2 className="text-xs font-semibold tracking-wide text-faint uppercase">Brief</h2>
            <span className="font-mono text-[11px] text-faint">
              {formatMs(done.total_elapsed_ms)} · {done.tool_calls} tool calls
            </span>
          </div>

          {brief?.unresolved && (
            <div className="rounded border border-danger-dim bg-danger-dim/30 p-4 text-sm text-danger">
              <p className="font-medium">Unresolved — no plausible cause corroborated this.</p>
              <p className="mt-1 text-muted">
                The impact figure below is an unreliable estimate, not a confident finding. Saying
                so is the intended behaviour: a system that reports what it could not explain beats
                one that invents a cause.
              </p>
            </div>
          )}

          <div className="rounded border border-hairline bg-surface p-5">
            <p className="text-sm leading-relaxed text-fg">{brief?.summary}</p>

            <dl className="mt-4 space-y-2 text-sm">
              <div className="flex gap-3">
                <dt className="w-32 shrink-0 text-faint">Blast radius</dt>
                <dd>
                  <Slice dimensions={investigation?.final_slice ?? []} />
                  {investigation?.final_lift != null && (
                    <span className="ml-2 text-muted">
                      lift {formatMultiple(investigation.final_lift)}
                    </span>
                  )}
                </dd>
              </div>
              <div className="flex gap-3">
                <dt className="w-32 shrink-0 text-faint">Probable cause</dt>
                <dd className="text-fg">
                  {correlation?.top_candidate_change_id
                    ? `change #${correlation.top_candidate_change_id}`
                    : 'none corroborated'}
                  <span className="ml-2 text-muted">{correlation?.confidence} confidence</span>
                </dd>
              </div>
              <div className="flex gap-3">
                <dt className="w-32 shrink-0 text-faint">Impact</dt>
                <dd className="text-fg">
                  {quantify?.affected_subscribers.toLocaleString()} subscribers · $
                  {quantify?.arr_at_risk_expected} ARR at risk
                  <span className="ml-2 text-muted">
                    (${quantify?.arr_at_risk_low}–${quantify?.arr_at_risk_high})
                  </span>
                </dd>
              </div>
              <div className="flex gap-3">
                <dt className="w-32 shrink-0 text-faint">Recommended</dt>
                <dd className="text-fg">{brief?.recommended_action}</dd>
              </div>
            </dl>
          </div>

          {correlation?.disconfirming_evidence && (
            <div className="rounded border border-hairline bg-surface p-4">
              <h3 className="mb-1 text-xs font-semibold tracking-wide text-faint uppercase">
                Disconfirming evidence the agent had to engage with
              </h3>
              <p className="text-sm leading-relaxed text-muted">
                {correlation.disconfirming_evidence}
              </p>
            </div>
          )}

          <div className="rounded border border-hairline bg-surface p-4">
            <div className="mb-2 flex items-baseline justify-between">
              <h3 className="text-xs font-semibold tracking-wide text-faint uppercase">
                Every claim, traced
              </h3>
              <span
                className={`font-mono text-[11px] ${done.citations_verified ? 'text-accent' : 'text-danger'}`}
              >
                {done.citations_verified
                  ? 'verified against the audit log'
                  : `citation check FAILED: ${done.citation_error}`}
              </span>
            </div>
            <ul className="space-y-1.5 text-sm">
              {brief?.claims.map((claim, i) => (
                <li key={i} className="flex gap-3">
                  <span className="shrink-0 font-mono text-[11px] text-faint">
                    #{claim.source.audit_index}
                  </span>
                  <span className="text-muted">
                    {claim.text}
                    <span className="ml-2 text-faint">via {claim.source.tool_name}</span>
                  </span>
                </li>
              ))}
            </ul>
          </div>

          {brief?.methodology_notes && (
            <p className="text-[11px] leading-relaxed text-faint">{brief.methodology_notes}</p>
          )}
        </section>
      )}
    </div>
  )
}
