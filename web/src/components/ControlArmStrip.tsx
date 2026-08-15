import { useInvestigation } from '../lib/api'
import { formatMs, formatUsd } from '../lib/format'
import type { AgentSliceDimension, Predicate } from '../types'

/** The deterministic walker, run alongside the agent as the control it is scored
 * against (results/comparison.json).
 *
 * On screen rather than in a JSON file on purpose: a measured baseline is the thing
 * that turns "the agent is good" from an assertion into a number, and it is only
 * convincing if the audience can watch both arms answer the same question at the same
 * time. It also keeps the honest half of the story visible -- the walker is free,
 * instant and deterministic, and on two of three planted incidents it is simply right.
 *
 * Both arms open their own SSE stream. They contend only for the ClickHouse gateway,
 * and the agent spends 45s of its ~45s waiting on the model (profiled: 0.2s of tool
 * time), so the walker's queries slot into gaps the agent is not using. */
export function ControlArmStrip({
  incidentId,
  agentSlice,
  agentChangeId,
  agentElapsedMs,
}: {
  incidentId: string
  agentSlice: AgentSliceDimension[] | null
  agentChangeId: string | null
  agentElapsedMs: number | null
}) {
  const { done, error, running } = useInvestigation(incidentId)
  const incident = done?.report.incidents[0] ?? null

  const walkerPredicates: Predicate[] = incident?.who_affected.predicates ?? []
  const walkerSlice = walkerPredicates.map((p) => `${p.dimension}=${p.value}`).join(' · ')
  // The walker types change_id as a number and the agent's schema as a string. Comparing
  // them directly is always unequal, which would report "causes differ" on every incident
  // including the ones where both arms agree -- exactly the wrong claim to get wrong on a
  // panel whose whole job is to compare the two honestly.
  const walkerChange =
    incident?.probable_cause.top?.change_id != null
      ? String(incident.probable_cause.top.change_id)
      : null

  const agentSliceText = agentSlice?.length
    ? agentSlice.map((d) => `${d.dimension}=${d.value}`).join(' · ')
    : null

  // Only claim a divergence once BOTH arms have actually answered.
  const bothAnswered = Boolean(done && agentSliceText !== null)
  const sameSlice = bothAnswered && walkerSlice === agentSliceText

  return (
    <section className="mb-8 rounded border border-hairline bg-surface/60 p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-xs font-semibold tracking-wide text-faint uppercase">
          Deterministic control · no model, $0
        </h2>
        <span className="font-mono text-[11px] text-faint">
          {running && 'running…'}
          {done && `answered in ${formatMs(done.total_elapsed_ms)}`}
          {error && <span className="text-danger">control failed</span>}
        </span>
      </div>

      {done && !incident && (
        <p className="text-sm text-muted">No incident found — the correct answer for a decoy.</p>
      )}

      {incident && (
        <div className="grid gap-x-6 gap-y-1 text-sm sm:grid-cols-[8rem_1fr]">
          <span className="text-faint">Blast radius</span>
          <span className="font-mono text-fg">{walkerSlice || 'whole population'}</span>
          <span className="text-faint">Probable cause</span>
          <span className="text-fg">
            {walkerChange ? `change #${walkerChange}` : 'none corroborated'}
          </span>
          <span className="text-faint">Impact</span>
          <span className="text-fg">
            {incident.impact.affected_subscribers.toLocaleString()} subscribers ·{' '}
            {formatUsd(incident.impact.arr_at_risk_expected)} ARR at risk
          </span>
        </div>
      )}

      {bothAnswered && (
        <p
          className={`mt-3 border-t border-hairline pt-3 text-sm ${
            sameSlice ? 'text-muted' : 'text-accent'
          }`}
        >
          {sameSlice ? (
            <>
              Both arms reached the same blast radius. The walker got there in{' '}
              {formatMs(done!.total_elapsed_ms)} at zero cost and no model calls
              {agentElapsedMs ? `, the agent in ${formatMs(agentElapsedMs)}` : ''} — on this
              incident the model earned no localisation advantage, and saying so is the point of
              running it.
            </>
          ) : (
            <>
              The two arms disagree. Walker: <span className="font-mono">{walkerSlice}</span>. Agent:{' '}
              <span className="font-mono">{agentSliceText}</span>
              {agentChangeId && walkerChange !== agentChangeId && (
                <> · causes differ too (#{walkerChange ?? 'none'} vs #{agentChangeId})</>
              )}
              .
            </>
          )}
        </p>
      )}
    </section>
  )
}
