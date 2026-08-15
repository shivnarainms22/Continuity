import { useState } from 'react'
import { formatMs, humanizeToolName } from '../lib/format'
import type { ToolCallArrival } from '../lib/api'

type Decision = 'pending' | 'approved' | 'rejected'

/** The ACT stage's gate on the agent path: the remediation the model wrote is a
 * PROPOSAL and nothing applies it.
 *
 * Deliberately separate from `ApprovalGate` rather than a generalisation of it. That one
 * is driven by the walker's typed `RecommendedAction` and per-stage timings; this one by
 * the agent's prose recommendation and its tool-call audit trail. Merging them would mean
 * a union prop and two code paths inside one component, which is more tangle than the
 * shared markup is worth. The guarantee they both make is identical and stated the same
 * way, because that is the part that must not drift.
 *
 * Approve/Reject is local UI state. Nothing here writes to ClickHouse or anywhere else,
 * which is the point: `propose_action` in continuity/agent/agents.py raises
 * ApprovalRequiredError rather than acting, and this is that constraint made visible. */
export function AgentApprovalGate({
  recommendedAction,
  unresolved,
  toolCalls,
}: {
  recommendedAction: string
  unresolved: boolean
  toolCalls: ToolCallArrival[]
}) {
  const [decision, setDecision] = useState<Decision>('pending')

  return (
    <div className="rounded border border-hairline-strong bg-surface p-4">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-[11px] font-semibold tracking-wide text-accent uppercase">
          Proposal — requires human approval
        </span>
        {decision === 'pending' && (
          <span className="text-[11px] text-faint">nothing has been written anywhere</span>
        )}
      </div>

      <p className="text-sm leading-relaxed text-fg">{recommendedAction}</p>

      {unresolved && (
        <p className="mt-2 text-[11px] leading-relaxed text-danger">
          The agent could not corroborate a cause for this incident. Treat the proposal as a
          starting point for a human investigation, not as a remediation to apply.
        </p>
      )}

      <div className="mt-4 flex items-center gap-2">
        {decision === 'pending' ? (
          <>
            <button
              type="button"
              onClick={() => setDecision('approved')}
              className="rounded bg-accent px-3 py-1.5 text-xs font-medium text-accent-fg transition-opacity hover:opacity-90"
            >
              Approve
            </button>
            <button
              type="button"
              onClick={() => setDecision('rejected')}
              className="rounded border border-hairline-strong px-3 py-1.5 text-xs font-medium text-muted transition-colors hover:border-danger hover:text-danger"
            >
              Reject
            </button>
          </>
        ) : decision === 'approved' ? (
          <span className="inline-flex items-center gap-1.5 rounded bg-success-dim px-2.5 py-1 text-xs font-medium text-success">
            Approved — no action was actually taken (demo)
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 rounded bg-danger-dim px-2.5 py-1 text-xs font-medium text-danger">
            Rejected
          </span>
        )}
      </div>

      <details className="mt-3">
        <summary className="cursor-pointer text-[11px] font-medium text-faint hover:text-accent">
          Audit trail — every measurement behind this proposal
        </summary>
        <ul className="mt-2 space-y-1 border-l border-hairline pl-3">
          {toolCalls.map((call) => (
            <li
              key={call.audit_index}
              className="flex items-center justify-between gap-3 text-[11px] text-muted"
            >
              <span>
                <span className="font-mono text-faint">#{call.audit_index}</span>{' '}
                {humanizeToolName(call.tool)}
              </span>
              <span className="font-mono text-faint">{formatMs(call.elapsed_ms)}</span>
            </li>
          ))}
        </ul>
      </details>
    </div>
  )
}
