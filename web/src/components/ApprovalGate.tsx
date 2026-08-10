import { useState } from 'react'
import { STAGE_LABELS, formatMs } from '../lib/format'
import type { PerformanceEntry, RecommendedAction } from '../types'

type Decision = 'pending' | 'approved' | 'rejected'

/** The governance gate: the recommended action is a PROPOSAL, never auto-applied.
 * Approve/Reject is local UI state only -- nothing here writes to ClickHouse or
 * anywhere else, by design. The audit trail lists exactly what was checked (and how
 * long each check took) before this recommendation was made. */
export function ApprovalGate({
  action,
  blastRadiusText,
  performance,
}: {
  action: RecommendedAction
  blastRadiusText: string
  performance: PerformanceEntry[]
}) {
  const [decision, setDecision] = useState<Decision>('pending')

  const actionText = action.has_candidate
    ? `Roll back or hotfix change #${action.change_id} (${action.component}): ${action.description}. It is the top-ranked probable cause for the impact on ${blastRadiusText}.`
    : `No change_log entry explains this incident. Escalate to the on-call team for manual investigation of ${blastRadiusText}.`

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

      <p className="text-sm leading-relaxed text-fg">{actionText}</p>

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
          Audit trail — what was checked before this recommendation
        </summary>
        <ul className="mt-2 space-y-1 border-l border-hairline pl-3">
          {performance
            .filter((p) => p.stage !== 'session_startup')
            .map((p) => (
              <li key={p.stage} className="flex items-center justify-between text-[11px] text-muted">
                <span>
                  {STAGE_LABELS[p.stage as keyof typeof STAGE_LABELS] ?? p.stage} · {p.queries} quer
                  {p.queries === 1 ? 'y' : 'ies'}
                </span>
                <span className="font-mono text-faint">{formatMs(p.elapsed_ms)}</span>
              </li>
            ))}
        </ul>
      </details>
    </div>
  )
}
