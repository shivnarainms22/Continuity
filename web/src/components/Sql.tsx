/** The anti-hallucination surface: every number in this product must be able to show
 * the SQL that produced it. A native <details> disclosure -- keyboard/focus behaviour
 * for free, zero extra state. */
export function Sql({ sql, label = 'View SQL' }: { sql: string; label?: string }) {
  return (
    <details className="group mt-1.5">
      <summary className="inline-flex cursor-pointer items-center gap-1 text-[11px] font-medium text-faint transition-colors hover:text-accent">
        <span className="inline-block transition-transform group-open:rotate-90">›</span>
        {label}
      </summary>
      <pre className="mt-2 max-h-64 overflow-auto rounded border border-hairline bg-bg p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-muted">
        {sql}
      </pre>
    </details>
  )
}
