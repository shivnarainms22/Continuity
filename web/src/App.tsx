import { useState } from 'react'
import { AgentInvestigationView } from './components/AgentInvestigationView'
import { IncidentFeed } from './components/IncidentFeed'
import { InvestigationView } from './components/InvestigationView'
import type { IncidentSummary } from './types'

/** Which arm investigates. The AGENT is the product; the walker is the deterministic
 * control it is measured against (see results/comparison.json), kept reachable so the
 * comparison can be shown live rather than only asserted. */
type Arm = 'agent' | 'walker'

function App() {
  const [selected, setSelected] = useState<IncidentSummary | null>(null)
  const [arm, setArm] = useState<Arm>('agent')

  if (!selected) return <IncidentFeed onSelect={setSelected} />

  const back = () => setSelected(null)

  return (
    <div>
      <div className="mx-auto flex max-w-3xl items-center justify-end gap-1 px-6 pt-6">
        {(['agent', 'walker'] as const).map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => setArm(option)}
            className={`rounded px-2.5 py-1 text-[11px] font-medium transition-colors ${
              arm === option
                ? 'bg-accent-dim text-accent'
                : 'text-faint hover:text-fg'
            }`}
          >
            {option === 'agent' ? 'Gemini agent' : 'Deterministic control'}
          </button>
        ))}
      </div>
      {arm === 'agent' ? (
        <AgentInvestigationView key={`agent-${selected.id}`} incident={selected} onBack={back} />
      ) : (
        <InvestigationView key={`walker-${selected.id}`} incident={selected} onBack={back} />
      )}
    </div>
  )
}

export default App
