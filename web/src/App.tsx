import { useState } from 'react'
import { IncidentFeed } from './components/IncidentFeed'
import { InvestigationView } from './components/InvestigationView'
import type { IncidentSummary } from './types'

function App() {
  const [selected, setSelected] = useState<IncidentSummary | null>(null)

  return selected ? (
    <InvestigationView incident={selected} onBack={() => setSelected(null)} />
  ) : (
    <IncidentFeed onSelect={setSelected} />
  )
}

export default App
