import { useCallback, useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

// --- types mirroring continuity/api's JSON shapes exactly -- no client-side guessing ---

type Incident = {
  id: string
  window: { start: string; end: string }
  predicate: Record<string, string>
  is_decoy: boolean
}

type StageFrame = {
  stage: string
  elapsed_ms: number
  payload: Record<string, unknown>
}

type StageArrival = StageFrame & { arrivedAt: number }

type DoneFrame = {
  total_elapsed_ms: number
  brief: string
}

function App() {
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [stages, setStages] = useState<StageArrival[]>([])
  const [brief, setBrief] = useState<string | null>(null)
  const [streamError, setStreamError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/incidents')
      .then((res) => {
        if (!res.ok) throw new Error(`GET /api/incidents -> ${res.status}`)
        return res.json() as Promise<Incident[]>
      })
      .then(setIncidents)
      .catch((err: unknown) => setLoadError(String(err)))
  }, [])

  const investigate = useCallback((incidentId: string) => {
    setSelectedId(incidentId)
    setStages([])
    setBrief(null)
    setStreamError(null)
    setRunning(true)

    const source = new EventSource(`/api/investigate/${incidentId}/stream`)

    source.addEventListener('stage', (event) => {
      const data = JSON.parse((event as MessageEvent).data) as StageFrame
      setStages((prev) => [...prev, { ...data, arrivedAt: Date.now() }])
    })

    source.addEventListener('done', (event) => {
      const data = JSON.parse((event as MessageEvent).data) as DoneFrame
      setBrief(data.brief)
      setRunning(false)
      source.close()
    })

    source.addEventListener('error', () => {
      setStreamError('Investigation stream failed or the connection was lost.')
      setRunning(false)
      source.close()
    })
  }, [])

  return (
    <div className="min-h-screen bg-slate-950 p-6 font-mono text-sm text-slate-100 sm:p-10">
      <header className="mb-8">
        <h1 className="text-xl font-bold text-slate-50">Continuity -- walking skeleton</h1>
        <p className="mt-1 text-slate-400">
          Deterministic investigation pipeline only. No LLM calls anywhere on this page.
          This is a proof of integration, not a finished product.
        </p>
      </header>

      {loadError && (
        <div className="mb-6 border border-red-700 bg-red-950 p-3 text-red-300">
          Failed to load incidents: {loadError}
        </div>
      )}

      <section className="mb-10">
        <h2 className="mb-3 text-base font-semibold text-slate-200">
          Incidents ({incidents.length})
        </h2>
        <ul className="space-y-2">
          {incidents.map((incident) => (
            <li
              key={incident.id}
              className="flex items-center justify-between gap-4 border border-slate-700 bg-slate-900 p-3"
            >
              <div className="min-w-0">
                <div className="font-bold text-slate-100">
                  {incident.id}
                  {incident.is_decoy && (
                    <span className="ml-2 text-yellow-400">[decoy -- no real fault]</span>
                  )}
                </div>
                <div className="truncate text-xs text-slate-400">
                  {incident.window.start} -&gt; {incident.window.end} -- {JSON.stringify(incident.predicate)}
                </div>
              </div>
              <button
                type="button"
                className="shrink-0 border border-slate-500 px-3 py-1 text-slate-100 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
                disabled={running}
                onClick={() => investigate(incident.id)}
              >
                Investigate
              </button>
            </li>
          ))}
        </ul>
      </section>

      {selectedId && (
        <section>
          <h2 className="mb-3 text-base font-semibold text-slate-200">
            Stages -- {selectedId} {running && <span className="text-slate-400">(running...)</span>}
          </h2>

          {streamError && (
            <div className="mb-4 border border-red-700 bg-red-950 p-3 text-red-300">{streamError}</div>
          )}

          <ul className="mb-6 space-y-1">
            {stages.map((stage, i) => (
              <li key={i} className="text-slate-300">
                [{new Date(stage.arrivedAt).toLocaleTimeString()}] {stage.stage} --{' '}
                {stage.elapsed_ms.toFixed(1)}ms -- {JSON.stringify(stage.payload)}
              </li>
            ))}
          </ul>

          {stages.length > 0 && (
            <div className="mb-6 h-60 w-full border border-slate-700 bg-slate-900 p-2">
              <ResponsiveContainer>
                <BarChart data={stages.map((s) => ({ stage: s.stage, elapsed_ms: s.elapsed_ms }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="stage" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" fontSize={10} unit="ms" />
                  <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155' }} />
                  <Bar dataKey="elapsed_ms" fill="#38bdf8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {brief && (
            <pre className="max-h-96 overflow-auto whitespace-pre-wrap border border-slate-700 bg-slate-900 p-3 text-xs text-slate-300">
              {brief}
            </pre>
          )}
        </section>
      )}
    </div>
  )
}

export default App
