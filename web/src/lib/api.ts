import { useEffect, useRef, useState } from 'react'
import type { DoneFrame, IncidentSeverity, IncidentSummary, StageFrame } from '../types'

export async function fetchIncidents(): Promise<IncidentSummary[]> {
  const res = await fetch('/api/incidents')
  if (!res.ok) throw new Error(`GET /api/incidents -> ${res.status}`)
  return (await res.json()) as IncidentSummary[]
}

export async function fetchIncidentSeverity(id: string): Promise<IncidentSeverity> {
  const res = await fetch(`/api/incidents/${encodeURIComponent(id)}/severity`)
  if (!res.ok) throw new Error(`GET /api/incidents/${id}/severity -> ${res.status}`)
  return (await res.json()) as IncidentSeverity
}

export type StageArrival = StageFrame & { arrivedAt: number }

export interface InvestigationState {
  stages: StageArrival[]
  done: DoneFrame | null
  error: string | null
  running: boolean
}

/** Streams one incident's investigation over SSE. A fresh EventSource is opened
 * whenever `incidentId` changes, and closed on unmount or before the next one opens --
 * never more than one live connection per hook instance. */
export function useInvestigation(incidentId: string | null): InvestigationState {
  const [state, setState] = useState<InvestigationState>({
    stages: [],
    done: null,
    error: null,
    running: false,
  })
  const sourceRef = useRef<EventSource | null>(null)

  useEffect(() => {
    sourceRef.current?.close()
    if (!incidentId) {
      setState({ stages: [], done: null, error: null, running: false })
      return
    }

    setState({ stages: [], done: null, error: null, running: true })
    const source = new EventSource(`/api/investigate/${encodeURIComponent(incidentId)}/stream`)
    sourceRef.current = source

    source.addEventListener('stage', (event) => {
      const data = JSON.parse((event as MessageEvent).data) as StageFrame
      setState((prev) => ({ ...prev, stages: [...prev.stages, { ...data, arrivedAt: Date.now() }] }))
    })

    source.addEventListener('done', (event) => {
      const data = JSON.parse((event as MessageEvent).data) as DoneFrame
      setState((prev) => ({ ...prev, done: data, running: false }))
      source.close()
    })

    source.addEventListener('error', () => {
      setState((prev) =>
        prev.done ? prev : { ...prev, error: 'Investigation stream failed or was lost.', running: false },
      )
      source.close()
    })

    return () => {
      source.close()
      if (sourceRef.current === source) sourceRef.current = null
    }
  }, [incidentId])

  return state
}
