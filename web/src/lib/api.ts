import { useEffect, useRef, useState } from 'react'
import type {
  AgentDetectFrame,
  AgentDoneFrame,
  AgentStageFrame,
  AgentToolCallFrame,
  DoneFrame,
  IncidentSeverity,
  IncidentSummary,
  StageFrame,
} from '../types'

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

export type ToolCallArrival = AgentToolCallFrame & { arrivedAt: number }

export interface AgentInvestigationState {
  detect: AgentDetectFrame | null
  stage: AgentStageFrame | null
  stagesSeen: AgentStageFrame[]
  toolCalls: ToolCallArrival[]
  done: AgentDoneFrame | null
  error: string | null
  running: boolean
}

const EMPTY_AGENT_STATE: AgentInvestigationState = {
  detect: null,
  stage: null,
  stagesSeen: [],
  toolCalls: [],
  done: null,
  error: null,
  running: false,
}

/** Streams the Gemini investigation, appending each measurement as it arrives.
 *
 * Deliberately accumulates `toolCalls` rather than replacing state per stage: the
 * point of this view is the trail of what was measured and which query produced it,
 * so nothing that has arrived is ever dropped. One live EventSource per hook, closed
 * on unmount or before the next one opens -- an abandoned agent stream would keep a
 * real investigation running and spending tokens on the server. */
export function useAgentInvestigation(incidentId: string | null): AgentInvestigationState {
  const [state, setState] = useState<AgentInvestigationState>(EMPTY_AGENT_STATE)
  const sourceRef = useRef<EventSource | null>(null)

  useEffect(() => {
    sourceRef.current?.close()
    if (!incidentId) {
      setState(EMPTY_AGENT_STATE)
      return
    }

    setState({ ...EMPTY_AGENT_STATE, running: true })
    const source = new EventSource(
      `/api/investigate/${encodeURIComponent(incidentId)}/agent-stream`,
    )
    sourceRef.current = source

    source.addEventListener('detect', (event) => {
      const data = JSON.parse((event as MessageEvent).data) as AgentDetectFrame
      setState((prev) => ({ ...prev, detect: data }))
    })

    source.addEventListener('stage', (event) => {
      const data = JSON.parse((event as MessageEvent).data) as AgentStageFrame
      setState((prev) => ({ ...prev, stage: data, stagesSeen: [...prev.stagesSeen, data] }))
    })

    source.addEventListener('tool_call', (event) => {
      const data = JSON.parse((event as MessageEvent).data) as AgentToolCallFrame
      setState((prev) => ({
        ...prev,
        toolCalls: [...prev.toolCalls, { ...data, arrivedAt: Date.now() }],
      }))
    })

    source.addEventListener('done', (event) => {
      const data = JSON.parse((event as MessageEvent).data) as AgentDoneFrame
      setState((prev) => ({ ...prev, done: data, stage: null, running: false }))
      source.close()
    })

    source.addEventListener('error', () => {
      setState((prev) =>
        prev.done
          ? prev
          : { ...prev, error: 'Agent stream failed or was lost.', running: false },
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
