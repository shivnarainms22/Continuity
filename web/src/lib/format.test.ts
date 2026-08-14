// Regression coverage for the naive-timestamp bug: continuity/api emits naive
// ISO-8601 strings (no timezone suffix) that are UTC by this project's convention
// (see continuity/config.py). `new Date("2026-02-13T02:10:00")` parses a naive
// string as the HOST's local time, not UTC -- on a UTC host these tests would pass
// even with the bug, so this suite must be run with a non-UTC `TZ` (e.g.
// `TZ=America/Los_Angeles npm run test`) to actually exercise the fix.
import { describe, expect, it } from 'vitest'
import {
  formatDateRange,
  formatDateTime,
  formatDuration,
  formatTime,
  parseUtc,
  formatZScore,
  summarizeToolCall,
} from './format'

describe('parseUtc', () => {
  it('treats a naive ISO string as UTC, not host-local time', () => {
    const date = parseUtc('2026-02-13T02:10:00')
    expect(date.getUTCFullYear()).toBe(2026)
    expect(date.getUTCMonth()).toBe(1) // February, 0-indexed
    expect(date.getUTCDate()).toBe(13)
    expect(date.getUTCHours()).toBe(2)
    expect(date.getUTCMinutes()).toBe(10)
  })

  it('produces the same instant regardless of host timezone', () => {
    // A naive string must always resolve to the same epoch millisecond -- if the
    // host offset leaked in, this would fail whenever TZ is not UTC.
    expect(parseUtc('2026-02-13T02:10:00').getTime()).toBe(new Date('2026-02-13T02:10:00Z').getTime())
  })

  it('leaves a string that already has an offset untouched', () => {
    const iso = '2026-02-13T02:10:00+05:00'
    expect(parseUtc(iso).getTime()).toBe(new Date(iso).getTime())
  })
})

describe('formatDateTime', () => {
  it('renders the same wall-clock time the naive string contains, labelled UTC', () => {
    expect(formatDateTime('2026-02-13T02:10:00')).toBe('Feb 13, 2026, 2:10 AM UTC')
  })
})

describe('formatTime', () => {
  it('renders the naive wall-clock time without a host-offset shift', () => {
    expect(formatTime('2026-02-13T02:10:00')).toBe('2:10:00 AM')
  })
})

describe('formatDateRange', () => {
  it('header and body windows for the same incident render in agreement', () => {
    // Regression for the exact defect reported: the population-level header window
    // and the refined body window must never disagree because one path parsed a
    // naive string as local time and the other did not.
    const header = formatDateRange('2026-02-12T18:00:00', '2026-02-13T02:00:00')
    const body = formatDateRange('2026-02-13T02:10:00', '2026-02-13T09:55:00')
    expect(header).toBe('Feb 12, 2026, 6:00 PM UTC – Feb 13, 2026, 2:00 AM UTC')
    expect(body).toBe('Feb 13, 2026, 2:10 AM – 9:55 AM UTC')
  })
})

describe('formatDuration', () => {
  it('computes elapsed time from the underlying instant, independent of host timezone', () => {
    expect(formatDuration('2026-02-12T18:00:00', '2026-02-13T02:00:00')).toBe('8h')
  })
})

describe('summarizeToolCall', () => {
  it('reads the gated dimension out of a split, not merely the first one', () => {
    const summary = summarizeToolCall('split_all_dimensions', {
      dimensions: [
        { dimension: 'country', top_value: 'us', lift: 1.02, meets_lift_gate: false },
        { dimension: 'device_type', top_value: 'roku', lift: 4.41, meets_lift_gate: true },
      ],
    })
    expect(summary).toBe('device_type=roku explains 4.4x its size')
  })

  it('surfaces a tool error as the summary rather than hiding it', () => {
    expect(summarizeToolCall('measure_slice', { error: 'unknown dimension foo' })).toBe(
      'unknown dimension foo',
    )
  })

  it('says so plainly when find_changes corroborated nothing', () => {
    expect(summarizeToolCall('find_changes', { candidates: [], rejected: [1, 2] })).toBe(
      'no plausible change found · 2 rejected',
    )
  })

  it('returns null rather than a wrong headline when the shape is unrecognised', () => {
    expect(summarizeToolCall('split_all_dimensions', { unexpected: true })).toBeNull()
    expect(summarizeToolCall('some_new_tool', { anything: 1 })).toBeNull()
  })
})

describe('formatZScore', () => {
  it('renders a z-score as deviations, never as a multiple', () => {
    // "90.5x robust deviations" is the wrong unit -- a z-score counts deviations,
    // it is not a ratio, and the shipped UI briefly said exactly that.
    expect(formatZScore(90.5)).toBe('90.5σ')
    expect(formatZScore(7)).not.toContain('x')
  })
})
