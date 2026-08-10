import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { IncidentSeries, SeriesPoint } from '../types'
import { formatMetricValue, formatTime, parseUtc } from '../lib/format'

interface ChartPoint {
  bucket: string
  value: number | null
  expected: number | null
  lower: number | null
  band: number | null
}

/** Raw buckets sit on a 5-minute grid (`qoe_rollup_5m`); a multi-hour incident holds
 * hundreds of them, which reads as noise rather than a trend. Points are averaged up
 * to this granularity before rendering -- and the legend says so below, per this
 * task's "never silently smooth" requirement. */
const DISPLAY_BUCKET_MINUTES = 15

function average(values: (number | null)[]): number | null {
  const present = values.filter((v): v is number => v !== null)
  if (present.length === 0) return null
  return present.reduce((sum, v) => sum + v, 0) / present.length
}

function rawBucketMinutes(points: SeriesPoint[]): number | null {
  if (points.length < 2) return null
  return (parseUtc(points[1].bucket).getTime() - parseUtc(points[0].bucket).getTime()) / 60_000
}

/** Collapses `groupSize` consecutive raw points into one averaged display point.
 * Each group keeps its first point's own timestamp, so `nearestBucket` below can
 * still snap an anomaly window's edge onto a real, plotted x-axis category. */
function aggregate(points: SeriesPoint[], groupSize: number): ChartPoint[] {
  const out: ChartPoint[] = []
  for (let i = 0; i < points.length; i += groupSize) {
    const group = points.slice(i, i + groupSize)
    const lower = average(group.map((p) => p.lower))
    const upper = average(group.map((p) => p.upper))
    out.push({
      bucket: group[0].bucket,
      value: average(group.map((p) => p.value)),
      expected: average(group.map((p) => p.expected)),
      lower,
      band: lower !== null && upper !== null ? upper - lower : null,
    })
  }
  return out
}

/** Snaps an anomaly window's start/end onto the nearest point actually plotted, so
 * the highlighted band lines up with a real x-axis category after aggregation. */
function nearestBucket(points: ChartPoint[], iso: string): string {
  const target = parseUtc(iso).getTime()
  return points.reduce((best, p) => {
    const bestDelta = Math.abs(parseUtc(best).getTime() - target)
    const delta = Math.abs(parseUtc(p.bucket).getTime() - target)
    return delta < bestDelta ? p.bucket : best
  }, points[0]?.bucket ?? iso)
}

function ChartTooltip({
  active,
  payload,
  metric,
}: {
  active?: boolean
  payload?: { payload: ChartPoint }[]
  metric: string
}) {
  if (!active || !payload?.length) return null
  const point = payload[0].payload
  return (
    <div className="rounded border border-hairline bg-surface p-2.5 text-[11px] shadow-none">
      <div className="mb-1 font-mono text-faint">{formatTime(point.bucket)} UTC</div>
      {point.value !== null && (
        <div className="font-mono text-fg">value {formatMetricValue(metric, point.value)}</div>
      )}
      {point.expected !== null && (
        <div className="font-mono text-muted">expected {formatMetricValue(metric, point.expected)}</div>
      )}
      {point.value === null && point.expected === null && (
        <div className="text-faint">unmeasurable (insufficient comparison history)</div>
      )}
    </div>
  )
}

/** The one hero chart: the metric line, its seasonality-aware baseline as a muted
 * band, and the true (refined) anomaly windows highlighted in the one accent colour --
 * "we know what normal looks like, and here is exactly where it broke." */
export function HeroChart({ series, metricLabel }: { series: IncidentSeries; metricLabel: string }) {
  const rawMinutes = rawBucketMinutes(series.points)
  const groupSize =
    rawMinutes && rawMinutes > 0 && rawMinutes < DISPLAY_BUCKET_MINUTES
      ? Math.round(DISPLAY_BUCKET_MINUTES / rawMinutes)
      : 1
  const data = aggregate(series.points, groupSize)
  if (data.length === 0) return null
  const isAggregated = groupSize > 1

  return (
    <div className="w-full">
      {/* Fixed-height box for the chart only -- the legend below is normal flow, so
          it can wrap onto a second line (e.g. the aggregation note) without
          overflowing a fixed-height box and overlapping the next section. */}
      <div className="h-72 w-full">
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--color-hairline)" strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="bucket"
              tickFormatter={formatTime}
              stroke="var(--color-faint)"
              fontSize={11}
              fontFamily="var(--font-mono)"
              tickLine={false}
              axisLine={{ stroke: 'var(--color-hairline)' }}
              minTickGap={40}
            />
            <YAxis
              tickFormatter={(v: number) => formatMetricValue(series.metric, v)}
              stroke="var(--color-faint)"
              fontSize={11}
              fontFamily="var(--font-mono)"
              tickLine={false}
              axisLine={false}
              width={72}
            />
            <Tooltip content={<ChartTooltip metric={series.metric} />} />

            {/* Baseline band: an invisible base up to `lower`, then a filled band of
                height `upper - lower` stacked on top of it -- the standard Recharts
                range-band trick, since there is no first-class "area between two
                lines" primitive. Rendered BEFORE the anomaly highlight below so that
                highlight paints on top of this band instead of being muddied under a
                semi-opaque gray fill. */}
            <Area
              type="monotone"
              dataKey="lower"
              stackId="band"
              stroke="none"
              fill="transparent"
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="band"
              stackId="band"
              stroke="var(--color-muted)"
              strokeOpacity={0.5}
              strokeWidth={1}
              fill="var(--color-muted)"
              fillOpacity={0.22}
              isAnimationActive={false}
            />

            {series.anomaly_windows.map((w, i) => (
              <ReferenceArea
                key={i}
                x1={nearestBucket(data, w.start)}
                x2={nearestBucket(data, w.end)}
                fill="var(--color-accent)"
                fillOpacity={0.1}
                stroke="var(--color-accent)"
                strokeOpacity={0.55}
                strokeWidth={1}
              />
            ))}

            <Line
              type="monotone"
              dataKey="expected"
              stroke="var(--color-muted)"
              strokeWidth={1.75}
              strokeDasharray="4 3"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke="var(--color-fg)"
              strokeOpacity={0.85}
              strokeWidth={1.25}
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-4 text-[11px] text-faint">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-3 bg-fg/85" /> {metricLabel}
          {isAggregated ? ` (${DISPLAY_BUCKET_MINUTES}-min avg)` : ''}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-3 border-t-2 border-dashed border-muted" /> seasonality
          baseline
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 border border-muted/60 bg-muted/25" /> normal range
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 border border-accent/60 bg-accent/15" /> anomalous window
        </span>
        {isAggregated && (
          <span className="text-faint/80">
            {series.points.length} raw {rawMinutes}-min buckets averaged to {data.length} shown
          </span>
        )}
      </div>
    </div>
  )
}
