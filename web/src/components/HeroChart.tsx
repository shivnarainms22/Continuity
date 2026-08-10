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
import type { IncidentSeries } from '../types'
import { formatMetricValue, formatTime } from '../lib/format'

interface ChartPoint {
  bucket: string
  value: number | null
  expected: number | null
  lower: number | null
  band: number | null
}

function toChartPoints(series: IncidentSeries): ChartPoint[] {
  return series.points.map((p) => ({
    bucket: p.bucket,
    value: p.value,
    expected: p.expected,
    lower: p.lower,
    band: p.lower !== null && p.upper !== null ? p.upper - p.lower : null,
  }))
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
  const data = toChartPoints(series)
  if (data.length === 0) return null

  return (
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

          {series.anomaly_windows.map((w, i) => (
            <ReferenceArea
              key={i}
              x1={w.start}
              x2={w.end}
              fill="var(--color-accent)"
              fillOpacity={0.12}
              stroke="none"
            />
          ))}

          {/* Baseline band: an invisible base up to `lower`, then a filled band of
              height `upper - lower` stacked on top of it -- the standard Recharts
              range-band trick, since there is no first-class "area between two
              lines" primitive. */}
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
            stroke="none"
            fill="var(--color-muted)"
            fillOpacity={0.14}
            isAnimationActive={false}
          />

          <Line
            type="monotone"
            dataKey="expected"
            stroke="var(--color-muted)"
            strokeWidth={1.25}
            strokeDasharray="3 3"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="var(--color-fg)"
            strokeWidth={1.75}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="mt-2 flex items-center gap-4 text-[11px] text-faint">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-3 bg-fg" /> {metricLabel}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-3 border-t border-dashed border-muted" /> seasonality
          baseline
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 bg-muted opacity-30" /> normal range
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 bg-accent opacity-40" /> anomalous window
        </span>
      </div>
    </div>
  )
}
